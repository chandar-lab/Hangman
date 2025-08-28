import sys
import os
import yaml
import argparse
import inspect
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.games import create_game
from hangman.players.llm_player import LLMPlayer
from hangman.engine import GameLoopController
from hangman.evaluation.judge import LLMJudge

# Import agent factory and classes
import hangman.agents as agents_pkg


# --- Helpers ---

def _instantiate_agent_from_spec(
    agent_spec: Dict[str, Any],
    providers_config_path: str,
    default_main: LLMProvider,
):
    """
    Create an agent instance from:
      - a dict:  {"ClassName": {kwargs...}} -> instantiate concrete class with kwargs
    """
    if isinstance(agent_spec, dict) and len(agent_spec) == 1:
        class_name, raw_kwargs = next(iter(agent_spec.items()))
        raw_kwargs = raw_kwargs or {}

        # Resolve class from agents package
        if not hasattr(agents_pkg, class_name):
            raise ValueError(f"Unknown agent class '{class_name}' in run config.")
        AgentClass = getattr(agents_pkg, class_name)

        # Auto-resolve any "*_llm_provider" entries from provider names to LLMProvider objects
        kwargs = {}
        for k, v in raw_kwargs.items():
            if k.endswith("_llm_provider") and isinstance(v, str):
                kwargs[k] = load_llm_provider(providers_config_path, v)
            else:
                kwargs[k] = v

        # Supply sensible defaults if constructor expects them but kwargs omitted
        sig = inspect.signature(AgentClass.__init__)
        params = sig.parameters

        if "llm_provider" in params and "llm_provider" not in kwargs:
            kwargs["llm_provider"] = default_main
        if "main_llm_provider" in params and "main_llm_provider" not in kwargs:
            kwargs["main_llm_provider"] = default_main
        if "responder_llm_provider" in params and "responder_llm_provider" not in kwargs:
            kwargs["responder_llm_provider"] = default_main
        if "updater_llm_provider" in params and "updater_llm_provider" not in kwargs:
            kwargs["updater_llm_provider"] = default_main
        
        # drop name from kwargs if it exists
        if "name" in kwargs:
            kwargs.pop("name")

        return AgentClass(**kwargs)

    raise ValueError(f"Invalid agent specification: {agent_spec}")


# --- Main Experiment Runner ---

def run_experiments(
    run_config_path: str = "./config/games_run.yaml",
    providers_config_path: str = "./config/config.yaml",
):
    """
    Main function to configure and run the batch of experiments.
    """
    # --- Experiment Configuration ---
    # Paths are provided via function arguments (CLI can override), with sensible defaults
    RUN_CONFIG_PATH = run_config_path
    PROVIDERS_CONFIG_PATH = providers_config_path

    # --- Load run configuration ---
    try:
        with open(RUN_CONFIG_PATH, "r") as f:
            run_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"❌ Run config not found at '{RUN_CONFIG_PATH}'. Create it or set RUN_CONFIG env var.")
        sys.exit(1)

    game_name = run_cfg.get("game", "hangman")
    agents_to_test = run_cfg.get("agents", ["ReActAgent"])
    num_trials = int(run_cfg.get("num_trials", 20))
    max_turns = int(run_cfg.get("max_turns", 20))
    base_results_dir = run_cfg.get("results_dir", f"results/{game_name}")

    # Evaluation configuration for LLMJudge
    judge_cfg = run_cfg.get("judge", {}) or {}
    eval_mode = judge_cfg.get("mode", "both")  # "memory" | "behavioral" | "both"
    judge_provider_name = judge_cfg.get("judge_llm_provider", None)
    metrics = run_cfg.get("metrics")  # Optional: ["intentionality", "secrecy", "mechanism", "coherence"]
    first_mover = run_cfg.get("first_mover", "player")

    # Provider defaults (for legacy/simple agent specs)
    providers = run_cfg.get("providers", {})
    MAIN_LLM_NAME = providers.get("main", "qwen3_14b_local_vllm_native")
    PLAYER_LLM_NAME = providers.get("player", MAIN_LLM_NAME)
    DEFAULT_JUDGE_LLM_NAME = providers.get("judge", MAIN_LLM_NAME)

    # 1. Load default/shared LLM Providers once
    print("--- 🧪 Initializing Experiment ---")
    try:
        main_llm_default = load_llm_provider(PROVIDERS_CONFIG_PATH, MAIN_LLM_NAME)
        player_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, PLAYER_LLM_NAME)
        judge_llm_name = judge_provider_name or DEFAULT_JUDGE_LLM_NAME
        judge_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, judge_llm_name)
        print("✅ All LLM Providers initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Providers: {e}. Exiting.")
        sys.exit(1)

    # 1.b Create selected game
    try:
        game_instance, normalized_game = create_game(game_name)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 2. Main Experiment Loop
    for agent_spec in agents_to_test:
        # Determine readable agent name for logging/results path
        if isinstance(agent_spec, str):
            agent_name = agent_spec
        elif isinstance(agent_spec, dict) and len(agent_spec) == 1:
            key, val = next(iter(agent_spec.items()))
            if isinstance(val, dict) and "name" in val:
                agent_name = val["name"]
            else:
                agent_name = key
        else:
            print(f"❌ Invalid agent spec in config: {agent_spec}. Skipping.")
            continue

        print(f"\n{'='*25} Starting Experiment Run for: {agent_name} {'='*25}")

        # Create a dedicated results directory for this agent
        agent_results_dir = os.path.join(base_results_dir, agent_name)
        os.makedirs(agent_results_dir, exist_ok=True)

        # --- RESUMABILITY LOGIC ---
        try:
            completed_trials = len([f for f in os.listdir(agent_results_dir) if f.endswith('.json')])
        except FileNotFoundError:
            completed_trials = 0

        if completed_trials >= num_trials:
            print(f"✅ Agent {agent_name} already has {completed_trials} trials completed. Skipping.")
            continue

        print(f"▶️  Found {completed_trials} existing trials. Starting from trial {completed_trials + 1}.")

        # Use tqdm for a progress bar over the trials
        needed_trials = num_trials - completed_trials
        for i in tqdm(range(needed_trials), desc=f"Agent: {agent_name}", unit="trial"):
            try:
                # 3. Instantiate fresh components for each trial
                agent = _instantiate_agent_from_spec(
                    agent_spec=agent_spec,
                    providers_config_path=PROVIDERS_CONFIG_PATH,
                    default_main=main_llm_default,
                )
                player = LLMPlayer(llm_provider=player_llm)
                # create a fresh game each trial to reset any internal prints
                game = create_game(normalized_game)[0]

                # Initialize LLMJudge for this game (prompts are selected per-call).
                llm_judge = LLMJudge(
                    judge_llm_provider=judge_llm,
                    game=normalized_game,
                    mode=eval_mode if isinstance(eval_mode, str) else "both",
                )

                # 4. Initialize and run the Game Loop Controller
                controller = GameLoopController(
                    agent=agent,
                    player=player,
                    game=game,
                    llm_judge=llm_judge,
                    max_turns=max_turns,
                    results_dir=agent_results_dir
                )
                controller.run(first_mover=first_mover, eval_mode=eval_mode, metrics=metrics)

            except Exception as e:
                print(f"\n--- ❌ ERROR on trial {i+1} for {agent_name}: {e} ---")
                # Log the error and continue to the next trial
                error_log_path = os.path.join(agent_results_dir, "error_log.txt")
                with open(error_log_path, "a") as f:
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Trial: {completed_trials+i+1}\nAgent: {agent_name}\n")
                    f.write(f"Error: {e}\n---\n")
                continue

    print(f"\n{'='*30} ✅ All Experiments Finished {'='*30}")
    print(f"Results saved in: {base_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hangman experiments from a YAML run config")
    parser.add_argument(
        "--run-config",
        "-r",
        default=os.environ.get("RUN_CONFIG", "./config/games_run.yaml"),
        help="Path to the YAML run configuration file (default: ./config/games_run.yaml or RUN_CONFIG env)",
    )
    parser.add_argument(
        "--providers-config",
        "-p",
        default="./config/config.yaml",
        help="Path to providers config YAML used to resolve LLM providers (default: ./config/config.yaml)",
    )
    args = parser.parse_args()

    run_experiments(run_config_path=args.run_config, providers_config_path=args.providers_config)