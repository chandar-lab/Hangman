import sys
import os
import yaml
import argparse
import inspect
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import cycle

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.games import create_game
from hangman.players.llm_player import LLMPlayer
from hangman.engine import GameLoopController
from hangman.evaluation.hybrid_evaluator import HybridEvaluator

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


# --- Worker entry point for parallel trial execution ---
def _run_trial_job(
    *,
    run_config: Dict[str, Any],
    providers_config_path: str,
    agent_spec: Any,
    game_name: str,
    max_turns: int,
    agent_results_dir: str,
    eval_mode: Any,
    metrics: Any,
    first_mover: str,
    main_provider_name: str,
    player_provider_name: str,
    judge_provider_name: str,
):
    try:
        # Load providers for this worker
        main_llm = load_llm_provider(providers_config_path, main_provider_name)
        player_llm = load_llm_provider(providers_config_path, player_provider_name)
        judge_llm = load_llm_provider(providers_config_path, judge_provider_name)

        # Instantiate components for this trial
        agent = _instantiate_agent_from_spec(
            agent_spec=agent_spec,
            providers_config_path=providers_config_path,
            default_main=main_llm,
        )
        player = LLMPlayer(llm_provider=player_llm)
        game = create_game(game_name)[0]
        evaluator = HybridEvaluator(
            judge_llm_provider=judge_llm,
            game=game_name,
            mode=eval_mode if isinstance(eval_mode, str) else "both",
        )

        controller = GameLoopController(
            agent=agent,
            player=player,
            game=game,
            evaluator=evaluator,
            max_turns=max_turns,
            results_dir=agent_results_dir,
        )
        controller.run(first_mover=first_mover, eval_mode=eval_mode, metrics=metrics)
        return True
    except Exception:
        return False


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

    # Evaluation configuration for evaluator (behavioral | memory | rule_based | both | all)
    evaluator_cfg = run_cfg.get("evaluator", {}) or {}
    eval_mode = evaluator_cfg.get("mode", "both")  # "memory" | "behavioral" | "both" | "rule_based" | "all"
    judge_provider_name = evaluator_cfg.get("judge_llm_provider", None)
    metrics = run_cfg.get("metrics")  # Optional: ["intentionality", "secrecy", "mechanism", "coherence"]
    first_mover = run_cfg.get("first_mover", "player")

    # Provider defaults and optional provider pools (for parallel runs)
    providers = run_cfg.get("providers", {})
    MAIN_LLM_NAME = providers.get("main", "qwen3_14b_local_vllm_native")
    PLAYER_LLM_NAME = providers.get("player", MAIN_LLM_NAME)
    DEFAULT_JUDGE_LLM_NAME = providers.get("judge", MAIN_LLM_NAME)

    # Optional provider pools; if specified, they enable multi-process execution
    MAIN_POOL = providers.get("main_pool") or []
    PLAYER_POOL = providers.get("player_pool") or []
    JUDGE_POOL = providers.get("judge_pool") or []
    # When pools are absent, fall back to single providers

    # Concurrency knob (Option B): synthesize pools from single provider names
    # if explicit pools are not provided.
    def _to_int(value: Any, default: int) -> int:
        try:
            iv = int(value)
            return iv if iv > 0 else default
        except Exception:
            return default

    concurrency = _to_int(providers.get("concurrency", 0), 0)
    judge_concurrency = _to_int(providers.get("judge_concurrency", 0), 0)

    if not MAIN_POOL and concurrency > 0:
        MAIN_POOL = [MAIN_LLM_NAME for _ in range(concurrency)]
    if not PLAYER_POOL and concurrency > 0:
        PLAYER_POOL = [PLAYER_LLM_NAME for _ in range(concurrency)]
    if not JUDGE_POOL and (judge_concurrency > 0 or concurrency > 0):
        eff = judge_concurrency if judge_concurrency > 0 else concurrency
        JUDGE_POOL = [(providers.get("judge", DEFAULT_JUDGE_LLM_NAME)) for _ in range(eff)]

    # 1. Load default/shared LLM Providers once (only if not using pools)
    print("--- 🧪 Initializing Experiment ---")
    if not MAIN_POOL:
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

        # --- CLEANUP/RESUME: remove incomplete logs without evaluation or with errors in the interaction log ---
        def _is_complete_log(filepath: str) -> bool:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                # Consider complete if it has an evaluation block with results
                is_dict = isinstance(data, dict) 
                has_eval = isinstance(data.get("evaluation", {}).get("results"), dict)
                has_interaction_log = isinstance(data.get("interaction_log", []), list)
                responses = [x[0] for x in data.get("interaction_log", [])]
                has_errors = any([response.startswith('Error: ') for response in responses])
                return is_dict and has_eval and has_interaction_log and not has_errors
            except Exception:
                return False

        try:
            for fname in os.listdir(agent_results_dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(agent_results_dir, fname)
                if not _is_complete_log(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        # Best-effort cleanup; ignore deletion failures
                        pass
        except FileNotFoundError:
            pass

        # --- RESUMABILITY LOGIC ---
        try:
            completed_trials = 0
            for f in os.listdir(agent_results_dir):
                if not f.endswith('.json'):
                    continue
                fpath = os.path.join(agent_results_dir, f)
                # Count only complete logs
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        data = yaml.safe_load(fp)
                    if isinstance(data, dict) and isinstance(data.get("evaluation", {}).get("results"), dict):
                        completed_trials += 1
                except Exception:
                    # Skip unreadable files
                    continue
        except FileNotFoundError:
            completed_trials = 0

        if completed_trials >= num_trials:
            print(f"✅ Agent {agent_name} already has {completed_trials} trials completed. Skipping.")
            continue

        print(f"▶️  Found {completed_trials} existing trials. Starting from trial {completed_trials + 1}.")

        needed_trials = num_trials - completed_trials

        # If MAIN_POOL is configured, run trials in parallel across providers
        if MAIN_POOL:
            judge_pool = JUDGE_POOL if JUDGE_POOL else [judge_provider_name or DEFAULT_JUDGE_LLM_NAME]
            player_pool = PLAYER_POOL if PLAYER_POOL else [PLAYER_LLM_NAME]

            main_cycle = cycle(MAIN_POOL)
            player_cycle = cycle(player_pool)
            judge_cycle = cycle(judge_pool)

            max_workers = len(MAIN_POOL)
            print(f"▶️  Parallel execution enabled with {max_workers} workers and provider pool: {MAIN_POOL}")

            jobs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in range(needed_trials):
                    jobs.append(
                        executor.submit(
                            _run_trial_job,
                            run_config=run_cfg,
                            providers_config_path=PROVIDERS_CONFIG_PATH,
                            agent_spec=agent_spec,
                            game_name=normalized_game,
                            max_turns=max_turns,
                            agent_results_dir=agent_results_dir,
                            eval_mode=eval_mode,
                            metrics=metrics,
                            first_mover=first_mover,
                            main_provider_name=next(main_cycle),
                            player_provider_name=next(player_cycle),
                            judge_provider_name=next(judge_cycle),
                        )
                    )

                # Progress bar for completions
                for _ in tqdm(as_completed(jobs), total=len(jobs), desc=f"Agent: {agent_name}", unit="trial"):
                    pass
        else:
            # Sequential path (legacy)
            for i in tqdm(range(needed_trials), desc=f"Agent: {agent_name}", unit="trial"):
                try:
                    agent = _instantiate_agent_from_spec(
                        agent_spec=agent_spec,
                        providers_config_path=PROVIDERS_CONFIG_PATH,
                        default_main=main_llm_default,
                    )
                    player = LLMPlayer(llm_provider=player_llm)
                    game = create_game(normalized_game)[0]
                    evaluator = HybridEvaluator(
                        judge_llm_provider=judge_llm,
                        game=normalized_game,
                        mode=eval_mode if isinstance(eval_mode, str) else "both",
                    )
                    controller = GameLoopController(
                        agent=agent,
                        player=player,
                        game=game,
                        evaluator=evaluator,
                        max_turns=max_turns,
                        results_dir=agent_results_dir
                    )
                    controller.run(first_mover=first_mover, eval_mode=eval_mode, metrics=metrics)

                except Exception as e:
                    print(f"\n--- ❌ ERROR on trial {i+1} for {agent_name}: {e} ---")
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