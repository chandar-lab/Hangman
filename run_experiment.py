import sys
import os
import yaml
from datetime import datetime
from tqdm import tqdm

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.games import create_game
from hangman.players.llm_player import LLMPlayer
from hangman.engine import GameLoopController
from hangman.agents.base_agent import BaseAgent
from hangman.evaluation.judge import LLMJudge

# Import all agent classes to be tested
from hangman.agents import create_agent

# --- Main Experiment Runner ---

def run_experiments():
    """
    Main function to configure and run the batch of experiments.
    """
    # --- Experiment Configuration ---
    PROVIDERS_CONFIG_PATH = "config.yaml"
    RUN_CONFIG_PATH = os.environ.get("RUN_CONFIG", "games_run.yaml")

    # --- Load run configuration ---
    try:
        with open(RUN_CONFIG_PATH, "r") as f:
            run_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"❌ Run config not found at '{RUN_CONFIG_PATH}'. Create it or set RUN_CONFIG env var.")
        sys.exit(1)

    game_name = run_cfg.get("game", "hangman")
    agents_to_test = run_cfg.get("agents", ["ReaDisPatActAgent"])  # default to your main agent
    num_trials = int(run_cfg.get("num_trials", 20))
    max_turns = int(run_cfg.get("max_turns", 20))
    base_results_dir = run_cfg.get("results_dir", f"results/{game_name}")

    # Evaluation configuration for LLMJudge
    eval_modes = run_cfg.get("eval_modes", "both")  # "memory" | "behavioral" | "both" | [..]
    metrics = run_cfg.get("metrics")  # Optional: ["intentionality", "secrecy", "mechanism", "coherence"]
    first_mover = run_cfg.get("first_mover", "player")

    # Provider names defined in config.yaml
    providers = run_cfg.get("providers", {})
    MAIN_LLM_NAME = providers.get("main", "qwen3_14b_local")
    DISTILL_LLM_NAME = providers.get("distill", MAIN_LLM_NAME)
    PLAYER_LLM_NAME = providers.get("player", MAIN_LLM_NAME)
    JUDGE_LLM_NAME = providers.get("judge", MAIN_LLM_NAME)
    
    # 1. Load LLM Providers Once
    print("--- 🧪 Initializing Experiment ---")
    try:
        main_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, provider_name=MAIN_LLM_NAME)
        distill_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, provider_name=DISTILL_LLM_NAME)
        player_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, provider_name=PLAYER_LLM_NAME)
        judge_llm = load_llm_provider(PROVIDERS_CONFIG_PATH, provider_name=JUDGE_LLM_NAME)
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
    for agent_name in agents_to_test:
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
                agent = create_agent(agent_name, main_llm, distill_llm)
                player = LLMPlayer(llm_provider=player_llm)
                # create a fresh game each trial to reset any internal prints
                game = create_game(normalized_game)[0]

                # Initialize LLMJudge for this game. Mode here is informational; prompts are selected per-call.
                llm_judge = LLMJudge(
                    judge_llm_provider=judge_llm,
                    game=normalized_game,
                    mode="behavioral",
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
                controller.run(first_mover=first_mover, eval_modes=eval_modes, metrics=metrics)
            
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
    run_experiments()


