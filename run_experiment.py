import sys
import os
import yaml
from tqdm import tqdm

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.games.hangman import HangmanGame
from hangman.players.llm_player import LLMPlayer
from hangman.engine import GameLoopController
from hangman.agents.base_agent import BaseAgent

# Import all agent classes to be tested
from hangman.agents import create_agent

# --- Main Experiment Runner ---

def run_experiments():
    """
    Main function to configure and run the batch of experiments.
    """
    # --- Experiment Configuration ---
    CONFIG_PATH = "config.yaml"
    BASE_RESULTS_DIR = "results/hangman"
    NUM_TRIALS = 20
    MAX_TURNS = 20
    
    AGENTS_TO_TEST = [
        "ReActAgent",
        "ReaKeeActAgent",
        "ReaDisOveActAgent",
        "ReaDisPatActAgent",
        "ReaDisUpdActAgent",
    ]
    
    # LLM provider names from your config.yaml
    MAIN_LLM_NAME = "qwen3_14b_local"
    DISTILL_LLM_NAME = "qwen3_14b_local"
    PLAYER_LLM_NAME = "qwen3_14b_local"
    JUDGE_LLM_NAME = "qwen3_14b_local"
    
    # 1. Load LLM Providers Once
    print("--- 🧪 Initializing Experiment ---")
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name=MAIN_LLM_NAME)
        distill_llm = load_llm_provider(CONFIG_PATH, provider_name=DISTILL_LLM_NAME)
        player_llm = load_llm_provider(CONFIG_PATH, provider_name=PLAYER_LLM_NAME)
        judge_llm = load_llm_provider(CONFIG_PATH, provider_name=JUDGE_LLM_NAME)
        print("✅ All LLM Providers initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Providers: {e}. Exiting.")
        sys.exit(1)

    # 2. Main Experiment Loop
    for agent_name in AGENTS_TO_TEST:
        print(f"\n{'='*25} Starting Experiment Run for: {agent_name} {'='*25}")
        
        # Create a dedicated results directory for this agent
        agent_results_dir = os.path.join(BASE_RESULTS_DIR, agent_name)
        os.makedirs(agent_results_dir, exist_ok=True)

        # --- RESUMABILITY LOGIC ---
        try:
            completed_trials = len([f for f in os.listdir(agent_results_dir) if f.endswith('.json')])
        except FileNotFoundError:
            completed_trials = 0

        if completed_trials >= NUM_TRIALS:
            print(f"✅ Agent {agent_name} already has {completed_trials} trials completed. Skipping.")
            continue
        
        print(f"▶️  Found {completed_trials} existing trials. Starting from trial {completed_trials + 1}.")
        
        # Use tqdm for a progress bar over the trials
        needed_trials = NUM_TRIALS - completed_trials
        for i in tqdm(range(needed_trials), desc=f"Agent: {agent_name}", unit="trial"):
            try:
                # 3. Instantiate fresh components for each trial
                agent = create_agent(agent_name, main_llm, distill_llm)
                player = LLMPlayer(llm_provider=player_llm)
                game = HangmanGame()
                
                # 4. Initialize and run the Game Loop Controller
                controller = GameLoopController(
                    agent=agent,
                    player=player,
                    game=game,
                    judge_llm_provider=judge_llm,
                    max_turns=MAX_TURNS,
                    results_dir=agent_results_dir
                )
                controller.run(first_mover="player")
            
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
    print(f"Results saved in: {BASE_RESULTS_DIR}")

if __name__ == "__main__":
    from datetime import datetime
    run_experiments()


