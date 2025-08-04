import os
import sys
import json
import yaml
import datetime
from typing import List, Dict, Any

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent
from hangman.agents.readispatact_agentagent import ReaDisPatActAgent
from hangman.players.base_player import BasePlayer
from hangman.games.base_game import BaseGame
from hangman.games.hangman import HangmanGame
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.players.llm_player import LLMPlayer

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- LangChain Core Imports ---
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


class GameLoopController:
    """
    Orchestrates a single game interaction between an agent and a player.

    This engine is designed to be generic. It takes fully instantiated agent,
    player, and game objects and manages the turn-by-turn flow. It is
    responsible for logging the entire interaction to a JSON file.
    """

    def __init__(
        self,
        agent: BaseAgent,
        player: BasePlayer,
        game: BaseGame,
        judge_llm_provider: LLMProvider, # Reserved for future use
        max_turns: int = 20,
        results_dir: str = "results"
    ):
        """
        Initializes the controller via dependency injection.

        Args:
            agent: An instantiated object that adheres to the BaseAgent interface.
            player: An instantiated object that adheres to the BasePlayer interface.
            game: An instantiated object that adheres to the BaseGame interface.
            judge_llm_provider: An LLM provider reserved for the game judge.
            max_turns: The maximum number of turns before the game is halted.
            results_dir: The base directory to save JSON log files.
        """
        self.agent = agent
        self.player = player
        self.game = game
        self.judge_llm_provider = judge_llm_provider
        self.max_turns = max_turns
        self.results_dir = results_dir
        self.log_filepath: str = ""

    def _prepare_log_file(self) -> None:
        """Creates the results directory and defines a unique log file path."""
        game_results_dir = os.path.join(self.results_dir, self.game.name)
        os.makedirs(game_results_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        agent_name = self.agent.__class__.__name__
        self.log_filepath = os.path.join(game_results_dir, f"{agent_name}_{timestamp}.json")
        print(f"Logging results to: {self.log_filepath}")

    def _write_log(self, status: str) -> None:
        """
        Writes the current state of the game to the log file.
        This method is called after every turn to ensure data is not lost.

        Args:
            status: The current status of the game (e.g., 'IN_PROGRESS').
        """
        log_data = {
            "metadata": {
                "game": self.game.name,
                "agent_class": self.agent.__class__.__name__,
                "player_class": self.player.__class__.__name__,
                "agent_llm": self.agent.llm_provider.config,
                "player_llm": self.player.llm_provider.config,
                "max_turns": self.max_turns,
                "game_status": status,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "interaction_log": self.game.get_full_state()
        }
        with open(self.log_filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

    def _is_game_over(self) -> bool:
        """
        Checks if the game has reached a terminal state (Win/Loss).

        TODO: This method will be implemented using the 'judge_llm_provider'
        to analyze the conversation messages and determine the game's state.
        For now, it relies solely on the max_turns limit.
        """
        # raise NotImplementedError("The LLM Judge logic has not been implemented yet.")
        return False

    def run(self, first_mover: str = "player") -> None:
        """
        Executes a single, complete game from start to finish.

        Args:
            first_mover: Determines who makes the first move, 'player' or 'agent'.
        """
        # 1. Reset all components for a clean run
        self.agent.reset()
        self.player.reset()
        self.game.reset()

        # 2. Prepare logging and initial state
        self._prepare_log_file()
        messages: List[BaseMessage] = []
        turn_count = 0
        
        # Initial log before any turns
        self._write_log(status="STARTED")
        print(f"\n--- Starting New Game: {self.game.name} ---")
        print(f"Agent: {self.agent.__class__.__name__}, Player: {self.player.__class__.__name__}")
        print(f"First mover: {first_mover}")

        # 3. Main Game Loop
        while turn_count < self.max_turns and not self._is_game_over():
            print(f"\n--- Turn {turn_count + 1} ---")
            
            # Determine whose turn it is
            current_actor_is_player = (turn_count % 2 == 0 and first_mover == "player") or \
                                      (turn_count % 2 != 0 and first_mover == "agent")

            if current_actor_is_player:
                print("Player's turn...")
                player_utterance = self.player.invoke(messages, system_prompt=self.game.player_start_prompt)
                print(f"Player: {player_utterance}")
                # The player acts as the 'user', so its message is a HumanMessage.
                messages.append(HumanMessage(content=player_utterance))
                self.game.update_state(utterance=player_utterance, private_state=None)
            else:
                print("Agent's turn...")
                # Assuming the agent needs a get_private_state() method
                # This may raise an AttributeError if not implemented on the agent.
                try:
                    agent_output = self.agent.invoke(messages)
                    agent_utterance = agent_output["response"]
                    private_state_str = self.agent.get_private_state()
                    
                    print(f"Agent: {agent_utterance}")
                    # The agent is the 'AI', so its own message is an AIMessage.
                    messages.append(AIMessage(content=agent_utterance))
                    self.game.update_state(utterance=agent_utterance, private_state=private_state_str)
                except Exception as e:
                    print(f"An error occurred during agent invocation: {e}")
                    self._write_log(status="ERROR_AGENT_INVOKE")
                    return # End game on error
            
            # Live-log after every single action
            self._write_log(status="IN_PROGRESS")
            turn_count += 1
        
        # 4. Finalize
        final_status = "COMPLETED_MAX_TURNS" if not self._is_game_over() else "COMPLETED_GAME_OVER"
        self._write_log(status=final_status)
        print(f"\n--- Game Finished: {final_status} ---")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # --- Configuration ---
    CONFIG_PATH = "config.yaml"
    RESULTS_DIR = "results/hangman/test"

    # Define the names of the LLM providers to use from the config file
    # NOTE: You may need to change these names to match your config.yaml
    AGENT_MAIN_LLM = "qwen3_14b_local"
    AGENT_DISTILL_LLM = "qwen3_14b_local"
    PLAYER_LLM = "qwen3_14b_local"
    JUDGE_LLM = "qwen3_14b_local" 

    print("--- 🧪 Starting Test Run ---")

    # 1. Load LLM Provider Configurations from YAML
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from '{CONFIG_PATH}'")
    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{CONFIG_PATH}'. Exiting.")
        sys.exit(1)

    # 2. Initialize LLM Providers
    try:
        agent_main_llm = load_llm_provider(CONFIG_PATH, provider_name=AGENT_MAIN_LLM)
        agent_distill_llm = load_llm_provider(CONFIG_PATH, provider_name=AGENT_DISTILL_LLM)
        player_llm = load_llm_provider(CONFIG_PATH, provider_name=PLAYER_LLM)
        judge_llm = load_llm_provider(CONFIG_PATH, provider_name=JUDGE_LLM)
        print("✅ All LLM Providers initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Providers: {e}")
        sys.exit(1)

    # 3. Instantiate Game, Player, and Agent
    print("🚀 Instantiating components...")
    game = HangmanGame()
    player = LLMPlayer(llm_provider=player_llm)
    agent = ReaDisPatActAgent(
        main_llm_provider=agent_main_llm,
        distillation_llm_provider=agent_distill_llm
    )
    
    print("✅ Components instantiated.")
    print("...Patched `get_private_state` method onto agent instance for logging.")


    # 4. Initialize the Game Loop Controller
    controller = GameLoopController(
        agent=agent,
        player=player,
        game=game,
        judge_llm_provider=judge_llm,
        max_turns=12,  # A game of hangman shouldn't take more than ~12 turns
        results_dir=RESULTS_DIR
    )
    print("✅ GameLoopController is ready.")

    # 5. Run the experiment
    try:
        controller.run(first_mover="player")
    except Exception as e:
        print(f"\n🚨 An error occurred during the game loop: {e}")
        print("   Please ensure all components are correctly configured and servers are running.")

    print("\n--- ✅ Test Run Finished ---")