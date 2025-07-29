import sys
import os
import yaml
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.players.base_player import BasePlayer


class LLMPlayer(BasePlayer):
    """
    A concrete implementation of BasePlayer for an LLM-powered conversational
    partner. This player can adopt any role defined by a system prompt.

    It maintains its own internal history of the conversation, which is
    synchronized with the history provided by the GameLoopController during
    the invoke call.
    """

    def __init__(self, llm_provider: LLMProvider, system_prompt: str = "You are a helpful assistant.") -> None:
        """
        Initializes the LLMPlayer with a configured LLM provider and an
        empty internal history.

        Args:
            llm_provider: An initialized instance of LLMProvider.
        """
        super().__init__(llm_provider)
        self.history: List[BaseMessage] = []
        self.system_prompt = system_prompt

    def invoke(self, history: List[BaseMessage], system_prompt: str) -> str:
        """
        Generates the player's next conversational turn by calling the LLM.

        It uses the official history from the game loop for the API call to
        ensure it is always in sync, and then updates its internal history.

        Args:
            history: The current public conversation history from the game loop.
            system_prompt: A string that defines the player's role.

        Returns:
            A single string containing the player's response.
        """
        print("\n--- LLM PLAYER TURN ---")
        # The messages for the LLM call are constructed with the system prompt
        # and the official history from the game loop.
        if system_prompt:
            system_message = SystemMessage(content=system_prompt)
        else:
            system_message = SystemMessage(content=self.system_prompt)

        messages = [system_message] + history

        # Invoke the LLM provider. This implementation doesn't need "thinking".
        model_output = self.llm_provider.invoke(messages, thinking=False)
        response_text = model_output["response"]

        # Update its own internal history state for logging or debugging.
        self.history = history + [AIMessage(content=response_text)]

        return response_text

    def reset(self) -> None:
        """
        Resets the player's internal history to an empty list.
        """
        self.history = []
        print("LLMPlayer state and history reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    print("--- Starting LLMPlayer CLI Test ---")
    
    # --- Configuration Loading ---
    CONFIG_PATH = "config.yaml"
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at '{CONFIG_PATH}'. Make sure you are running this from the project root.")
        sys.exit(1)

    # --- Initialize LLM Provider ---
    try:
        # Change 'qwen3_14b_local' to a provider name from your config.yaml
        llm_provider = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        print(f"✅ LLM Provider '{llm_provider.config['name']}' loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        sys.exit(1)

    # --- Initialize Player and Game Context ---
    player = LLMPlayer(llm_provider=llm_provider)
    
    # This prompt tells the LLMPlayer how to behave for the test.
    test_system_prompt = (
        "You are a funny assistant"
    )
    
    # The history list for the conversation.
    history = []
    
    print("\nLLM Player is ready.")
    print("Type 'quit', 'exit', or 'q' to end the session.")
    print("-" * 50)

    # --- Main Interaction Loop ---
    while True:
        user_input = input("You > ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending session.")
            break
        
        # Add user message to history
        history.append(HumanMessage(content=user_input))
        
        # Get the player's response
        try:
            player_response = player.invoke(history, test_system_prompt)
        except Exception as e:
            print(f"\n🚨 An error occurred during player invocation: {e}")
            print("Please ensure your LLM server (e.g., vLLM) is running.")
            break
            
        # Add player's response to history for the next turn's context
        history.append(AIMessage(content=player_response))

        print(f"\nHost > {player_response}")