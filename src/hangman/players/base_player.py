from abc import ABC, abstractmethod
from typing import List

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage

# --- Project-Specific Imports ---
from hangman.providers.llmprovider import LLMProvider


class BasePlayer(ABC):
    """
    An abstract base class that defines a standardized contract for any
    LLM-powered conversational partner that interacts with the main agent.

    This class represents a "role-player" (e.g., game host, collaborator)
    and is designed to be simpler than the BaseAgent. Its primary function
    is to generate a conversational turn based on the messages and a
    system prompt that defines its role.
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initializes the player with a configured LLM provider.

        Args:
            llm_provider: An initialized instance of LLMProvider.
        """
        self.llm_provider = llm_provider

    @abstractmethod
    def invoke(self, messages: List[BaseMessage], system_prompt: str) -> str:
        """
        Generates the player's next conversational turn.

        Args:
            messages: The current public conversation messages.
            system_prompt: A string that defines the player's role and
                           instructions for the current game.

        Returns:
            A single string containing the player's response.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets any internal state the player might have.

        This ensures a consistent interface and allows for clean restarts
        between experimental trials.
        """
        pass