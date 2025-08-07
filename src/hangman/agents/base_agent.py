from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

# Assuming LLMProvider and ModelOutput are in this location
from hangman.providers.llmprovider import LLMProvider, ModelOutput

# --- Abstract Base Agent ---

class BaseAgent(ABC):
    """
    An abstract base class for all agents in the experiment.

    It defines a standard interface for initialization, invocation, state retrieval,
    and resetting. All concrete agent implementations must inherit from this class.
    """
    def __init__(self, llm_provider: LLMProvider):
        """
        Initializes the agent with a configured LLM provider.

        Args:
            llm_provider: An initialized instance of LLMProvider.
        """
        self.llm_provider = llm_provider
        self.workflow: StateGraph = self._build_workflow()

    @abstractmethod
    def _build_workflow(self) -> StateGraph:
        """
        Constructs and returns the agent's internal LangGraph workflow.
        This method must be implemented by all subclasses.

        Returns:
            A compiled StateGraph instance.
        """
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """
        Runs the agent for a single turn.

        This is the primary entry point for interacting with the agent. It takes
        the current conversation messages and returns the model's output.

        Args:
            messages: The current public conversation messages.

        Returns:
            A ModelOutput dictionary containing the 'response' and 'thinking' trace.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the agent's current internal state.
        
        This is crucial for logging and evaluation, allowing inspection of the
        agent's messages and private working memory.

        Returns:
            The current Agent's State dictionary.
        """
        pass

    @abstractmethod
    def get_private_state(self) -> str:
        """
        Retrieves the agent's current internal state and returns it as a string to be passed to the Engine.
        
        This is crucial for logging and evaluation, allowing inspection of the
        agent's messages and private working memory.

        Returns:
            The current Agent's Private State as a string.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the agent's internal state and working memory to its initial condition.
        This is called between experimental runs to ensure a clean start.
        """
        pass