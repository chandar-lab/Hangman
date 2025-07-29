from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

# Assuming LLMProvider and ModelOutput are in this location
from hangman.providers.llmprovider import LLMProvider, ModelOutput


# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the complete state of an agent at any point in time.
    This structure is used by the LangGraph state machine.
    """
    # The public conversation history
    history: List[BaseMessage]
    
    # The agent's private, internal knowledge and scratchpad
    working_memory: Dict[str, Any]
    
    # The public-facing response for the current turn
    response: str
    
    # The private thinking trace for the current turn
    thinking: str

    # The diff of changes made to the working memory
    diff: str


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
    def invoke(self, history: List[BaseMessage]) -> ModelOutput:
        """
        Runs the agent for a single turn.

        This is the primary entry point for interacting with the agent. It takes
        the current conversation history and returns the model's output.

        Args:
            history: The current public conversation history.

        Returns:
            A ModelOutput dictionary containing the 'response' and 'thinking' trace.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> AgentState:
        """
        Retrieves the agent's current internal state.
        
        This is crucial for logging and evaluation, allowing inspection of the
        agent's history and private working memory.

        Returns:
            The current AgentState dictionary.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the agent's internal state and working memory to its initial condition.
        This is called between experimental runs to ensure a clean start.
        """
        pass