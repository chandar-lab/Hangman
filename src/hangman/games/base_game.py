from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict

class BaseGame(ABC):
    """
    An abstract base class that defines a standardized contract for any game.

    This class adopts a "game as a log" philosophy. It is not responsible for
    enforcing game rules or determining win/loss conditions. Its sole purpose
    is to act as a chronological logger of the interactions, capturing the
    public utterances and the private state snapshots of the agent at each
    turn.

    The determination of the game's status (e.g., in_progress, win, loss) is
    delegated to an external component, such as an "LLM Judge," which analyzes
    the conversation history managed by this class.
    """

    def __init__(self):
        """
        Initializes the game by setting up an empty interaction log.
        The log will store tuples of (utterance, private_state_snapshot).
        """
        self.interaction_log: List[Tuple[str, Optional[str]]] = []

    @abstractmethod
    def reset(self) -> None:
        """
        Clears the interaction log to reset the game to a clean state.
        This is essential for running consecutive experimental trials.
        """
        self.interaction_log = []
        print("Game log has been reset.")

    @abstractmethod
    def update_state(self, utterance: str, private_state: Optional[str] = None) -> None:
        """
        Records a single turn by appending the utterance and an optional
        private state snapshot to the interaction log.

        Args:
            utterance: The public conversational message from the player or agent.
            private_state: An optional string representing the agent's internal
                           state (e.g., working memory) after producing the utterance.
                           This is typically None for players that are not the
                           primary agent being tested.
        """
        self.interaction_log.append((utterance, private_state))

    @abstractmethod
    def get_conversation_history(self) -> List[str]:
        """
        Retrieves the public-facing conversation history.

        Returns:
            A list of strings, where each string is an utterance from the log.
        """
        return [utterance for utterance, _ in self.interaction_log]

    @abstractmethod
    def get_full_state(self) -> List[Tuple[str, Optional[str]]]:
        """
        Retrieves the complete, unabridged interaction log for detailed
        analysis and evaluation.

        Returns:
            The entire interaction log, containing tuples of
            (utterance, private_state_snapshot).
        """
        return self.interaction_log