from typing import List, Tuple, Optional

# --- Project-Specific Imports ---
from hangman.games.base_game import BaseGame
from hangman.prompts.zendo import AGENT_START_PROMPT, PLAYER_START_PROMPT


class ZendoGame(BaseGame):
    """
    A concrete implementation of BaseGame for the text-based Zendo game.

    This class follows the same "game as a log" approach as HangmanGame and
    TwentyQuestionsGame. It does not implement game mechanics; it only provides
    the initial prompts and relies on the BaseGame to record public utterances
    and optional private state snapshots per turn.
    """

    @property
    def name(self) -> str:
        """A string identifier for the game."""
        return "zendo"

    def __init__(self):
        """Initialize Zendo with its specific prompts and a fresh log."""
        super().__init__()
        self.agent_start_prompt: str = AGENT_START_PROMPT
        self.player_start_prompt: str = PLAYER_START_PROMPT
        print("ZendoGame initialized with specific prompts.")

    def reset(self) -> None:
        """Resets the game by clearing the interaction log."""
        super().reset()

    def update_state(self, utterance: str, private_state: Optional[str] = None) -> None:
        """
        Records a single turn in the interaction log.

        Args:
            utterance: The public conversational message.
            private_state: An optional string of the agent's internal state.
        """
        super().update_state(utterance, private_state)

    def get_conversation_messages(self) -> List[str]:
        """
        Retrieves the public-facing conversation history.

        Returns:
            A list of all utterances from the game log.
        """
        return super().get_conversation_messages()

    def get_full_state(self) -> List[Tuple[str, Optional[str]]]:
        """
        Retrieves the complete interaction log for analysis.

        Returns:
            The entire log, containing (utterance, private_state) tuples.
        """
        return super().get_full_state()
