from typing import List, Tuple, Optional

# --- Project-Specific Imports ---
# Import the abstract base class for games.
from hangman.games.base_game import BaseGame

# Import the game-specific prompts. This file is expected to exist.
from hangman.prompts.hangman import AGENT_START_PROMPT, PLAYER_START_PROMPT

class HangmanGame(BaseGame):
    """
    A concrete implementation of BaseGame for the game of Hangman.

    This class adheres to the "game as a log" principle. It does not contain
    any game logic (like tracking lives or the secret word). Its primary
    purpose is to serve as a configuration holder for a Hangman game,
    providing the initial prompts required by the agent and player, while
    relying on the parent BaseGame class for all logging functionality.
    """

    @property
    def name(self) -> str:
        """A string identifier for the game."""
        return "hangman"

    def __init__(self):
        """
        Initializes the Hangman game by loading its specific prompts and
        setting up the interaction log via the parent class.
        """
        super().__init__()
        self.agent_start_prompt: str = AGENT_START_PROMPT
        self.player_start_prompt: str = PLAYER_START_PROMPT
        print("HangmanGame initialized with specific prompts.")

    def reset(self) -> None:
        """
        Resets the game by clearing the interaction log.
        """
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