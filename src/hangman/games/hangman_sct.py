from typing import List, Tuple, Optional

# --- Project-Specific Imports ---
from hangman.games.base_game import BaseGame
from hangman.prompts.hangman_sct import (
    PLAYER_START_PROMPT,
    PLAYER_GUESS_PROMPT,
)


class HangmanSCTGame(BaseGame):
    """
    Self-Consistency Test (SCT) variant of the Hangman game.

    This class adheres to the "game as a log" principle. It does not contain
    any gameplay mechanics. Its primary purpose is to provide SCT-specific
    prompts for a deterministic scripted player and to log each turn as
    (utterance, private_state) pairs for downstream analysis.
    """

    @property
    def name(self) -> str:
        """A string identifier for the game."""
        return "hangman_sct"

    def __init__(self):
        """
        Initializes the SCT game by loading its specific prompts and setting up
        the interaction log via the parent class.
        """
        super().__init__()
        # The agent plays the host; we do not require an agent start prompt here.
        self.agent_start_prompt: str = ""
        self.player_start_prompt: str = PLAYER_START_PROMPT
        # Expose the guess prompt template for deterministic players (optional convenience)
        self.player_guess_prompt_template: str = PLAYER_GUESS_PROMPT
        print("HangmanSCTGame initialized with SCT prompts.")

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


