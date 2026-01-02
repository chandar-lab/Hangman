from typing import List, Tuple, Optional

# --- Project-Specific Imports ---
from hangman.games.base_game import BaseGame
from hangman.prompts.diagnosis_simulator_sct import (
    PLAYER_START_PROMPT,
)


class DiagnosisSimulatorSCTGame(BaseGame):
    """
    Self-Consistency Test (SCT) variant of the Diagnosis Simulator game.

    This mirrors the Hangman SCT game wrapper: it exposes domain-specific prompts
    and manages a chronological interaction log of (utterance, private_state) pairs.
    """

    @property
    def name(self) -> str:
        return "diagnosis_simulator_sct"

    def __init__(self):
        super().__init__()
        self.agent_start_prompt: str = ""
        self.player_start_prompt: str = PLAYER_START_PROMPT
        print("DiagnosisSimulatorSCTGame initialized with SCT prompts.")

    def reset(self) -> None:
        super().reset()

    def update_state(self, utterance: str, private_state: Optional[str] = None) -> None:
        super().update_state(utterance, private_state)

    def get_conversation_messages(self) -> List[str]:
        return super().get_conversation_messages()

    def get_full_state(self) -> List[Tuple[str, Optional[str]]]:
        return super().get_full_state()





