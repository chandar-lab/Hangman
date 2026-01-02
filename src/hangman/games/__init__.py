from hangman.games.base_game import BaseGame
from hangman.games.hangman_sct import HangmanSCTGame
from hangman.games.diagnosis_simulator_sct import DiagnosisSimulatorSCTGame


def create_game(game_name: str):
    """
    Factory to instantiate a game class by name.

    Supported names (case-insensitive):
      - hangman_sct, hg_sct
      - diagnosis_simulator_sct, diagnosis_sct, ds_sct
    Returns: (game_instance, normalized_name)
    """
    name = (game_name or "hangman_sct").lower()
    if name in ("hangman_sct", "hg_sct", "hangman", "hg"):
        return HangmanSCTGame(), "hangman_sct"
    if name in ("diagnosis_simulator_sct", "diagnosis_sct", "ds_sct", "diagnosis", "diagnosis_simulator"):
        return DiagnosisSimulatorSCTGame(), "diagnosis_simulator_sct"
    raise ValueError(
        f"Unknown game '{game_name}'. Supported: hangman_sct, diagnosis_simulator_sct"
    )
