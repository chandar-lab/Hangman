from hangman.games.hangman import HangmanGame
from hangman.games.twenty_questions import TwentyQuestionsGame
from hangman.games.zendo import ZendoGame
from hangman.games.diagnosis_simulator import DiagnosisSimulatorGame


def create_game(game_name: str):
    """
    Factory to instantiate a game class by name.

    Supported names (case-insensitive):
      - hangman, hg
      - 20_questions, 20q, twenty_questions
      - zendo
      - diagnosis, diagnosis_simulator, meddiag, md
    Returns: (game_instance, normalized_name)
    """
    name = (game_name or "hangman").lower()
    if name in ("hangman", "hg"):
        return HangmanGame(), "hangman"
    if name in ("20_questions", "20q", "twenty_questions"):
        return TwentyQuestionsGame(), "twenty_questions"
    if name == "zendo":
        return ZendoGame(), "zendo"
    if name in ("diagnosis", "diagnosis_simulator", "meddiag", "md"):
        return DiagnosisSimulatorGame(), "diagnosis_simulator"
    raise ValueError(
        f"Unknown game '{game_name}'. Supported: hangman, 20_questions, zendo, diagnosis_simulator"
    )
