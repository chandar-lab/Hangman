import os
import random
from typing import List, Optional, Set

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage

# --- Project-Specific Imports ---
from hangman.players.base_player import BasePlayer
from hangman.prompts.hangman_sct import (
    PLAYER_START_PROMPT,
    PLAYER_GUESS_PROMPT,
    SCT_YES_NO_PROMPT,
)

LETTER_FREQUENCIES = {
    'E': 12.02, 
    'T': 9.10, 
    'A': 8.12, 
    'O': 7.68, 
    'I': 7.31, 
    'N': 6.95, 
    'S': 6.28, 
    'R': 6.02, 
    'H': 5.92, 
    'D': 4.32, 
    'L': 3.98, 
    'U': 2.88, 
    'C': 2.71, 
    'M': 2.61, 
    'F': 2.30, 
    'Y': 2.11, 
    'W': 2.09, 
    'G': 2.03, 
    'P': 1.82, 
    'B': 1.49, 
    'V': 1.11, 
    'K': 0.69, 
    'X': 0.17, 
    'Q': 0.11, 
    'J': 0.10, 
    'Z': 0.07, 
}

class DeterministicHangmanPlayer(BasePlayer):
    """
    Deterministic scripted Hangman player for SCT.

    Behavior:
    - Turn 0: emit a fixed opening message that starts the game.
    - Turns 1..(t_fork-1): emit single-letter guesses chosen by a seeded policy
      (currently a random permutation of the alphabet driven by `random_seed`).
    - Turn == t_fork: emit a final yes/no hypothesis prompt using the provided
      final guess word.

    Notes:
    - This player is deterministic given `random_seed` and `t_fork`.
    - BasePlayer requires `invoke(messages, system_prompt) -> str`; the
      `system_prompt` is unused here because the player is scripted.
    - Set the `final_guess_word` via constructor or the setter before
      the `t_fork` turn.
    """

    def __init__(
        self,
        llm_provider=None,
        *,
        random_seed: int = 1337,
        t_fork: int = 6,
        final_guess_word: Optional[str] = None,
        trial_index: int = 0,
        temperature: float = 0.8,
    ) -> None:
        super().__init__(llm_provider)
        self.random_seed = int(random_seed)
        self.t_fork = int(t_fork)
        self._final_guess_word: Optional[str] = final_guess_word
        # Mix factor for frequency-driven vs uniform sampling: 0.0 -> uniform, 1.0 -> pure frequency
        self.temperature: float = max(0.0, min(1.0, float(temperature)))

        # Internal state
        # Trial-aware RNG: vary per trial deterministically for same settings
        self.trial_index: int = int(trial_index or 0)
        self.effective_seed: int = self.random_seed + self.trial_index
        self._rng: random.Random = random.Random(self.effective_seed)
        self._letters_order: List[str] = self._generate_letter_order()
        self._used_letters: Set[str] = set()
        self._letter_idx: int = 0
        self.n_turn: int = 0  # how many times invoke() has been called

    # -----------------
    # Public interface
    # -----------------
    def set_final_guess_word(self, word: Optional[str]) -> None:
        self._final_guess_word = (word or None)

    def opening(self) -> str:
        return PLAYER_START_PROMPT

    def next_guess(self) -> str:
        # Frequency-aware, temperature-controlled sampling over remaining letters
        remaining = [ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch not in self._used_letters]
        if not remaining:
            # Should not happen; fallback to 'e'
            letter = 'e'
        else:
            # Build mixed weights: w = (1 - temp) * uniform + temp * freq
            # Normalize LETTER_FREQUENCIES to lowercase mapping
            freq_map = {k.lower(): float(v) for k, v in LETTER_FREQUENCIES.items()}
            # To avoid zero-probability, give a small epsilon for missing keys
            epsilon = 0.0001
            weights = []
            for ch in remaining:
                freq_w = freq_map.get(ch, epsilon)
                uniform_w = 1.0
                mixed = (1.0 - self.temperature) * uniform_w + self.temperature * freq_w
                weights.append(max(mixed, epsilon))
            # Sample deterministically via our RNG
            letter = self._rng.choices(remaining, weights=weights, k=1)[0]
        self._used_letters.add(letter)
        return PLAYER_GUESS_PROMPT.format(letter=letter)

    def final_guess(self, word: Optional[str] = None) -> str:
        # Default to configured final_guess_word; fallback to a safe dummy
        chosen = word or self._final_guess_word or "hangman"
        return SCT_YES_NO_PROMPT.format(word=chosen)

    def invoke(self, messages: List[BaseMessage], system_prompt: str) -> str:
        """
        Deterministic step based on `n_turn`:
        - 0  -> opening()
        - == t_fork -> final_guess()
        - else -> next_guess()
        Increments `n_turn` after producing the output.
        """
        if self.n_turn == 0:
            out = self.opening()
        elif self.n_turn == self.t_fork:
            out = self.final_guess()
        else:
            out = self.next_guess()

        self.n_turn += 1
        return out

    def reset(self) -> None:
        # Preserve trial-differentiated effective seed across resets
        self._rng.seed(self.effective_seed)
        self._letters_order = self._generate_letter_order()
        self._used_letters.clear()
        self._letter_idx = 0
        self.n_turn = 0

    # -----------------
    # Internal helpers
    # -----------------
    def get_used_letters(self) -> Set[str]:
        return set(self._used_letters)

    def _generate_letter_order(self) -> List[str]:
        # Currently: seeded random permutation of the lowercase alphabet
        letters = list("abcdefghijklmnopqrstuvwxyz")
        self._rng.shuffle(letters)
        return letters


