import os
import csv
import ast
import random
from typing import Dict, List, Optional, Set, Tuple

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import BaseMessage

# --- Project-Specific Imports ---
from hangman.players.base_player import BasePlayer
from hangman.prompts.diagnosis_simulator_sct import (
    PLAYER_START_PROMPT,
    SCT_YES_NO_PROMPT,
)


class DeterministicDiagnosisSimulatorPlayer(BasePlayer):
    """
    Deterministic scripted Diagnosis Simulator player for SCT.

    Behavior:
    - Turn 0: emit a fixed opening message that starts the diagnostic session.
    - Turns 1..(t_fork-1): emit yes/no questions sampled deterministically from the
      ddxplus evidences list, using a seeded information-seeking policy based on
      current consistent candidate conditions.
    - At any point after receiving the agent's answer to the previous question,
      call `update_with_agent_answer(answer_text)` to filter the candidate set:
        * If answer contains "yes" (case-insensitive), keep only conditions that
          include the last asked evidence code.
        * If answer contains "no" (case-insensitive), remove all conditions that
          include the last asked evidence code.

    Notes:
    - The player is deterministic given `random_seed` and `trial_index`.
    - The player is dataset-driven; it ignores `PLAYER_GUESS_PROMPT` and uses the
      ddxplus-provided natural language question text for each evidence.
    - Provide `final_guess(diagnosis)` to format the SCT yes/no hypothesis prompt.
    """

    def __init__(
        self,
        llm_provider=None,
        *,
        evidences_csv_path: str,
        conditions_csv_path: str,
        random_seed: int = 1337,
        trial_index: int = 0,
        t_fork: int = 6,
        temperature: float = 0.5,
    ) -> None:
        super().__init__(llm_provider)
        self.random_seed = int(random_seed)
        self.trial_index = int(trial_index or 0)
        self.t_fork = int(t_fork)
        # Temperature mixes uniform vs informativeness weighting when breaking ties
        self.temperature: float = max(0.0, min(1.0, float(temperature)))

        # Trial-aware RNG
        self.effective_seed: int = self.random_seed + self.trial_index
        self._rng: random.Random = random.Random(self.effective_seed)

        # Dataset paths
        self.evidences_csv_path = evidences_csv_path
        self.conditions_csv_path = conditions_csv_path

        # Data structures populated from CSVs
        # Evidence code -> question text
        self._evidence_code_to_question: Dict[str, str] = {}
        # Condition name -> set of related evidence codes
        self._condition_to_evidence_codes: Dict[str, Set[str]] = {}
        # Evidence code -> number of conditions (global support)
        self._evidence_support_global: Dict[str, int] = {}

        # Internal state for the trial
        self.n_turn: int = 0
        self._asked_codes: Dict[str, str] = {}  # code -> "yes"|"no"
        self._last_asked_code: Optional[str] = None
        self._candidate_conditions: List[str] = []

        self._load_dataset()
        self.reset()

    # -----------------
    # Dataset loading
    # -----------------
    def _load_dataset(self) -> None:
        # evidences.csv: columns: code_question, question_en
        if not os.path.exists(self.evidences_csv_path):
            raise FileNotFoundError(f"Evidences CSV not found: {self.evidences_csv_path}")
        with open(self.evidences_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code_question") or "").strip()
                q = (row.get("question_en") or "").strip()
                if code and q:
                    self._evidence_code_to_question[code] = q

        # conditions.csv: columns: condition_name, related (stringified python list of codes)
        if not os.path.exists(self.conditions_csv_path):
            raise FileNotFoundError(f"Conditions CSV not found: {self.conditions_csv_path}")
        with open(self.conditions_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cond = (row.get("condition_name") or "").strip()
                related_raw = (row.get("related") or "").strip()
                if not cond:
                    continue
                try:
                    codes_list = ast.literal_eval(related_raw) if related_raw else []
                    codes_set = {str(c).strip() for c in (codes_list or []) if str(c).strip()}
                except Exception:
                    codes_set = set()
                self._condition_to_evidence_codes[cond] = codes_set

        # Precompute global support counts per evidence
        support: Dict[str, int] = {}
        for codes in self._condition_to_evidence_codes.values():
            for code in codes:
                support[code] = support.get(code, 0) + 1
        self._evidence_support_global = support

    # -----------------
    # Public interface
    # -----------------
    def opening(self) -> str:
        return PLAYER_START_PROMPT

    def update_with_agent_answer(self, answer_text: Optional[str]) -> None:
        """
        Update candidate conditions based on the agent's answer to the last asked question.
        Interprets the text as positive if it contains 'yes' (case-insensitive),
        negative if it contains 'no'. If neither, no update is performed.
        """
        if not self._last_asked_code or not isinstance(answer_text, str):
            return
        t = answer_text.strip().lower()
        is_yes = ("yes" in t)
        is_no = ("no" in t) and not is_yes
        if not (is_yes or is_no):
            return

        code = self._last_asked_code
        self._asked_codes[code] = "yes" if is_yes else "no"

        codes_map = self._condition_to_evidence_codes
        before = len(self._candidate_conditions)
        if is_yes:
            # Keep only conditions that include the code
            self._candidate_conditions = [
                c for c in self._candidate_conditions if code in (codes_map.get(c) or set())
            ]
        else:
            # Remove conditions that include the code
            self._candidate_conditions = [
                c for c in self._candidate_conditions if code not in (codes_map.get(c) or set())
            ]
        # Ensure we never drop to empty set; if so, restore previous and ignore this update
        if len(self._candidate_conditions) == 0 and before > 0:
            self._candidate_conditions = [c for c in codes_map.keys()]

    def next_guess(self) -> str:
        """
        Choose the next evidence question deterministically conditioned on the
        current candidate set. Prefer evidence codes whose presence among
        candidate conditions is closest to 50% (maximizing information gain).

        Returns the natural-language question text from the dataset.
        """
        # Remaining evidence codes
        remaining_codes = [
            code for code in self._evidence_code_to_question.keys()
            if code not in self._asked_codes
        ]
        if not remaining_codes:
            # Fallback: ask a benign question if we've exhausted the list
            # Choose a random evidence deterministically to keep flow
            remaining_codes = list(self._evidence_code_to_question.keys())

        # Compute support over current candidates
        candidate_set = set(self._candidate_conditions)
        total = max(1, len(candidate_set))

        def support_in_candidates(code: str) -> int:
            count = 0
            for c in candidate_set:
                if code in (self._condition_to_evidence_codes.get(c) or set()):
                    count += 1
            return count

        # Score codes by closeness to half, break ties with weighted randomness
        scored: List[Tuple[str, float, int]] = []  # (code, abs_diff, support)
        for code in remaining_codes:
            supp = support_in_candidates(code)
            diff = abs(supp - (total / 2.0))
            scored.append((code, diff, supp))

        # Sort by ascending diff (closer to half is better)
        scored.sort(key=lambda x: x[1])

        # Take top-K near-best and sample deterministically with a weight that
        # mixes uniform and global support to stabilize choices
        top_k = min(5, len(scored))
        candidates = scored[:top_k]
        codes_only = [c[0] for c in candidates]
        if len(codes_only) == 1:
            chosen_code = codes_only[0]
        else:
            weights = []
            for code, _, _ in candidates:
                uniform_w = 1.0
                global_supp = float(self._evidence_support_global.get(code, 1))
                mixed = (1.0 - self.temperature) * uniform_w + self.temperature * global_supp
                weights.append(max(mixed, 1e-6))
            chosen_code = self._rng.choices(codes_only, weights=weights, k=1)[0]

        self._last_asked_code = chosen_code
        question = self._evidence_code_to_question.get(chosen_code) or f"Do you have symptom \"{chosen_code}\"?"
        return question

    def final_guess(self, diagnosis: Optional[str] = None) -> str:
        """
        Compose the SCT yes/no hypothesis using the provided (or inferred) diagnosis.
        If `diagnosis` is not provided, choose a plausible candidate from the
        current candidate set (first lexicographically as a deterministic fallback).
        """
        chosen = (diagnosis or self._choose_default_condition() or "influenza").strip()
        return SCT_YES_NO_PROMPT.format(diagnosis=chosen)

    def invoke(self, messages: List[BaseMessage], system_prompt: str) -> str:
        """
        Deterministic step based on `n_turn`:
        - 0  -> opening()
        - == t_fork -> final_guess() without argument (engine should pass a specific
          diagnosis via its own forked flow; here we fall back to inferred default)
        - else -> next_guess()
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
        # Reset RNG and internal state
        self._rng.seed(self.effective_seed)
        self._asked_codes.clear()
        self._last_asked_code = None
        self._candidate_conditions = list(self._condition_to_evidence_codes.keys())
        self.n_turn = 0

    # -----------------
    # Helpers
    # -----------------
    def get_used_features(self) -> Dict[str, str]:
        """Return a mapping of asked evidence codes to 'yes'/'no' labels."""
        return dict(self._asked_codes)

    def _choose_default_condition(self) -> Optional[str]:
        if not self._candidate_conditions:
            return None
        # Deterministic: choose lexicographically first
        return sorted(self._candidate_conditions)[0]

    def get_conditions_catalog(self) -> Dict[str, Set[str]]:
        """
        Return the condition -> set(evidence_codes) catalog loaded from ddxplus.
        The returned mapping is a shallow copy to prevent external mutation.
        """
        out: Dict[str, Set[str]] = {}
        for k, v in self._condition_to_evidence_codes.items():
            out[k] = set(v)
        return out


