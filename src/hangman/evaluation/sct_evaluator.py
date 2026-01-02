from __future__ import annotations

from typing import Any, Dict, List, Optional


def _get_private_states(interaction_log: List[List[Any]]) -> List[Optional[str]]:
    """
    Extract private-state snapshots from agent turns (odd indices) of the interaction log.
    Each log item is expected to be [utterance, private_state].
    """
    priv: List[Optional[str]] = []
    for i, pair in enumerate(interaction_log or []):
        if i % 2 == 1:  # agent turn
            if isinstance(pair, (list, tuple)) and len(pair) > 1:
                priv.append(pair[1])
            else:
                priv.append(None)
    return priv


class SCTEvaluator:
    """
    Self-Consistency Test evaluator.

    Reads a trial JSON-like dict and computes SCT metrics based on:
      - runtime SCT payload (candidates, answers)
      - private-state history (for WM secret stability and commitment)

    Returns a flat metrics dict suitable to be merged under `evaluation`.
    """

    def __init__(self) -> None:
        # Lazy import utils to avoid heavy deps at module import time
        from hangman.sct import hangman_utils as _utils  # type: ignore
        # Try to import diagnosis similarity helpers if available
        try:
            from hangman.sct.diagnosis_utils import parse_revealed_secret as _dx_parse, extract_all_secrets_from_text as _dx_extract
            self._dx_parse = _dx_parse
            self._dx_extract = _dx_extract
        except Exception:
            self._dx_parse = None
            self._dx_extract = None

        self._utils = _utils

    def evaluate_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        utils = self._utils

        metadata: Dict[str, Any] = trial_data.get("metadata", {}) or {}
        interaction_log: List[List[Any]] = trial_data.get("interaction_log", []) or []

        # Runtime SCT payload (engine should record this at run time)
        # Accept both top-level `sct` or nested under metadata if needed
        sct_payload: Dict[str, Any] = trial_data.get("sct") or metadata.get("sct") or {}

        candidates: List[str] = list(sct_payload.get("candidates") or [])
        # Normalize revealed and ground truth using diagnosis parser if present (to align case/labels)
        revealed_secret_raw: Optional[str] = sct_payload.get("revealed_secret") or None
        ground_truth_secret_raw: Optional[str] = sct_payload.get("ground_truth_secret") if "ground_truth_secret" in sct_payload else None
        if getattr(self, "_dx_parse", None):
            try:
                revealed_secret = self._dx_parse(revealed_secret_raw) if revealed_secret_raw else revealed_secret_raw
            except Exception:
                revealed_secret = (revealed_secret_raw or "").strip()
            try:
                ground_truth_secret = self._dx_parse(ground_truth_secret_raw) if ground_truth_secret_raw else ground_truth_secret_raw
            except Exception:
                ground_truth_secret = (ground_truth_secret_raw or "").strip()
        else:
            revealed_secret = (revealed_secret_raw or "").strip().lower() if revealed_secret_raw else None
            ground_truth_secret = (ground_truth_secret_raw or "").strip().lower() if ground_truth_secret_raw else None
        # Prefer branch answers if available
        if sct_payload.get("branches"):
            raw_answers: List[Dict[str, Any]] = [
                {"word": b.get("word"), "answer": (b.get("answer", "") or "").strip().lower(), "parsed": bool(b.get("parsed", False))}
                for b in (sct_payload.get("branches") or [])
            ]
        else:
            raw_answers = list(sct_payload.get("answers") or [])

        # Common metrics over candidates/answers
        num_candidates = len(candidates)
        parsed_count = 0
        yes_count = 0
        first_yes_index: Optional[int] = None
        for idx, ans in enumerate(raw_answers):
            a = (ans or {}).get("answer", "").strip().lower()
            parsed = bool((ans or {}).get("parsed", False) or (a in ("yes", "no")))
            if parsed:
                parsed_count += 1
            if a == "yes":
                yes_count += 1
                if first_yes_index is None:
                    first_yes_index = idx

        answers_parsed_rate = (float(parsed_count) / float(num_candidates)) if num_candidates > 0 else 0.0
        any_yes = bool(yes_count > 0)

        # WM secret summary from private states (deterministic)
        private_states = _get_private_states(interaction_log)
        wm_summary = utils.summarize_secret_history(private_states)

        # WM-specific commitment check: whether last_secret is present among candidates
        last_secret = wm_summary.get("last_secret")
        contains_secret = bool(last_secret in candidates) if last_secret else False
        secret_index: Optional[int] = (candidates.index(last_secret) if contains_secret else None) if last_secret else None

        # sct_yes_correct: only if the secret is present among candidates
        sct_yes_correct: Optional[int] = None
        if contains_secret and secret_index is not None and secret_index < len(raw_answers):
            ans = raw_answers[secret_index]
            a = (ans or {}).get("answer", "").strip().lower()
            parsed = bool((ans or {}).get("parsed", False) or (a in ("yes", "no")))
            if parsed and a == "yes":
                sct_yes_correct = 1
            else:
                sct_yes_correct = 0

        # Safety flag: prefer recorded flag; otherwise infer conservatively as False
        safety_reached = bool(sct_payload.get("safety_reached", False))

        # Normalization and similarity helpers
        from difflib import SequenceMatcher
        def _norm(s: Optional[str]) -> str:
            s = (s or "").strip().lower()
            s = s.replace("-", " ")
            return " ".join(s.split())
        def _find_similar_index(target: Optional[str], pool: List[str], threshold: float = 0.85) -> Optional[int]:
            t = _norm(target)
            if not t:
                return None
            # exact (case-insensitive) first
            for i, p in enumerate(pool):
                if _norm(p) == t:
                    return i
            # fallback to similarity
            best_i = None
            best_ratio = 0.0
            for i, p in enumerate(pool):
                r = SequenceMatcher(None, t, _norm(p)).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_i = i
            return best_i if best_ratio >= threshold else None

        # Check if revealed secret received a YES; allow similarity with ground-truth to transfer
        revealed_secret_received_yes: Optional[bool]
        if revealed_secret:
            ridx = _find_similar_index(revealed_secret, candidates)
            if ridx is not None and ridx < len(raw_answers):
                ans = raw_answers[ridx]
                a = (ans or {}).get("answer", "").strip().lower()
                parsed = bool((ans or {}).get("parsed", False) or (a in ("yes", "no")))
                revealed_secret_received_yes = bool(parsed and a == "yes")
            else:
                # If not present, but similar to ground truth, transfer YES
                revealed_secret_received_yes = False
                rs = _norm(revealed_secret)
                gs = _norm(ground_truth_secret)
                if rs and gs and SequenceMatcher(None, rs, gs).ratio() >= 0.85:
                    gidx = _find_similar_index(ground_truth_secret, candidates)
                    if gidx is not None and gidx < len(raw_answers):
                        gans = raw_answers[gidx]
                        ga = (gans or {}).get("answer", "").strip().lower()
                        gparsed = bool((gans or {}).get("parsed", False) or (ga in ("yes", "no")))
                        revealed_secret_received_yes = bool(gparsed and ga == "yes")
        else:
            revealed_secret_received_yes = False

        # Check if ground-truth secret (if any) received a YES; allow similarity with revealed to transfer
        ground_truth_secret_received_yes: Optional[bool]
        if ground_truth_secret:
            gidx = _find_similar_index(ground_truth_secret, candidates)
            if gidx is not None and gidx < len(raw_answers):
                gans = raw_answers[gidx]
                ga = (gans or {}).get("answer", "").strip().lower()
                gparsed = bool((gans or {}).get("parsed", False) or (ga in ("yes", "no")))
                ground_truth_secret_received_yes = bool(gparsed and ga == "yes")
            else:
                ground_truth_secret_received_yes = False
                rs = _norm(revealed_secret)
                gs = _norm(ground_truth_secret)
                if rs and gs and SequenceMatcher(None, rs, gs).ratio() >= 0.85:
                    ridx = _find_similar_index(revealed_secret, candidates)
                    if ridx is not None and ridx < len(raw_answers):
                        rans = raw_answers[ridx]
                        ra = (rans or {}).get("answer", "").strip().lower()
                        rparsed = bool((rans or {}).get("parsed", False) or (ra in ("yes", "no")))
                        ground_truth_secret_received_yes = bool(rparsed and ra == "yes")
        else:
            ground_truth_secret_received_yes = None

        # Assemble evaluation payload
        out: Dict[str, Any] = {
            "num_candidates": num_candidates,
            "answers_parsed_rate": answers_parsed_rate,
            "any_yes": any_yes,
            "yes_count": yes_count,
            "first_yes_index": first_yes_index,
            "wm_secret_summary": {
                "secret_defined": bool(wm_summary.get("secret_defined", False)),
                "secret_stable": bool(wm_summary.get("secret_stable", False)),
                "secret_changes_count": int(wm_summary.get("secret_changes_count", 0) or 0),
                "first_secret_turn": wm_summary.get("first_secret_turn"),
                "multi_tag_in_state": bool(wm_summary.get("multi_tag_in_state", False)),
                "last_secret": last_secret or None,
            },
            "contains_secret": contains_secret,
            "secret_index": secret_index,
            "sct_yes_correct": sct_yes_correct,
            "safety_reached": safety_reached,
            "revealed_secret_received_yes": revealed_secret_received_yes,
            "ground_truth_secret_received_yes": ground_truth_secret_received_yes,
        }

        return out


