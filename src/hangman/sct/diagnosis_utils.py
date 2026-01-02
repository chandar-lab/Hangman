from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import BaseMessage, HumanMessage

from hangman.providers.llmprovider import LLMProvider
from hangman.prompts.diagnosis_simulator_sct import (
    CANDIDATE_GENERATION_PROMPT,
    SECRET_EXTRACTION_PROMPT,
)


# -----------------------------
# Secret extraction (from WM)
# -----------------------------

_SECRET_TAG_RE_TEMPLATE = r"<\s*{tag}\s*>([\s\S]*?)<\s*/\s*{tag}\s*>"


def extract_all_secrets_from_text(private_state_text: str, tag: str = "secret") -> List[str]:
    """
    Return all occurrences of <secret>...</secret> (case-insensitive on tag, whitespace tolerant)
    found in a single private-state string, preserving order of appearance.
    """
    if not private_state_text:
        return []
    pattern = re.compile(_SECRET_TAG_RE_TEMPLATE.format(tag=re.escape(tag)), flags=re.IGNORECASE)
    secrets: List[str] = []
    for m in pattern.finditer(private_state_text):
        val = (m.group(1) or "").strip()
        if not val:
            continue
        # If the model prepends a label like "Hidden condition: ...", split only when the left side looks like a label
        if ":" in val:
            try:
                left, right = val.split(":", 1)
                if re.search(r"\b(hidden|condition|hidden\s+condition)\b", left, flags=re.IGNORECASE):
                    val = right.strip()
            except Exception:
                pass
        if val:
            secrets.append(val)
    return secrets


def extract_last_secret(private_state_text: str, tag: str = "secret") -> Optional[str]:
    """
    Return the last non-empty <secret>...</secret> in the given text, or None if none exists.
    """
    secrets = extract_all_secrets_from_text(private_state_text, tag=tag)
    return secrets[-1] if secrets else None


def summarize_secret_history(private_states: Sequence[Optional[str]], tag: str = "secret") -> Dict[str, Any]:
    """
    Walk the sequence of private-state snapshots (chronological) and compute summary stats:
    - secrets_by_turn: list[(turn_idx, secret)]
    - unique_secrets: sorted list of unique non-empty secrets
    - last_secret: last observed secret or None
    - secret_defined: bool
    - secret_stable: bool (no changes across turns)
    - secret_changes_count: number of changes between adjacent occurrences
    - first_secret_turn: first turn index where any secret was observed, or None
    - multi_tag_in_state: True if any single snapshot included multiple <tag> tags
    """
    secrets_by_turn: List[Tuple[int, str]] = []
    unique: List[str] = []
    secret_changes_count = 0
    first_secret_turn: Optional[int] = None
    multi_tag_in_state = False

    prev: Optional[str] = None
    for idx, st in enumerate(private_states):
        all_here = extract_all_secrets_from_text(st or "", tag=tag)
        if len(all_here) > 1:
            multi_tag_in_state = True
        if all_here:
            cur = all_here[-1]
            secrets_by_turn.append((idx, cur))
            if cur not in unique:
                unique.append(cur)
            if first_secret_turn is None:
                first_secret_turn = idx
            if prev is not None and cur != prev:
                secret_changes_count += 1
            prev = cur

    last_secret = secrets_by_turn[-1][1] if secrets_by_turn else None
    secret_defined = last_secret is not None
    secret_stable = (len(unique) <= 1)

    return {
        "secrets_by_turn": secrets_by_turn,
        "unique_secrets": unique,
        "last_secret": last_secret,
        "secret_defined": secret_defined,
        "secret_stable": secret_stable,
        "secret_changes_count": secret_changes_count,
        "first_secret_turn": first_secret_turn,
        "multi_tag_in_state": multi_tag_in_state,
    }


def summarize_secret_history_with_llm(
    private_states: Sequence[Optional[str]],
    *,
    llm_provider: Optional[LLMProvider],
    tag: str = "secret",
) -> Dict[str, Any]:
    """
    LLM fallback flow for secret extraction:
    - Ask LLM to extract a JSON array of diagnosis labels in order of first appearance.
    - Use the returned list to construct a synthetic sequence of private-state tags for deterministic summarization.
    - Then call the deterministic summarizer over that synthetic list to compute the same metadata shape.
    If LLM is unavailable or parsing fails, fall back to deterministic summarization over original states.
    """
    if llm_provider is None:
        return summarize_secret_history(private_states, tag=tag)

    try:
        # Prepare concatenated private states as text for the prompt
        blocks: List[str] = []
        for i, st in enumerate(private_states):
            if not st:
                continue
            blocks.append(f"[STATE {i}]\n{st}")
        joined = "\n\n".join(blocks)

        prompt = SECRET_EXTRACTION_PROMPT.format(private_states=joined)
        resp = llm_provider.invoke([HumanMessage(content=prompt)], thinking=False)
        text = resp.get("response", "")

        # Expect a JSON array of strings (multi-word allowed)
        secrets_list: List[str] = []
        arr = json.loads(text)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    w = item.strip().lower()
                    # allow multi-word conditions; require at least one letter
                    if w and re.search(r"[a-zA-Z]", w):
                        secrets_list.append(w)

        synthetic_states: List[str] = []
        for s in secrets_list:
            synthetic_states.append(f"<{tag}>{s}</{tag}>")

        return summarize_secret_history(synthetic_states, tag=tag)
    except Exception:
        return summarize_secret_history(private_states, tag=tag)


# --------------------------------------
# Transcript formatting for LLM prompts
# --------------------------------------

def format_interaction_log(log: Sequence[Sequence[Any]]) -> str:
    """
    Best-effort formatting of the interaction log into a readable, turn-by-turn string.
    Each item in log is [utterance, private_state] or similar.
    Private states are intentionally excluded from this formatted transcript.
    """
    parts: List[str] = []
    for i, pair in enumerate(log):
        turn_no = i + 1
        utter = pair[0] if isinstance(pair, (list, tuple)) and len(pair) > 0 else ""
        actor = "Player" if (i % 2 == 0) else "Agent"
        parts.append(f"--- TURN {turn_no} ({actor}) ---")
        if actor == "Player":
            parts.append("PLAYER UTTERANCE:")
        else:
            parts.append("AGENT UTTERANCE:")
        parts.append(str(utter))
        parts.append("")
    return "\n".join(parts)


# --------------------------------------
# Reveal parsing (diagnosis variant)
# --------------------------------------

def parse_revealed_secret(text: str) -> Optional[str]:
    """
    Parse a direct reveal answer. Accept multi-word diagnosis strings (letters, spaces, hyphens).
    Return a normalized lowercase diagnosis if present, else None.
    """
    if not text:
        return None
    t = (text or "").strip().lower()
    # Strip surrounding quotes/backticks
    if (t.startswith("`") and t.endswith("`")) or (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    # Keep letters, digits, spaces, and hyphens; collapse whitespace
    # Ensure at least one alphanumeric remains
    # First, extract the longest span containing [A-Za-z0-9 -]
    m = re.search(r"([A-Za-z0-9][A-Za-z0-9\-\s]*[A-Za-z0-9])", t)
    if not m:
        # Allow single alphanumeric token
        m = re.search(r"[A-Za-z0-9]", t)
        if not m:
            return None
        frag = m.group(0)
    else:
        frag = m.group(1)
    frag = re.sub(r"\s+", " ", frag).strip()
    return frag if frag else None


# --------------------------------------
# Candidate generation
# --------------------------------------

def estimate_candidates_from_transcript(
    transcript_text: str,
    *,
    n: int,
    method: str,
    dictionary_path: Optional[str] = None,
    llm_provider: Optional[LLMProvider] = None,
    llm_max_n: Optional[int] = None,
    used_features: Optional[Dict[str, str]] = None,
    conditions_catalog: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
    """
    Estimate up to n candidate diagnosis labels consistent with the transcript.

    - method == 'deterministic':
        * Requires a `conditions_catalog` mapping: condition -> set(evidence_codes)
        * Applies constraints from `used_features` if provided (evidence_code -> 'yes'/'no').
        * If `used_features` is None, returns [] (controller should pass player's state).
    - method == 'llm':
        * Use CANDIDATE_GENERATION_PROMPT and the provided provider; parse JSON array and return up to n labels.
    """
    if n <= 0:
        return []

    method_key = (method or "").strip().lower()
    if method_key == "deterministic":
        if not conditions_catalog:
            return []
        if not used_features:
            return []
        return _estimate_candidates_deterministic_catalog(
            n=n,
            used_features=used_features,
            conditions_catalog=conditions_catalog,
        )

    if method_key == "llm":
        return _estimate_candidates_llm(transcript_text, n=n, llm_provider=llm_provider, llm_max_n=llm_max_n)

    return []


def _estimate_candidates_deterministic_catalog(
    *,
    n: int,
    used_features: Dict[str, str],
    conditions_catalog: Dict[str, Set[str]],
) -> List[str]:
    # Apply yes/no constraints to filter conditions
    yes_codes = {code for code, lab in used_features.items() if (lab or "").strip().lower() == "yes"}
    no_codes = {code for code, lab in used_features.items() if (lab or "").strip().lower() == "no"}

    def is_consistent(cond: str) -> bool:
        codes = conditions_catalog.get(cond) or set()
        # Must include all yes codes
        if any((c not in codes) for c in yes_codes):
            return False
        # Must exclude all no codes
        if any((c in codes) for c in no_codes):
            return False
        return True

    filtered: List[str] = [c for c in conditions_catalog.keys() if is_consistent(c)]
    if len(filtered) <= n:
        return filtered

    # Deterministic prioritization: more yes-code coverage first, fewer no-code overlaps, then lexicographic
    def score(cond: str) -> Tuple[int, int, str]:
        codes = conditions_catalog.get(cond) or set()
        yes_cov = sum(1 for c in yes_codes if c in codes)
        no_cov = sum(1 for c in no_codes if c in codes)
        return (-yes_cov, no_cov, cond)

    filtered.sort(key=score)
    return filtered[:n]


def _estimate_candidates_llm(
    transcript_text: str,
    *,
    n: int,
    llm_provider: Optional[LLMProvider],
    llm_max_n: Optional[int],
) -> List[str]:
    if llm_provider is None:
        return []
    try:
        max_n = int(llm_max_n) if llm_max_n is not None else max(n, 10)
    except Exception:
        max_n = max(n, 10)

    prompt = CANDIDATE_GENERATION_PROMPT.format(transcript=transcript_text, max_n=max_n)
    resp = llm_provider.invoke([HumanMessage(content=prompt)], thinking=False)
    text = resp.get("response", "")

    candidates: List[str] = []
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    w = item.strip().lower()
                    # allow multi-word; require at least one letter
                    if w and re.search(r"[a-zA-Z]", w):
                        candidates.append(w)
    except Exception:
        # Best-effort: try to locate the first JSON array in text
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            try:
                arr = json.loads(text[s : e + 1])
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, str):
                            w = item.strip().lower()
                            if w and re.search(r"[a-zA-Z]", w):
                                candidates.append(w)
            except Exception:
                pass

    # Deduplicate and cap to n
    seen: Set[str] = set()
    out: List[str] = []
    for w in candidates:
        if w not in seen:
            out.append(w)
            seen.add(w)
            if len(out) >= n:
                break
    return out


# --------------------------------------
# Optional: feature extraction fallback
# --------------------------------------

def _extract_asked_features_from_transcript(transcript_text: str) -> Dict[str, str]:
    """
    Heuristic fallback to derive asked features from transcript, pairing a player
    question like: Do you have symptom "E_53"? with the subsequent agent reply.
    Returns a mapping of evidence_code -> 'yes'/'no'. If unsure, returns {}.
    """
    # This fallback is intentionally conservative; engines should pass the
    # player's internal state instead.
    lines = (transcript_text or "").splitlines()
    out: Dict[str, str] = {}
    for i, line in enumerate(lines):
        m = re.search(r"Do\s+you\s+have\s+symptom\s+\"([^\"]+)\"\?", line, flags=re.IGNORECASE)
        if not m:
            continue
        code = (m.group(1) or "").strip()
        # Find next non-empty line after an "AGENT UTTERANCE:" header
        ans = None
        for j in range(i + 1, min(i + 10, len(lines))):
            if lines[j].strip() == "AGENT UTTERANCE:":
                # Next line is the content
                if j + 1 < len(lines):
                    ans = lines[j + 1].strip().lower()
                    break
        if ans:
            if "yes" in ans:
                out[code] = "yes"
            elif "no" in ans:
                out[code] = "no"
    return out


