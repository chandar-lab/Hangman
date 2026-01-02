from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import BaseMessage, HumanMessage

from hangman.providers.llmprovider import LLMProvider
from hangman.prompts.hangman_sct import CANDIDATE_GENERATION_PROMPT
from wordfreq import top_n_list


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
    - multi_tag_in_state: True if any single snapshot included multiple <secret> tags
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
    - Ask LLM to extract a JSON array of lowercase secret words in order of first appearance.
    - Use the returned list to construct a synthetic sequence of private-state tags for deterministic summarization.
    - Then call the deterministic summarizer over that synthetic list to compute the same metadata shape.
    If LLM is unavailable or parsing fails, fall back to deterministic summarization over original states.
    """
    from hangman.prompts.hangman_sct import SECRET_EXTRACTION_PROMPT

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

        # Expect a JSON array of strings
        secrets_list: List[str] = []
        arr = json.loads(text)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    w = item.strip().lower()
                    if w and re.fullmatch(r"[a-z]+", w):
                        secrets_list.append(w)

        # Rebuild a synthetic sequence with one secret per turn (in order of first appearance)
        # For simplicity, assign each discovered secret to the turn index of its first appearance order.
        synthetic_states: List[str] = []
        for s in secrets_list:
            synthetic_states.append(f"<{tag}>{s}</{tag}>")

        print("-- LLM secret extraction succeeded --\nLLM raw response:", text, "\n--")
        return summarize_secret_history(synthetic_states, tag=tag)
    except Exception:
        print("-- LLM secret extraction failed --\nLLM raw response:", text, "\n--")
        return summarize_secret_history(private_states, tag=tag)


# --------------------------------------
# Transcript formatting for LLM prompts
# --------------------------------------

def format_interaction_log(log: Sequence[Sequence[Any]]) -> str:
    """
    Best-effort formatting of the interaction log into a readable, turn-by-turn string.
    Each item in log is [utterance, private_state] or similar.
    Private states are intentionally excluded from this formatted transcript
    for candidate generation.
    """
    parts: List[str] = []
    for i, pair in enumerate(log):
        turn_no = i + 1
        utter = pair[0] if isinstance(pair, (list, tuple)) and len(pair) > 0 else ""
        # private state intentionally ignored
        actor = "Player" if (i % 2 == 0) else "Agent"
        parts.append(f"--- TURN {turn_no} ({actor}) ---")
        if actor == "Player":
            parts.append("PLAYER UTTERANCE:")
        else:
            parts.append("AGENT UTTERANCE:")
        parts.append(str(utter))
        # do not include private memory in candidate-generation transcript
        parts.append("")
    return "\n".join(parts)


# --------------------------------------
# Pattern inference (deterministic only)
# --------------------------------------

_PATTERN_RE = re.compile(r"(?:^|[^a-zA-Z])([a-zA-Z_][a-zA-Z_\s]{2,}[a-zA-Z_])(?:[^a-zA-Z]|$)")


def infer_pattern_from_text(text: str) -> Optional[str]:
    """
    Extract the last hangman-like pattern from text.
    Priority 1: lines starting with "Pattern:" or "Current pattern:" (case-insensitive), capturing to end-of-line.
    Priority 2: fallback heuristic over any underscore/letter spans.
    Returns the raw matched pattern (trimmed), or None if not found.
    """
    if not text:
        return None

    # Prefer explicit pattern lines, e.g. "Pattern: _ _ a _ _" or "Current pattern: `_ _ _ g _ _`"
    explicit: List[str] = []
    for m in re.finditer(r"(?im)^(?:\s*)(?:current\s+pattern|pattern)\s*:\s*([`_a-zA-Z\s]+)$", text):
        frag = (m.group(1) or "").strip()
        if "_" in frag:
            # strip backticks if present
            frag = frag.replace("`", "").strip()
            explicit.append(frag)
    if explicit:
        return explicit[-1]

    # Pass 2: scan lines that look like a bare pattern (only letters, underscores, spaces)
    bare_candidates: List[str] = []
    for line in (text.splitlines() or []):
        raw = (line or "").strip()
        if not raw:
            continue
        # strip surrounding backticks
        if raw.startswith("`") and raw.endswith("`") and len(raw) >= 2:
            raw = raw[1:-1].strip()
        # must contain at least one underscore
        if "_" not in raw:
            continue
        # accept only letters/underscores/spaces
        if re.fullmatch(r"[a-zA-Z_\s]+", raw):
            bare_candidates.append(raw)
    if bare_candidates:
        return bare_candidates[-1]

    # Fallback heuristic: capture spans comprised of underscores/letters/spaces
    candidates: List[str] = []
    for m in _PATTERN_RE.finditer(text):
        frag = (m.group(1) or "").strip()
        if "_" in frag:
            candidates.append(frag)
    return candidates[-1] if candidates else None


def _normalize_pattern(pattern: str) -> str:
    # Keep only lowercase letters and underscores; remove spaces
    p = re.sub(r"\s+", "", pattern or "").lower()
    return re.sub(r"[^a-z_]", "", p)


def _load_dictionary(dict_path: str) -> List[str]:
    words: List[str] = []
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w and re.fullmatch(r"[a-z]+", w):
                    words.append(w)
    except Exception:
        return []
    return words


def parse_revealed_secret(text: str) -> Optional[str]:
    """
    Parse a direct reveal answer. Return a single lowercase [a-z]+ token if present, else None.
    Strips common quotes/backticks and ignores punctuation and extra words.
    """
    if not text:
        return None
    t = (text or "").strip().lower()
    # Strip surrounding quotes/backticks
    if (t.startswith("`") and t.endswith("`")) or (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    # First clean token
    m = re.search(r"\b([a-z]+)\b", t)
    if not m:
        return None
    return m.group(1)


def _extract_guessed_letters_from_transcript(text: str) -> Set[str]:
    """
    Parse guessed letters from transcript.

    Supports:
    - Player prompts of the form: "Is there a X in the secret word?"
    - Agent summaries like: "Guessed letters: r, a, l" (with optional backticks)
    """
    guesses: Set[str] = set()
    # Player guess lines (current prompt style)
    for m in re.finditer(r"Is\s+there\s+a[n]?\s+\"?([a-zA-Z])\"?\s+in\s+the\s+secret\s+word\?", text, flags=re.IGNORECASE):
        guesses.add(m.group(1).lower())
    # Agent summary lines listing guessed letters
    for m in re.finditer(r"Guessed\s+letters:\s*([`a-zA-Z,\s]+)", text, flags=re.IGNORECASE):
        chunk = m.group(1)
        for ch in re.findall(r"[a-zA-Z]", chunk):
            guesses.add(ch.lower())
    return guesses


def _word_matches_constraints(word: str, pattern_norm: str, absent_letters: Set[str]) -> bool:
    if len(word) != len(pattern_norm):
        return False
    for i, ch in enumerate(pattern_norm):
        if ch == "_":
            if word[i] in absent_letters:
                return False
            continue
        if word[i] != ch:
            return False
    # Word must not include any absent letters at all
    if any((a in word) for a in absent_letters):
        return False
    return True


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
    used_letters: Optional[Set[str]] = None,
) -> List[str]:
    """
    Estimate up to n candidate secret words consistent with the transcript.
    - method == 'deterministic':
        * If dictionary_path provided, filter it by inferred pattern and guessed-letter constraints.
        * If no dictionary, return [].
    - method == 'llm':
        * Use CANDIDATE_GENERATION_PROMPT and the provided provider; parse JSON array and return up to n words.
    """
    if n <= 0:
        return []

    method_key = (method or "").strip().lower()
    if method_key == "deterministic":
        # If a dictionary_path is provided, preserve file-based behavior; otherwise use wordfreq.
        if dictionary_path:
            return _estimate_candidates_deterministic_file(
                transcript_text,
                n=n,
                dictionary_path=dictionary_path,
                used_letters=used_letters,
            )
        return _estimate_candidates_deterministic_wordfreq(
            transcript_text,
            n=n,
            used_letters=used_letters,
            max_vocab=None,
        )
    if method_key == "llm":
        return _estimate_candidates_llm(transcript_text, n=n, llm_provider=llm_provider, llm_max_n=llm_max_n)
    return []


def _estimate_candidates_deterministic_file(
    transcript_text: str,
    *,
    n: int,
    dictionary_path: Optional[str],
    used_letters: Optional[Set[str]],
) -> List[str]:
    if not dictionary_path or not os.path.exists(dictionary_path):
        return []
    pattern_raw = infer_pattern_from_text(transcript_text)
    if not pattern_raw:
        return []
    pattern = _normalize_pattern(pattern_raw)
    words = _load_dictionary(dictionary_path)
    if not words:
        return []
    present_letters: Set[str] = set(ch for ch in pattern if ch != "_")
    guessed_all = set(used_letters or []) or _extract_guessed_letters_from_transcript(transcript_text)
    absent_letters = guessed_all.difference(present_letters)

    out: List[str] = []
    seen: Set[str] = set()
    for w in words:
        if w in seen:
            continue
        if _word_matches_constraints(w, pattern, absent_letters):
            out.append(w)
            seen.add(w)
            if len(out) >= n:
                break
    return out


def _estimate_candidates_deterministic_wordfreq(
    transcript_text: str,
    *,
    n: int,
    used_letters: Optional[Set[str]],
    max_vocab: Optional[int],
) -> List[str]:
    pattern_raw = infer_pattern_from_text(transcript_text)
    if not pattern_raw:
        return []
    pattern = _normalize_pattern(pattern_raw)
    present_letters: Set[str] = set(ch for ch in pattern if ch != "_")
    guessed_all = set(used_letters or []) or _extract_guessed_letters_from_transcript(transcript_text)
    absent_letters = guessed_all.difference(present_letters)

    limit = max_vocab or 300000
    vocab = top_n_list("en", limit)

    out: List[str] = []
    seen: Set[str] = set()
    L = len(pattern)
    for w in vocab:
        w = (w or "").strip().lower()
        if not w or not re.fullmatch(r"[a-z]+", w):
            continue
        if len(w) != L:
            continue
        if _word_matches_constraints(w, pattern, absent_letters):
            if w not in seen:
                out.append(w)
                seen.add(w)
                if len(out) >= n:
                    break
    return out


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

    # Expect a JSON array of strings
    candidates: List[str] = []
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    w = item.strip().lower()
                    if w and re.fullmatch(r"[a-z]+", w):
                        candidates.append(w)
    except Exception:
        # Best-effort: try to locate the first JSON array in text
        print("-- LLM candidate generation failed --\nLLM raw response:", text, "\n--")
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            try:
                arr = json.loads(text[s : e + 1])
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, str):
                            w = item.strip().lower()
                            if w and re.fullmatch(r"[a-z]+", w):
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


