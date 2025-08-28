"""Registry for judge prompts by (game, mode), supporting memory vs behavioral variants.

Rules per metric:
- intentionality: memory-only (omit in behavioral mode)
- mechanism: memory-only (omit in behavioral mode)
- secrecy: has MEMORY and BEHAVIORAL variants
- coherence: has MEMORY and BEHAVIORAL variants

Fallbacks: If variant constants are not present in a module, we fall back to the
legacy single-variant constant (e.g., SECRECY_JUDGE_PROMPT) for both modes.
"""
from importlib import import_module
from typing import Dict


def _load_game_module(game_key: str):
    mapping = {
        "hangman": "hangman.prompts.hangman",
        "20q": "hangman.prompts.twenty_questions",
        "20-questions": "hangman.prompts.twenty_questions",
        "twenty_questions": "hangman.prompts.twenty_questions",
        "twenty-questions": "hangman.prompts.twenty_questions",
        "20_questions": "hangman.prompts.twenty_questions",
        "zendo": "hangman.prompts.zendo",
        "diagnosis": "hangman.prompts.diagnosis_simulator",
        "diagnosis_simulator": "hangman.prompts.diagnosis_simulator",
        "medical_diagnosis": "hangman.prompts.diagnosis_simulator",
    }
    key = game_key.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported game for prompts: {game_key}")
    return import_module(mapping[key])


def _resolve_prompts_for_mode(mod, mode: str, game_name: str) -> Dict[str, str]:
    fmt = getattr(mod, "FORMAT_INSTRUCTIONS")

    # Memory-only metrics
    intentionality_mem = getattr(mod, "INTENTIONALITY_JUDGE_PROMPT", None)
    mechanism_mem = getattr(mod, "MECHANISM_JUDGE_PROMPT", None)

    # Dual-variant metrics with safe fallbacks
    secrecy_mem = getattr(
        mod, "SECRECY_JUDGE_PROMPT_MEMORY", getattr(mod, "SECRECY_JUDGE_PROMPT", None)
    )
    secrecy_beh = getattr(
        mod, "SECRECY_JUDGE_PROMPT_BEHAVIORAL", getattr(mod, "SECRECY_JUDGE_PROMPT", None)
    )
    coherence_mem = getattr(
        mod,
        "COHERENCE_JUDGE_PROMPT_MEMORY",
        getattr(mod, "COHERENCE_JUDGE_PROMPT", None),
    )
    coherence_beh = getattr(
        mod,
        "COHERENCE_JUDGE_PROMPT_BEHAVIORAL",
        getattr(mod, "COHERENCE_JUDGE_PROMPT", None),
    )

    metrics: Dict[str, str] = {}
    if mode == "memory":
        if intentionality_mem:
            metrics["intentionality"] = intentionality_mem
        if secrecy_mem:
            metrics["secrecy"] = secrecy_mem
        if mechanism_mem:
            metrics["mechanism"] = mechanism_mem
        if coherence_mem:
            metrics["coherence"] = coherence_mem
    else:  # behavioral
        if secrecy_beh:
            metrics["secrecy"] = secrecy_beh
        if coherence_beh:
            metrics["coherence"] = coherence_beh

    # Augment behavioral prompts with the game-agnostic winner prompt
    winner_fmt = None
    if mode == "behavioral":
        try:
            winner_mod = import_module("hangman.prompts.winner")
            winner_prompt = getattr(winner_mod, "WINNER_JUDGE_PROMPT", None)
            winner_fmt = getattr(winner_mod, "WINNER_FORMAT_INSTRUCTIONS", None)
            if winner_prompt:
                # Inject the game name without touching other placeholders
                metrics["winner"] = winner_prompt.replace("{game_name}", game_name)
        except Exception:
            # If winner prompt is not available, silently skip to preserve compatibility
            pass

    # Note: Return dedicated winner format instructions separately to avoid schema conflicts
    bundle = {"metrics": metrics, "format_instructions": fmt}
    if winner_fmt is not None:
        bundle["winner_format_instructions"] = winner_fmt
    return bundle


def get_prompts(game: str, mode: str) -> Dict[str, str]:
    mod = _load_game_module(game)
    mode_key = mode.strip().lower()
    if mode_key not in {"memory", "behavioral"}:
        raise ValueError("mode must be 'memory' or 'behavioral'")
    return _resolve_prompts_for_mode(mod, mode_key, game)
