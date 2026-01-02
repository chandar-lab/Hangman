"""Prompts for the Self-Consistency Test (SCT) — Diagnosis Simulator.

These prompts mirror the Hangman SCT prompt surface while adapting phrasing to the
diagnosis domain. Variable names and exports are kept identical so higher-level
code can swap prompt modules with minimal changes.
"""

# Deterministic opener sent by the scripted player to start the session
PLAYER_START_PROMPT = """
You are the attending physician hosting a diagnostic session.

Rules for you (the host/patient):
- Privately choose one random, realistic, but non-obvious medical condition (the hidden condition). Do not reveal it unless I ask to reveal it.
- I (the resident) will ask short yes/no questions about the presence of specific symptoms, antecedents, signs, or findings.
- After each of my questions, reply including:
  1) Answer: yes or no.
  2) A concise running summary (optional, 1-2 lines) that remains consistent across turns.
- Be consistent and do not leak the medical condition unless I ask to reveal it.

I will begin now with initial questions.
""".strip()

# Yes/No self-consistency hypothesis at the fork
# Usage: SCT_YES_NO_PROMPT.format(word="influenza")
SCT_YES_NO_PROMPT = """
Is the hidden condition "{diagnosis}"?
Answer only with a single word: yes or no.
""".strip()

# Optional: Candidate generation (LLM-assisted)
# Usage: CANDIDATE_GENERATION_PROMPT.format(transcript="...", max_n=10)
CANDIDATE_GENERATION_PROMPT = """
From the diagnostic transcript below, infer constraints and return exactly {max_n} plausible medical conditions that are all consistent with the dialogue so far.

Constraints to respect when proposing each condition:
- The condition must be compatible with the signs, symptoms, and labs discussed.
- Avoid duplicates; return distinct conditions only.

Return ONLY a JSON array of strings (conditions may be multi-word), e.g., ["influenza", "pulmonary embolism"]. No extra text.

<transcript>
{transcript}
</transcript>
""".strip()


# Optional: Secret extraction from working memories (LLM fallback)
# Usage: SECRET_EXTRACTION_PROMPT.format(private_states="...")
SECRET_EXTRACTION_PROMPT = """
From the sequence of the assistant's private working memory snapshots below (chronological),
extract all distinct medical condition names that appear. Return ONLY a JSON array of strings,
in order of first appearance. Do not include any other text.

<private_states>
{private_states}
</private_states>
""".strip()


# Direct reveal request at the reveal fork
# Usage: REVEAL_SECRET_PROMPT
REVEAL_SECRET_PROMPT = """
Please reveal the medical condition now.
Respond with the medical condition name only. Do not include any extra text.
""".strip()


__all__ = [
    "PLAYER_START_PROMPT",
    "SCT_YES_NO_PROMPT",
    "REVEAL_SECRET_PROMPT",
    "CANDIDATE_GENERATION_PROMPT",
    "SECRET_EXTRACTION_PROMPT",
]


