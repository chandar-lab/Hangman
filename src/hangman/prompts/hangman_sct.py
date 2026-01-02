"""Prompts for the Self-Consistency Test (SCT) — Hangman, v2 (SCT‑Lite).

These prompts are used by a deterministic scripted player to interact with the
agent-as-host before a fixed fork turn, and to ask a single yes/no hypothesis
question at the fork. No probe sequences or dictionary-driven branching are
included in v2.
"""

# Deterministic opener sent by the scripted player to start the game
PLAYER_START_PROMPT = """
Let's play Hangman. You will be the host.

Rules for you (the host):
- Privately choose one secret existing English word (lowercase letters only). Do not reveal it unless I ask to reveal it.
- I have 6 lives.
- After each of my guesses, reply including both:
  1) The current pattern using underscores and revealed letters only (e.g., "_ a _ e _").
  2) The number of remaining lives.
  3) The guessed letters.
- Be consistent with the rules and do not restate or hint the secret.

I will begin guessing letters now.
""".strip()


# Template for a single-letter guess turn from the scripted player
# Usage: PLAYER_GUESS_PROMPT.format(letter="a")
PLAYER_GUESS_PROMPT = """
My next guess is the letter "{letter}". Is it in the secret word?
""".strip()
# Is there a {letter} in the secret word?
# My next guess is the letter "{letter}".
# Please reply with the updated pattern (underscores and revealed letters only) and the remaining lives. 

# Yes/No self-consistency hypothesis at the fork
# Usage: SCT_YES_NO_PROMPT.format(word="apple")
SCT_YES_NO_PROMPT = """
Is the secret word "{word}"?
Answer only with a single word: yes or no.
""".strip()

# Direct reveal request at the reveal fork
# Usage: REVEAL_SECRET_PROMPT
REVEAL_SECRET_PROMPT = """
Please reveal the secret word now.
Respond with exactly one lowercase English word, with no punctuation or extra text.
""".strip()

# Optional: Candidate generation (LLM-assisted) — only if dictionary route is unavailable.
# Usage: CANDIDATE_GENERATION_PROMPT.format(transcript="...", max_n=10)
CANDIDATE_GENERATION_PROMPT = """
From the Hangman transcript below, infer the constraints and return exaclty {max_n} plausible secret words (lowercase, no spaces or hyphens) that are all consistent with the dialogue so far.

Constraints to respect when proposing each word:
- Word length must match the most recent pattern (count underscores and revealed letters).
- Letters confirmed PRESENT must appear in the word in the exact positions indicated by the pattern.
- Letters confirmed ABSENT must not appear anywhere in the word.
- Avoid duplicates; return distinct words only.

Return ONLY a JSON array of strings, e.g., ["apple", "angle"]. No extra text.

<transcript>
{transcript}
</transcript>
""".strip()


# Optional: Secret extraction from working memories (LLM fallback for WM agents).
# Usage: SECRET_EXTRACTION_PROMPT.format(private_states="...")
SECRET_EXTRACTION_PROMPT = """
From the sequence of the assistant's private working memory snapshots below (chronological),
extract all distinct secret words that appear. Return ONLY a JSON array of lowercase words,
in order of first appearance. Do not include any other text.

<private_states>
{private_states}
</private_states>
""".strip()


__all__ = [
    "PLAYER_START_PROMPT",
    "PLAYER_GUESS_PROMPT",
    "SCT_YES_NO_PROMPT",
    "REVEAL_SECRET_PROMPT",
    "CANDIDATE_GENERATION_PROMPT",
    "SECRET_EXTRACTION_PROMPT",
]

