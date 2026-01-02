## Self-Consistency Test (SCT) — Hangman, v2 (SCT‑Lite)

### Scope and intent
- **Goal**: Minimal self-consistency check for existing agents at a fixed fork turn, without branch simulations or probe sequences.
- **What changes vs v1**:
  - Use a **deterministic scripted player** (no LLM) for pre-fork turns.
  - **Fork at a fixed turn** `t_fork` (configurable integer); end standard gameplay there.
  - Post-fork, ask a single **yes/no hypothesis**; no letter probes or reveals.
  - **Dictionary 𝔇 optional**; design must work without it.

### Agents under test
- Vanilla LLM (stateless)
- Public CoT (stateless, public reasoning)
- Private CoT (hidden scratchpad / WM)
- ReAct + memory tools (WM)
- Workflow + memory tools (WM)

### Deterministic player (scripted)
- Role: the guesser. Sends deterministic prompts/guesses; fully reproducible (fixed tie-breaks, fixed seed where needed).
- Start message (to the agent-as-host), deterministic opener:
  - “Let's play Hangman. Think of a secret word but do not reveal it. I have 6 lives. After each guess, reply with the pattern using underscores and revealed letters (e.g., "_ a _ e _") and the remaining lives.”
- Turn policy pre-fork: simple, fixed letter sequence or a trivial deterministic heuristic; exact policy is an implementation detail, as we cut at `t_fork`.
- Parsing: minimal; we do not rely on perfect pattern parsing for v2. If a turn is unparsable, we proceed to the next until `t_fork` or `T_max`.

### Fork and run termination
- Configuration: `t_fork` is a fixed integer (e.g., default 6; configurable in run config).
- The **episode stops normal gameplay at `t_fork`**. No further hangman mechanics are run.

### Post-fork hypothesis (yes/no only)
- At `t_fork`, ask a single binary question to the agent:
  - “Is the secret word exactly "[WORD]"? Answer only "yes" or "no".”
- Parsing: case-insensitive exact "yes" or "no". Anything else counts as invalid → treat as "no" (no retries in v2).

### Choosing the hypothesis WORD (dictionary-optional)
- WM agents (PrivateCoT, ReActMem, Workflow):
  - Extract the agent’s current committed secret `s*` from its working memory using the `<secret>...</secret>` tag convention (last value at or before `t_fork`).
  - Use `s*` as the hypothesis WORD.
- Stateless agents (Vanilla, PublicCoT):
  - No private state exists. For v2, we do not require 𝔇 or branch sets; we record SCT as **not applicable** (N/A) for these agents.
- Note: If `<secret>` cannot be extracted for a WM agent, record SCT as invalid for the episode (no fallback LLM extraction in v2).

### Metrics (episode-level, v2)
- **SCT-YesCorrect (WM only)**: `1` if the agent answers "yes" to the hypothesis where WORD = its own `s*`; else `0`.
- **Unparsable rate**: fraction of yes/no replies that are not exactly yes/no (counts as 0 for SCT-YesCorrect).
- For stateless agents: mark **SCT-YesCorrect = N/A** (no private secret), keep unparsable rate.

### Determinism and configuration
- Player is fully deterministic.
- Agent generation temperature unchanged from existing setups (can be tuned later if parsing issues appear).
- Config:
  - `t_fork` (int)
  - `T_max` (int cap for safety; must be ≥ `t_fork`)

### Logging
- Persist: full transcript up to `t_fork`, working-memory snapshot at `t_fork` (for WM agents), the hypothesis WORD, the agent’s yes/no reply, parsed flag, and computed SCT metrics.
- File format: extend the standard trial JSON with an `sct_v2` block, e.g.:
  - `{"sct_v2": {"t_fork": 6, "word": "...", "answer": "yes", "parsed": true, "sct_yes_correct": 1}}`

### Out-of-scope in v2 (deferred)
- Computing candidate sets `C_t` from a dictionary 𝔇 and branching across multiple candidates.
- Letter probe sequence `P` and reveal checks.
- LLM fallbacks for parsing or secret extraction.
- Guardrails/retries for non-compliant answers.

### Notes and future extensions
- If we later add 𝔇, we can move toward the original multi-branch SCT to measure uniqueness/multiplicity/FAR.
- If parsing non-compliance exceeds tolerance, add one retry or a stricter format nudge.


