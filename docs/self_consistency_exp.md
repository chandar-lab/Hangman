# Self-Consistency Test (SCT) — Hangman

Agents
	•	Vanilla LLM (POCA)
	•	Private CoT (hidden scratchpad)
	•	ReAct + memory tools
	•	Workflow + memory tools

Player
	•	Scripted letter-guessing policy
	•	Maintains candidate set C_t from dictionary \mathcal{D}
	•	Guesses: greedy info-gain (pre-fork), fixed probe sequence P (post-fork)

Fork
	•	At first turn t_{\text{fork}} where 6 \le |C_t| \le 20
	•	Discard episode if no such turn or if |C_t|=1

Candidate Secrets
	•	Sample N=\min(10,|C|) from C_{t_{\text{fork}}}
	•	WM agents: force-include committed secret \(s^\*\) (from <secret> tag)
	•	Vanilla: no private state → just N samples

Post-Fork Branches

For each s \in S_N:
	1.	Copy transcript up to t_{\text{fork}}
	2.	Run probe sequence P (length L=6)
	3.	Ask for reveal
	4.	Accept s if (a) probe turns match Hangman rules for s, and (b) final reveal = s

Result: set of accepted secrets W \subseteq S_N

Metrics
	•	SCT-Accuracy (WM only): \(1[W=\{s^\*\}]\)
	•	SCT-Uniqueness (all): 1[|W|=1]
	•	Multiplicity: |W|
	•	False Acceptance Rate: fraction of non-true secrets accepted
	•	Desync rate (WM): \(s^\* \notin C_{t_{\text{fork}}}\)

Parsing
	•	Regex for _ a _ / “positions 2,4” / “yes/no”
	•	Fallback: LLM parser → {guess, outcome, positions}
	•	If unparsable → mark inconsistent

Expected Outcomes
	•	WM agents: high SCT-Accuracy / Uniqueness (~1), low FAR
	•	Vanilla: low Uniqueness, high Multiplicity/FAR

# TODOs for Self-Consistency Test (SCT) — Hangman (Players & Fork Logic)

0) Global Defaults
	•	Dictionary \mathcal{D}: lowercase, ASCII, no spaces/hyphens.
	•	Determinism: Player is fully deterministic (fixed policy, fixed tie-breaks).
	•	Agent creativity: Set temperature = 0.6–0.8 for agents only (Maybe even 0.3 and indicating to be creative?).
	•	Turn caps: T_{\max}=8; fork when 6 \le |C_t| \le 20, or for at 4 \le t_{fork} \le 8

1) Player: Start Prompt & Policy

1.1 PLAYER_START_PROMPT (deterministic opener), something like

```
You are the assistant in a Hangman game. Please play strictly by these rules:

1) Privately choose ONE secret English word from your own imagination. Do not reveal it unless explicitly asked to reveal the final answer at the end.
2) I have 6 lives.
3) When I guess a letter, reply in free text but include:
   - Pattern with underscores and revealed letters, e.g., "_ a _ _ e"
   - Remaining lives
4) Be concise and consistent. Do not restate the secret.

I will begin guessing letters now.
```

	•	Design choice (subtle): We do not constrain format rigidly (to keep realism), but we require explicit pattern + hit/miss + positions to enable parsing.

1.2 Player letter policy (deterministic)
	•	Maintain C_t and guessed set G_t.
	•	Guess selection: info-gain greedy over \mathcal{D} restricted to C_{t-1}. Tie-break by fixed alphabet order.
	•	Parsing rule: From each agent reply, extract:
	•	pattern (e.g., "_ a _ _ e")
	•	hit/miss, and positions (1-based)
	•	Ambiguity handling: If unparsable → treat as misspecified turn and discard episode (keep rate ≤5% via prompt polish).

Subtle choice: Keep player strictly deterministic; all randomness comes from agent (so differences across runs are due to secrets, not player variance).

2) Fork Logic t_{\text{fork}}
	•	After each parsed turn, update C_t.
	•	Fork at earliest t with 6 \le |C_t| \le 20. If never met by T_{\max}, discard episode, or
	•	fix a t_{\text{fork}}

Subtle choice: This range balances identifiability pressure vs under-constraint, making branch contradictions more likely for no-memory agents.

3) Candidate Set C_{t_{\text{fork}}}

3.1 Deterministic construction (preferred)
	•	Recompute C_{t_{\text{fork}}} from transcript using strict Hangman rules over \mathcal{D}.
	•	Sampling for SCT: N=\min(10, |C|). If |C|>10, sample uniformly with fixed seed; else use all.

Subtle choice: Avoid LLM for candidate generation here to keep the oracle deterministic and reproducible.

3.2 (Optional) LLM-assisted variant
	•	If dictionary is too large or patterns noisy, ask a small LLM:
“Given this transcript, list up to 10 plausible English words consistent with it, lowercase only.”
	•	Then filter the LLM list through deterministic checker against \mathcal{D}.
Use only if deterministic route fails to provide enough candidates.

4) Secret Extraction from Working Memory (WM agents)

4.1 Deterministic regex (primary)
	•	Memory schema includes:

<working_memory>
  ...
  <secret>YOURWORD</secret>
  ...
</working_memory>

	•	Rule: Take the last <secret> value at or before t_{\text{fork}}.
	•	Regex: <secret>\s*([a-zA-Z\- ]+)\s*</secret>
	•	Normalize: lowercase, strip spaces/hyphens, ASCII fold.
	•	Validity: must be in \mathcal{D}.

4.2 LLM-fallback (only if regex fails)
	•	Prompt: “Extract the secret word from the working memory. Return ONLY the word in lowercase, no punctuation.”
	•	Validate against \mathcal{D}.

Subtle choice: Use deterministic regex first to avoid evaluation bias; LLM only as rescue.

5) Self-Consistency Check (Post-Fork)

5.1 Branch construction
	•	Create N branches, one per s \in S_N\subseteq C_{t_{\text{fork}}}.
	•	WM agents: If \(s^\\) (extracted) ∈ C_{t_{\text{fork}}}, force-include it in S_N (replace randomly with fixed seed if needed). If s^\\notin C → desync (record, score 0 for SCT).
	•	Context:
	•	Prefix: identical public transcript up to t_{\text{fork}}.
	•	WM agents: attach same private memory snapshot up to t_{\text{fork}}.
	•	Vanilla: no private state added.

5.2 Yes/No probe (parsable)

For each candidate s:

We are continuing the SAME game from the previous messages.
Question: Is the secret word exactly “[s]”? 
Only answer with a single token: "yes" or "no".

	•	Parse rule: case-insensitive exact yes/no; otherwise mark invalid → no.

Subtle choice: We avoid revealing the whole candidate list; each branch sees only one hypothesis to reduce priming/telemetry effects.

5.3 (Optional) Pattern probe (to stabilize parsing)

Before the yes/no:

Please restate the current pattern with underscores and letters only (e.g., "_ a _ e _").
Do NOT reveal the secret.

	•	Used to sanity-check state drift post-fork; optional to keep minimalism.

6) Metrics (episode-level)
	•	SCT-Accuracy (WM only): 1[ exactly one “yes” AND it is s* ]
	•	SCT-Uniqueness (all): 1[ exactly one “yes” ]
	•	Multiplicity: number of “yes”
	•	Desync rate (WM): 1[ s* ∉ C_{t_fork} ]
	•	Unparsable rate: fraction of yes/no replies not exactly yes/no

Subtle choice: Using yes/no makes evaluation trivial, avoids fragile full-simulation parsing at post-fork.

7) Parsing & Robustness

7.1 Pre-fork parser (for C_t)
	•	Regex patterns for _ a _ _, “positions 2, 4”, “present/absent”.
	•	If fail, tiny LLM parser → {guess, outcome, positions}.
	•	Validate on 100 labeled turns; require ≥95% F1.

7.2 Guardrails
	•	If agent refuses meta-questions, send one retry reminding rules.
	•	If still non-compliant → mark branch invalid (counts as “no”).

Subtle choice: Keep retries minimal to avoid training the agent toward the target answer format.

8) Randomness & Reproducibility
	•	Fix seeds for: candidate sampling (if |C|>10), agent prompts (where supported), and episode selection.
	•	Log: full transcripts, WM snapshots, C_t per turn, and branch decisions.

9) Risks & Mitigations (subtle points)
	•	Agent hashes/memory leaks: Do not show candidate lists in public; keep private memory invisible to public transcript.
	•	Vanilla “mental secret”: If Vanilla consistently returns one “yes” across branches, that is evidence of implicit private state; report it—this strengthens the paper’s thesis.
	•	Temperature too high: If unparsable rate >5%, lower agent temp by 0.1 and/or restate pattern requirement in start prompt.

8) Prompts to be defined
    1.	PLAYER_START_PROMPT – Player asks the agent to start Hangman with rules.
    2.	PLAYER_GUESS_PROMPT – Player asks agent about a specific guessed letter.
    3.	CANDIDATE_GENERATION_PROMPT (optional, if using LLM to propose candidate words).
    4.	SECRET_EXTRACTION_PROMPT (LLM fallback, if regex fails to extract <secret>).
    5.	SELF_CONSISTENCY_CHECK_PROMPT – “Is the secret word [WORD]? Only answer yes/no.”
    6.	PARSER_FALLBACK_PROMPT – Convert free-form agent response into {guess, outcome, positions}.
    7.	RETRY_PROMPT – Nudge agent to comply if answer is non-parsable or non-yes/no.