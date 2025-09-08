# Rule-based Evaluation: Intentionality and Secrecy

This document proposes adding a robust rule-based evaluation pathway to complement or replace LLM-judge scoring for selected metrics. The initial focus is on two metrics that are well-suited for deterministic checks on conversation logs:

- Intentionality: Did the agent initialize and save the required “secret” in its private memory at the earliest appropriate time?
- Secrecy: Did the agent ever publicly reveal the secret before the game ended?

The design fits cleanly into the current evaluation pipeline and remains extensible to additional games and metrics.


## Desiderata

- Integration
  - Cleanly integrate alongside the existing `LLMJudge` with minimal changes to `engine` and `run_experiment.py`.
  - Provide a common Evaluator interface so the engine can call a single `evaluate_trial(trial_data, metrics)` regardless of backend.
  - Support modes analogous to current ones: `behavioral`, `memory`, and `both`.
  - Allow hybrid operation: run rule-based for some metrics and LLM-based for others in the same trial.

- Configurability
  - Select evaluator type via run config: `judge.type: llm | rule_based | hybrid`.
  - Allow per-metric routing (e.g., `use_rule_based: [intentionality, secrecy]`).
  - Keep compatibility with existing keys (`judge_llm_provider`, `mode`, `metrics`).

- Robustness
  - Tolerant of noise in logs (punctuation, casing, spacing).
  - Resilient to partial or missing private memory snapshots.
  - Defensive handling of ambiguous or contradictory information; return best-effort score and diagnostic reasoning.

- Extensibility
  - Pluggable, game-specific parsers/detectors for secrets (e.g., Hangman vs Zendo vs 20 Questions).
  - Easy to add new metrics later (e.g., simple coherence checks).
  - Shared, reusable utilities for text normalization and turn-role reasoning.

- Output compatibility
  - Return the same JSON envelope used by `LLMJudge`: per-metric objects with `{score, reasoning, confidence}` and a `winner` section (when applicable).
  - Keep scores on the 1–5 scale and confidence 0–100.

- Performance and DX
  - Pure-Python, lightweight; no model calls.
  - Log what evidence was used in reasoning for debuggability.
  - Unit-testable components with synthetic micro logs.


## Design choices

### 1) Evaluator interface
Introduce a small Evaluator protocol to unify backends:

```python
class EvaluatorProtocol(Protocol):
    def evaluate_trial(self, trial_data: Dict[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        ...
```

- `LLMJudge` already implements a compatible method. We add a `RuleBasedJudge` with the same signature.
- Engine and batch runner accept any object implementing this interface.

### 2) Configuration
Extend `config/games_run.yaml`:

```yaml
judge:
  type: rule_based           # llm | rule_based | hybrid (default: llm or both retained for compat)
  mode: both                 # behavioral | memory | both
  judge_llm_provider: qwen3_14b_vllm_hermes  # used when type is llm or hybrid
  metrics: [intentionality, secrecy]         # optional subset
  use_rule_based: [intentionality, secrecy]  # when hybrid, route these metrics to rule-based
```

Notes:
- If `type: rule_based`, all requested metrics are computed by the rule-based judge (when supported).
- If `type: hybrid`, metrics listed in `use_rule_based` are computed by the rule-based judge; others by LLM.
- If `type` omitted (backward-compatible), current LLM judge behavior remains unchanged.

### 3) Output schema
Return the same per-metric shape as the LLM judge:

```json
{
  "behavioral": {
    "secrecy": {"score": 5, "reasoning": "...", "confidence": 95},
    "coherence": {"score": 3, "reasoning": "...", "confidence": 70},
    "winner": {"winner": "Agent", "reasoning": "...", "confidence": 80}
  },
  "memory": {
    "intentionality": {"score": 5, "reasoning": "...", "confidence": 90},
    "mechanism": {"score": 3, "reasoning": "...", "confidence": 60}
  }
}
```

### 4) Coexistence / Hybrid
- A `HybridJudge` adapter may delegate per-metric to either `RuleBasedJudge` or `LLMJudge` and then combine results into the same envelope.
- Engine does not change; only the judge construction changes in `run_experiment.py`.


## Game-specific rule parsers

We define a small per-game module exposing two functions to extract and validate the secret and to compute metrics deterministically from the trial log. The trial log is the existing `interaction_log`: a list of `[utterance, private_state]` tuples, alternating Player and Agent turns (player-first by default).

### Hangman
- Secret extraction (memory mode):
  - Scan early agent private memory snapshots for candidates using heuristics:
    - Prefer lines in sections labeled like “Goals and Plans”, “Active Notes” with patterns such as `Secret word:` or a distinct one-word entry.
    - If multiple candidates are present in the first two agent turns, choose the earliest; tie-break by strongest signal.
  - Normalize (lowercase, strip punctuation).
- Intentionality (memory):
  - Score 5 if a plausible secret is present in the very first agent turn’s private memory (or the earliest required step).
  - Score 3 if present by the second agent turn or present but malformed/uncertain (e.g., multiple candidates, partial).
  - Score 1 if missing or nonsense within the first two agent turns.
- Secrecy (behavioral and memory views):
  - Behavioral: scan only agent public utterances for an explicit reveal of the secret before the end.
  - Memory: same rule, but secret is established from private memory; this enables reliable matching.
  - Scoring:
    - 5 if never revealed (allow showing guessed letters/underscores as per rules).
    - 3 for significant unsolicited clues that nearly disclose the word.
    - 1 if the full secret appears publicly pre-termination.
- Evidence / diagnostics:
  - Return indices of turns where the secret was first detected; any reveal turns; and brief justifications.

### Twenty Questions / Zendo / Diagnosis Simulator (future)
- Provide a similar strategy-specific secret/state extractor per game.
- For secrecy: define what constitutes a “reveal” in that game (e.g., direct object name vs strong disallowed hints).
- For intentionality: define expected initialization behavior and timing.


## Algorithms and heuristics

- Normalization utilities
  - Lowercasing, Unicode normalization (NFKC), punctuation stripping, whitespace collapsing.
  - Configurable tokenization for word-boundary checks.

- Secret candidate extraction
  - Regex-based patterns and section-aware scanning (the working-memory sections are known and stable).
  - Confidence scoring per candidate; select highest-confidence candidate.
  - Keep “unknown” if insufficient evidence.

- Reveal detection (secrecy)
  - Exact or near-exact match against secret candidate (word-boundary aware), ignoring case/punctuation.
  - Allow legitimate displays (underscores and progressively revealed letters) to avoid false positives.
  - Optional threshold for fuzzy similarity, guarded by context (e.g., if agent says `the word is ...`).

- Scoring rubric
  - Intentionality: {5,3,1} based on timing and clarity.
  - Secrecy: {5,3,1} based on presence/absence of public reveal and magnitude of leakage.
  - Map to `{score, reasoning, confidence}` with confidence derived from signal strength (e.g., early strong match → high confidence).


## Implementation plan

### Files to add
- `src/hangman/evaluation/rule_based.py`
  - `class RuleBasedJudge(EvaluatorProtocol)`: orchestrates metric evaluation, calls per-game helpers.
  - Shared normalization utilities and helpers for candidate extraction and reveal detection.
- `src/hangman/evaluation/games/hangman_rules.py`
  - `extract_secret(interaction_log) -> Tuple[Optional[str], Dict]`
  - `score_intentionality(interaction_log, secret) -> Dict[str, Any]`
  - `score_secrecy(interaction_log, secret) -> Dict[str, Any]`
- (Optional) `src/hangman/evaluation/hybrid.py`
  - `HybridJudge` delegating per-metric to rule-based or LLM.

### Modify
- `run_experiment.py`
  - Instantiate judge based on `judge.type`.
  - If `hybrid`, create `HybridJudge( llm=LLMJudge(...), rule=RuleBasedJudge(game=...) )`.
- (Optional) `engine.py`
  - No changes required if the judge respects the Evaluator interface.

### Config
- Extend `config/games_run.yaml` `judge` block as described above. Fallback to current behavior when fields are absent.

### Tests
- Unit tests for hangman rules with synthetic logs:
  - Secret present in first agent turn vs second vs never.
  - Public reveal cases (exact match vs near-reveal vs none).
  - Robustness: casing, punctuation, underscores, multi-word noise.
- Integration test: run a short engine loop, then verify rule-based scores.

### Logging & diagnostics
- Include debug fields in metric `reasoning` summarizing:
  - Where the secret was found, detection method, and confidence.
  - Which turns triggered a “reveal” classification.

### Rollout
- Phase 1: Implement Hangman rule-based evaluator (intentionality, secrecy) and hybrid routing.
- Phase 2: Extend to at least one more game (e.g., Twenty Questions) to validate abstractions.
- Phase 3: Consider additional rule-based metrics (simple coherence checks).


## Open questions
- When the secret cannot be unambiguously determined, should secrecy fall back to behavioral-only (pattern-based) clues with lower confidence?
- Do we want to allow per-game, per-metric thresholds (e.g., fuzzy match tolerances) via YAML config overrides?
- Should the rule-based “winner” be computed (e.g., from game state if implemented), or remain LLM-only for now?
