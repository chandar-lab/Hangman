## Diagnosis Simulator — Self-Consistency Test (SCT)

This document lists the minimal files and integration points required to reproduce the Hangman SCT pipeline for the Diagnosis Simulator game. The intent is to mirror the Hangman SCT design while allowing game‑specific prompts, a deterministic scripted player, candidate generation, and evaluation.

Assumption: generic filenames without a game qualifier (e.g., `engine_sct.py`) refer to the Hangman version. For Diagnosis, we add diagnosis‑specific counterparts where needed.

### New code to add

- `src/hangman/games/diagnosis_simulator_sct.py`
  - Game wrapper ("game as a log") tailored to Diagnosis SCT.
  - Responsibilities:
    - Load diagnosis‑specific start/turn prompts from `prompts/diagnosis_simulator_sct.py`.
    - Provide `name = "diagnosis_simulator_sct"`.
    - Maintain `interaction_log` with `(utterance, private_state)`.

- `src/hangman/prompts/diagnosis_simulator_sct.py`
  - Prompt set analogous to Hangman SCT prompts:
    - `PLAYER_START_PROMPT`: starts the diagnostic session with explicit rules (no leakage, structured replies, etc.).
    - `PLAYER_QUERY_PROMPT`: deterministic player's probe turn(s) before fork (e.g., ask for symptoms/status checks).
    - `SCT_YES_NO_PROMPT`: yes/no hypothesis question: “Is the diagnosis "{label}"? Answer only: yes or no.”
    - `REVEAL_SECRET_PROMPT`: direct reveal: “Please reveal the diagnosis. Respond with exactly one lowercase term …”.
    - Optional: `CANDIDATE_GENERATION_PROMPT`, `SECRET_EXTRACTION_PROMPT` variants adapted to diagnosis text.

- `src/hangman/players/deterministic_diagnosis_simulator_player.py`
  - Deterministic scripted player for Diagnosis (pre‑fork behavior):
    - Seeded policy to emit a reproducible sequence of patient symptom disclosures/queries.
    - Final turn at `t_fork`: either ask the direct yes/no hypothesis or simply stop before engine forks.
    - Public accessor for used “features” if needed (mirrors `get_used_letters()`; e.g., `get_used_features()`).

- `src/hangman/sct/diagnosis_utils.py`
  - Diagnosis‑specific utilities (parallel to `sct/utils.py` but domain‑aware where needed):
    - `summarize_secret_history(...)`: extract latest diagnosis tag from WM (e.g., `<diagnosis>...</diagnosis>`), keep interface compatible.
    - `format_interaction_log(...)`: re‑use common version or minor tweaks for diagnosis semantics.
    - Deterministic candidate generation for diagnosis labels:
      - Option A (deterministic): filter a curated list (ICD‑like or in‑repo list) consistent with transcript constraints.
      - Option B (LLM): same JSON‑array flow as Hangman; reuse provider interface.
    - `parse_revealed_secret(text)` compatible with diagnosis strings.

- `src/hangman/evaluation/sct_evaluator_diagnosis.py`
  - Evaluator parallel to `evaluation/sct_evaluator.py`, reusing the same output schema:
    - `num_candidates`, `answers_parsed_rate`, `any_yes`, `yes_count`, `first_yes_index`.
    - `wm_secret_summary` (with diagnosis tag), `contains_secret`, `secret_index`, `sct_yes_correct`.
    - `revealed_secret_received_yes`, `ground_truth_secret_received_yes`.
  - If the generic evaluator already suffices (current one is game‑agnostic), you can reuse it and only adjust tag names in `diagnosis_utils`.

- `src/hangman/engine_sct_diagnosis.py`
  - Engine/controller for Diagnosis SCT (mirrors `engine_sct.py`):
    - Wires `DiagnosisSimulatorSCTGame`, `DeterministicDiagnosisSimulatorPlayer`, and diagnosis prompts.
    - Performs the reveal fork using `REVEAL_SECRET_PROMPT`.
    - Builds candidates using diagnosis utils; compose {WM secret, revealed secret, base} with dedup and cap to `n_candidate_secrets`.
    - Executes yes/no branches per candidate with diagnosis `SCT_YES_NO_PROMPT`.
    - Merges `evaluation` (using shared or diagnosis‑specific evaluator) and persists the trial JSON.

- `config/diagnosis_simulator_sct_run.yaml`
  - Run config mirroring Hangman SCT but with:
    - `game: diagnosis_simulator_sct`
    - `agents: [...]`
    - `providers: { main, concurrency/main_pool }`
    - `sct: { t_fork, T_max, random_seed, n_candidate_secrets, stateless_candidates: { method: deterministic|llm, ... } }`

### Existing files to update

- `src/hangman/games/__init__.py`
  - Route `"diagnosis_simulator_sct"` (and short alias) to `DiagnosisSimulatorSCTGame()`.

- `run_experiment_sct.py`
  - Already supports `game` via YAML; ensure it works with `diagnosis_simulator_sct`.
  - Optionally, add examples in the docstring/README for diagnosis runs.

- `src/hangman/sct/utils.py`
  - If you prefer a single shared utils module, ensure tag names and parsers are parameterizable (e.g., pass `tag="diagnosis"`). Otherwise keep `diagnosis_utils.py` separate.

### Optional / nice‑to‑have

- `notebooks/test_sct_diagnosis.ipynb`
  - Quick smoke tests for the reveal flow, candidate generation, and parsing.

- `docs/diagnosis_simulator_sct.md` (this file)
  - Keep as the single source of truth for required components.

### Directory and log expectations

- Results default path: `results/<provider_or_group>/diagnosis_simulator_sct/<agent_name>/...json` or `results/diagnosis_simulator_sct/<agent_name>/...json` depending on your runner settings.
- Trial JSON structure mirrors Hangman SCT with `sct.revealed_secret`, `sct.candidates`, `sct.branches`, and `evaluation` block unchanged.

### Minimal implementation checklist

1) Game wrapper
   - [ ] `src/hangman/games/diagnosis_simulator_sct.py`
   - [ ] `src/hangman/games/__init__.py` route

2) Prompts
   - [ ] `src/hangman/prompts/diagnosis_simulator_sct.py` with START / QUERY / YES_NO / REVEAL / optional candidates & WM prompts

3) Deterministic player
   - [ ] `src/hangman/players/deterministic_diagnosis_simulator_player.py` with seeded pre‑fork behavior and accessor(s)

4) SCT utilities
   - [ ] `src/hangman/sct/diagnosis_utils.py` (or parameterize `sct/utils.py`) for WM parsing, transcript formatting, candidates, reveal parsing

5) Engine
   - [ ] `src/hangman/engine_sct_diagnosis.py` (or generalize `engine_sct.py` to accept a prompt module and player/game factories)

6) Evaluator
   - [ ] `src/hangman/evaluation/sct_evaluator_diagnosis.py` (or reuse generic evaluator if fields align)

7) Config & run
   - [ ] `config/diagnosis_simulator_sct_run.yaml`
   - [ ] Verify `run_experiment_sct.py -r config/diagnosis_simulator_sct_run.yaml` completes

### Notes on generalization (future refactor)

- Make `engine_sct.py` game‑agnostic by injecting a `prompt_module` and `player_factory`/`game_factory` so the same engine supports multiple SCT games.
- In utils, standardize WM tag and candidate‑generation hooks per game (e.g., registry keyed by game name).
- Keep evaluator generic; pass game‑specific parsing via utils.


