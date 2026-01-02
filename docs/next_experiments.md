## Next Experiments Plan

### Scope (within ~$400)
- Complete gpt_oss_120b (aka gpt_oss_big) on all 3 PSITBench games.
- Run qwen_3_235b on all 3 games (family/scale contrast vs existing Qwen runs).
- Optional (budget permitting): one-off Gemini Flash run on Hangman for a premium closed-model reference.

### Estimated per-game cost (one game × 8 baselines × 50 conversations)
- Assumptions:
  - Judge=Kimi K2 at ~$0.006/eval; compute 4 evals/conv (behavioral coherence, behavioral winner, memory intentionality, memory secrecy) ⇒ ~$9.60 per game.
  - Model cost scaled vs `gpt_oss_120b` baseline ($7) using blended input/output token pricing (≈50/50 mix).

- gpt_oss_120b: ≈ $7.00 model + $9.60 judge ≈ $16.60 per game
- qwen_3_235b: factor ≈ 1.391 → ≈ $9.74 model + $9.60 judge ≈ $19.34 per game
- gemini_flash: factor ≈ 6.548 → ≈ $45.83 model + $9.60 judge ≈ $55.43 per game

Budget illustration for full 3-game sweeps:
- gpt_oss_120b (2 remaining games): ≈ 2 × $16.60 = $33.20
- qwen_3_235b (3 games): ≈ 3 × $19.34 = $58.02
- Optional gemini_flash (Hangman only): ≈ $55.43

Totals fit comfortably under $400 with room for retries/ablations.

---

### Steps to set up and run

1) Update provider entries in `config/config.yaml`
- Ensure entries exist with correct pricing/hosts:
  - gpt_oss_120b via OpenRouter (parsing_format: think_tags; tool_parser: openai)
  - qwen_3_235b via OpenRouter (same parsing/prompting settings)
  - Judge provider `kimi_k2_openrouter` already present; verify API key `OPENROUTER_API_KEY` is exported.

2) Create run-configs per game and model
- Copy existing templates (e.g., `config/hangman_run.yaml`) and set:
  - `game`: hangman | twenty_questions | diagnosis_simulator
  - `agents`: 8 baselines (Workflow/ReAct variants with strategies)
  - `providers.main`: model under test (e.g., `gpt_oss_120b_openrouter`, `qwen_3_235b_openrouter`)
  - `evaluator.mode`: both (behavioral+memory) or ["behavioral","rule_based"]
  - `metrics`: per-mode trimming (e.g., `behavioral: ["coherence","secrecy","winner"]`, `memory: ["intentionality","secrecy"]`)
  - `num_trials`: 50, `max_turns`: 25

Suggested files:
- `config/hangman_gptoss120b_run.yaml`
- `config/twenty_questions_gptoss120b_run.yaml`
- `config/diagnosis_simulator_gptoss120b_run.yaml`
- `config/hangman_qwen3_235b_run.yaml`
- `config/twenty_questions_qwen3_235b_run.yaml`
- `config/diagnosis_simulator_qwen3_235b_run.yaml`

3) Create bash scripts in `scripts/` (local or Slurm)
- Pattern (sbatch-ready):
  - activate venv
  - export `OPENROUTER_API_KEY`
  - run: `python run_experiment.py --run-config ./config/<run>.yaml --providers-config ./config/config.yaml`

Suggested scripts:
- `scripts/hangman_run_gptoss120b`
- `scripts/twenty_questions_run_gptoss120b`
- `scripts/diagnosis_run_gptoss120b`
- `scripts/hangman_run_qwen3_235b`
- `scripts/twenty_questions_run_qwen3_235b`
- `scripts/diagnosis_run_qwen3_235b`

4) Sanity checks before launch
- `python -m hangman.providers.llmprovider` quick call via `load_llm_provider` for each provider
- Dry-run one trial per run-config (num_trials=1) to validate logging and evaluation blocks

5) Launch
- Local: run scripts directly or via `nohup`.
- Slurm: `sbatch scripts/<script_name>`

6) Tracking and resume
- Results under `results/<game>/<agent>/...`
- The runner skips completed trials and cleans incomplete logs; re-run is safe.

---

### Notes
- If judge eval set changes (e.g., add/remove a metric), update the `metrics` dict per mode to avoid unnecessary judge calls and control cost.
- For concurrency on OpenRouter, consider staggering to avoid rate limits; log failures and auto-retry if needed.

