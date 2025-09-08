### FORALI: How to run the Twenty Questions multi‑GPU experiments (Qwen3)

This guide shows you how to clone the repo, set up the environment, and submit the multi‑GPU SLURM job that runs the Twenty Questions experiments using Qwen3‑14B with one conversation per GPU.

### Prerequisites
- SLURM cluster node with 4× A100 80GB GPUs (or adjust GPU_LIST accordingly)
- Linux with Bash, curl
- Python 3.10+ (recommended)
- Outbound internet for model downloads (Hugging Face) and for the judge provider (OpenRouter)
- An OpenRouter API key exported as `OPENROUTER_API_KEY` (used by the judge LLM `kimi_k2_openrouter`)

### 1) Clone the repository
```bash
git clone https://github.com/Dundalia/Hangman.git
cd hangman
```

### 2) Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
# Install the project (uses pyproject.toml)
pip install -e .
```

If you prefer Poetry (optional):
```bash
pip install poetry
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

### 3) Export required environment variables
- OpenRouter (judge provider):
```bash
export OPENROUTER_API_KEY="<your_openrouter_api_key>"
```
- Optional: choose a local cache for Hugging Face models (to speed up downloads):
```bash
export HF_HOME="$HOME/scratch"
```

### 4) Verify configuration
- Providers are declared in `config/config.yaml`. For Qwen3 multi‑GPU we defined four providers bound to localhost ports 8001–8004:
  - `qwen3_14b_vllm_hermes_gpu0` → `http://localhost:8001`
  - `qwen3_14b_vllm_hermes_gpu1` → `http://localhost:8002`
  - `qwen3_14b_vllm_hermes_gpu2` → `http://localhost:8003`
  - `qwen3_14b_vllm_hermes_gpu3` → `http://localhost:8004`

- The run config for Twenty Questions with Qwen3 pools is: `config/twenty_questions_qwen3_run.yaml`.
  - Results are written under `results/qwen_3/twenty_questions/`.
  - Judge LLM is `kimi_k2_openrouter` (requires `OPENROUTER_API_KEY`).
  - `num_trials: 50`, `max_turns: 30`, and eight agents are configured.

### 5) Submit the multi‑GPU job
Use the provided SLURM script which launches one vLLM server per GPU, waits for health on ports 8001–8004, and then runs the experiment in parallel.

```bash
sbatch scripts/twenty_questions_run_multigpu
```

Optional environment overrides (passed through to the launcher):
```bash
# Default model and settings; change as needed
sbatch scripts/twenty_questions_run_multigpu
```

### 6) Monitor the job
```bash
squeue -u $USER
```

You should see lines indicating:
- Multi‑GPU servers started and health checks passed
- Parallel execution enabled with 4 workers
- Results being written under `results/qwen_3/twenty_questions/<agent_name>/...json`

### 7) Resumability and partial runs
- If a job is preempted or times out, partial JSON logs that lack evaluation will be auto‑deleted on the next run, and only unfinished trials will be re‑run.
- Simply re‑submit the same SLURM script; completed trials are detected by the presence of `evaluation.results` in their JSONs.

### 8) Where results live
- Twenty Questions (Qwen3): `results/qwen_3/twenty_questions/<agent_name>/...json`

Each JSON contains:
- `metadata` (providers used, timestamps)
- `interaction_log` (turn‑by‑turn transcript)
- `evaluation` with `behavioral` and `memory` (LLM judge) and optional `rule_based` metrics

### Quickstart (copy‑paste)
```bash
git clone https://github.com/<your-org-or-user>/hangman.git && cd hangman
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel && pip install -e .
export OPENROUTER_API_KEY="<your_openrouter_api_key>"
export HF_HOME="$HOME/scratch"
sbatch scripts/twenty_questions_run_multigpu
```


