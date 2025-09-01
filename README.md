# AI Agent with Working Memory & Reasoning Distillation
Conversational agents with private working memory and a reasoning distillation loop, evaluated on simple games (default: Hangman). Backed by vLLM via an OpenAI-compatible API.

![Alt text](assets/Main_Figure_Hangman.png "Optional Title")

## What this repo does
- Benchmarks agents that keep a private state (“working memory”) and refine it each turn via think → distill → patch.
- Runs single games and batch experiments, logs full interactions (public utterances + private memory), and scores with an LLM Judge.

## How it works (flow in one page)
1) Providers
    - `config.yaml` declares LLM endpoints (local vLLM or OpenRouter), model names, parsing format (think tags vs direct), and gen params.
    - `LLMProvider` wraps an OpenAI-compatible client and parses optional <think>…</think> text.

2) Agent (example: ReaDisPatActAgent)
    - LangGraph workflow: generate_response → generate_diff → apply_diff.
    - Response uses `working_memory` as context; distillation LLM emits a text patch; diff-match-patch updates `working_memory`.
    - A persistent “main_thread” stores messages + memory across turns.

3) Player
    - `LLMPlayer` role-plays using a system prompt and a role-reversed copy of the public conversation.

4) Game engine
    - `GameLoopController` alternates turns (Player ↔ Agent), appends a per-turn record: (utterance, agent_private_state), and live-writes a JSON.
    - End-of-run, it calls `LLMJudge` to score metrics and merges results into the same JSON.

5) Judge
    - `LLMJudge` selects prompts via `evaluation/prompt_registry.py` for “memory” vs “behavioral” modes and extracts a JSON with scores for metrics (intentionality, secrecy, mechanism, coherence).

## Architecture (modules)
- providers: `LLMProvider`, `load_llm_provider` (OpenAI-compatible client + think-tag parsing)
- agents: `BaseAgent` + variants (notably `ReaDisPatActAgent` with LangGraph)
- players: `LLMPlayer` (role-player)
- games: “Game as a log” (`BaseGame`, `HangmanGame` for prompts)
- engine: `GameLoopController` (turn loop, JSON logs, judge)
- evaluation: `LLMJudge` + prompt registry
- prompts: game- and agent-specific prompt templates

## Data flow (inputs → engine → outputs)
- Inputs: `games_run.yaml` (game, agents, trials, max_turns, eval modes/metrics, provider names) + `config.yaml` (provider specs).
- Runtime: Engine passes LangChain messages between Player (as user) and Agent (as assistant). Agent maintains private `working_memory` via diff/patch.
- Outputs: One JSON per trial under `results/<game>/<agent>/…` with metadata, `interaction_log` (list of [utterance, private_state]), and `evaluation`.

---

## Setup 
Prereqs: Python 3.11+ and Poetry installed.

Clone and enter the repo:
```Bash
git clone https://github.com/chandar-lab/Hangman.git
cd hangman
```

Install deps (Poetry creates .venv and installs from pyproject.toml):
```Bash
poetry install
```

Activate the virtualenv:
``` Bash
source ./.venv/bin/activate
```

Add packages (optional):
```Bash
poetry add [package_name]
```

Serve the local model with the native vLLM server (Terminal 1):
```bash
export HF_HOME=~/scratch

# Recommended: native two-pass server with reasoning-trace token control
./run_qwen_vllm_native_server.sh

# Env overrides (optional):
# MODEL=Qwen/Qwen3-14B PORT=8001 ./run_qwen_vllm_native_server.sh
```

This launches `src/hangman/providers/vllm_http_server.py`, a lightweight FastAPI server that implements a two-pass generate flow (thinking → answer) and allows controlling the number of tokens allocated to the reasoning trace (`max_thinking_tokens`) separately from the final response (`max_response_tokens`).

Legacy (optional): you can still use the OpenAI-compatible vLLM server if needed:
```bash
./run_qwen_openai_server.sh
```

Notes on API keys for local vLLM:
- If you run a local server at http://localhost (or 127.0.0.1), this project will allow an empty API key even when an `api_key_env` is configured in `config.yaml`.
- You can set `VLLM_API_KEY` if your local server enforces authentication, but it’s not required for typical localhost setups.
- For remote endpoints (e.g., OpenRouter), a valid API key is still required and the app will fail fast if it’s missing.

## Run
Quick chat loops (Terminal 2, venv activated):
```Bash
python src/hangman/agents/readispatactagent.py
```
or
```bash
python src/hangman/agents/react.py
```

Agent vs Player (engine-driven game loop):
```Bash
python src/hangman/engine.py
```

Suggested first message:
```
Let's play Hangman! You be the host. Think of a secret word, but don't tell me what it is. I'll try to guess it, one letter at a time. Just show me the blank spaces for the word to start.
```

Batch experiments:
- Edit `games_run.yaml` (agents, trials, max_turns, providers, eval modes)
- Then run:
```bash
python run_experiment.py
```

### Running big experiments on Slurm (sbatch)
Under `scripts/`, there are Slurm job scripts to launch large experiments on a cluster while automatically bringing up the local vLLM native server and tearing it down afterward:

- `scripts/hangman_run`
- `scripts/ds_run` (Diagnosis Simulator)
- `scripts/tq_run` (Twenty Questions)
- `scripts/zendo_run`

How they work:
- Each script:
  - activates the repo virtualenv
  - starts `./run_qwen_vllm_native_server.sh` in the background
  - waits for `http://localhost:8001/health` to become ready
  - runs the batch driver with a specific run config (e.g., `python run_experiment.py --run-config ./config/hangman_run.yaml`)
  - cleans up the background server on exit

Relation to config files:
- The `--run-config` passed by each script points to a YAML in `config/` that defines:
  - the game (`hangman`, `diagnosis_simulator`, `twenty_questions`, `zendo`)
  - which agents to run and how many trials
  - provider names that must exist in `config/config.yaml` (e.g., `qwen3_14b_local_vllm_native`)
- Provider connection details and reasoning-token controls are taken from `config/config.yaml` and the native server (`vllm_http_server.py`).

Usage:
```bash
sbatch scripts/hangman_run
```

## Results
- JSON logs written to `results/<game>/<agent>/…` with:
  - `metadata`: game, agent, provider configs, timestamps
  - `interaction_log`: `[utterance, private_state]` per turn
  - `evaluation`: LLMJudge scores per metric/mode