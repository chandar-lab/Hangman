# AI Agent with Working Memory & Reasoning Distillation
Conversational agents with private working memory and a reasoning distillation loop, evaluated on simple games (default: Hangman). Backed by vLLM via an OpenAI-compatible API.

![Alt text](assets/Main_Figure_Hangman.png "Optional Title")

## What this repo does
- Builds and evaluates conversational agents with private working memory that persists across turns.
- Supports two agent paradigms: WorkflowAgent (two-LLM responder → updater) and ReActMemAgent (single-LLM ReAct with tools), plus CoT/stateless variants.
- Updates memory using three strategies via structured tools: overwrite; patch_and_replace; append_and_delete.
- Runs single games or batch experiments (Hangman, Twenty Questions, Zendo, Diagnosis Simulator), logs full interactions, and evaluates with an LLM Judge (behavioral and memory views, plus winner).
- Plugs into multiple model backends through `LLMProvider`: `vllm_native`, `openrouter_sdk`, and OpenAI-compatible servers.

### Supported agents
- `WorkflowAgent` (two-LLM, two-stage: responder → updater)
- `ReActMemAgent` (single-LLM ReAct with memory-edit tools)
- `PublicCoTAgent` (publishes chain-of-thought)
- `PrivateCoTAgent` (stores chain-of-thought privately)
- `ReActAgent` (stateless ReAct)
- `VanillaLLMAgent` (plain chat)

### Memory update strategies (tools)
All memoryful agents can use one of three strategies for updating private `working_memory`:
- overwrite: `overwrite_memory`
- patch_and_replace: `patch_memory` and/or `replace_in_memory`
- append_and_delete: `append_in_memory` and/or `delete_from_memory`

## How it works 
1) Providers
    - `config/config.yaml` declares LLM endpoints, model names, parsing format (think tags vs direct), and generation params.
    - Backends supported by `LLMProvider`:
        - `vllm_native`: custom FastAPI server (`src/hangman/providers/vllm_http_server.py`) with two-pass generation and tool-call support.
        - `openrouter_sdk`: OpenRouter via the OpenAI SDK; optional reasoning text is wrapped into `<think>` for uniform parsing.
        - OpenAI-compatible HTTP clients (default fallback) for other compatible servers.
    - `LLMProvider` parses optional `<think>…</think>` text (when `parsing_format: think_tags`) into `thinking` vs public `response`.

2) Agents (ReAct vs Workflow)
    - WorkflowAgent (two-stage):
        - Responder LLM produces the public reply (optionally with `<think>` reasoning).
        - Updater LLM returns STRICT-JSON tool calls which are executed to update private `working_memory`.
        - Persists conversation and memory via LangGraph checkpoints.
    - ReActMemAgent (single-LLM ReAct with tools):
        - Model may emit tool calls inline; the agent executes memory tools sequentially and persists updated `working_memory`.
        - History is pruned per turn to drop within-turn tool chatter while keeping the final AI message.
    - Other variants: `PublicCoTAgent` (thinking appended publicly), `PrivateCoTAgent` (thinking stored privately), `ReActAgent` (stateless), `VanillaLLMAgent`.

3) Player
    - `LLMPlayer` role-plays using a system prompt and a role-reversed copy of the public conversation.

4) Game engine
    - `GameLoopController` alternates turns (Player ↔ Agent), appends a per-turn record: (utterance, agent_private_state), and live-writes a JSON.
    - End-of-run, it calls `LLMJudge` to score metrics and merges results into the same JSON.

5) Judge
    - `LLMJudge` selects prompts via `evaluation/prompt_registry.py` for “memory” vs “behavioral” modes and extracts a JSON with scores for metrics (intentionality, secrecy, mechanism, coherence).

## Architecture (modules)
- providers: `LLMProvider`, `load_llm_provider` (native vLLM / OpenRouter / OpenAI-compatible)
- agents: `BaseAgent` + variants (`WorkflowAgent`, `ReActMemAgent`, `PublicCoTAgent`, `PrivateCoTAgent`, `ReActAgent`, `VanillaLLMAgent`)
- players: `LLMPlayer` (role-player)
- games: “Game as a log” (`BaseGame`, `HangmanGame` for prompts)
- engine: `GameLoopController` (turn loop, JSON logs, judge)
- evaluation: `LLMJudge` + prompt registry
- prompts: game- and agent-specific prompt templates

## Data flow (inputs → engine → outputs)
- Inputs: `games_run.yaml` (game, agents, trials, max_turns, eval modes/metrics, provider names) + `config.yaml` (provider specs).
- Runtime: Engine passes LangChain messages between Player (as user) and Agent (as assistant). Agent maintains private `working_memory` via structured memory tools (overwrite, patch/replace, append/delete).
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
python src/hangman/agents/workflow_agent.py
```
or
```bash
python src/hangman/agents/reactmem_agent.py
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
python run_experiment.py --run-config ./config/games_run.yaml --providers-config ./config/config.yaml
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
  - provider names that must exist in `config/config.yaml` (e.g., `qwen3_14b_vllm_hermes`)
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