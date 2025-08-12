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

Serve the local model with vLLM (Terminal 1):
```bash
export HF_HOME=~/scratch

python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen3-14B \
     --trust-remote-code \
     --port 8000 \
     --dtype bfloat16 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes
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

## Results
- JSON logs written to `results/<game>/<agent>/…` with:
  - `metadata`: game, agent, provider configs, timestamps
  - `interaction_log`: `[utterance, private_state]` per turn
  - `evaluation`: LLMJudge scores per metric/mode