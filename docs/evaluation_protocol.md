# Evaluation Protocol

This document summarizes how evaluation is performed for all games in this repository, how it integrates with the engine and experiment runner, and what metrics are produced.

## Overview
- The engine (`src/hangman/engine.py`) logs every turn of a trial into a JSON file under `results/...` as the game runs.
- At the end of a trial, the engine calls the evaluator (`HybridEvaluator`) with the interaction log to compute metrics.
- The experiment runner (`run_experiment.py`) orchestrates multiple trials, agents, and concurrency; it writes one JSON per trial and resumes incomplete runs.
- A maintenance tool (`re_evaluate.py`) can re-run evaluation over existing logs that are missing/errored.

## Evaluator
`HybridEvaluator` (in `src/hangman/evaluation/hybrid_evaluator.py`) supports three families of evaluation:
- Behavioral (LLM-judged, using only public utterances)
- Memory (LLM-judged, using public utterances + agent private state)
- Rule-based (deterministic checks over the interaction log)

Invocation:
- Inputs: `trial_data = {"interaction_log": List[List[Any]]}` where each turn is `[utterance, private_state]`.
- Mode: one of `behavioral`, `memory`, `rule_based`, `both` (behavioral+memory), or `all` (behavioral+memory+rule_based).
- Metrics subset: optional list, otherwise defaults are used per mode.

Prompts:
- Prompts are resolved per game and mode via `prompt_registry.get_prompts(game, mode)`, which loads from `src/hangman/prompts/*`.
- A generic, game-agnostic “winner” prompt is injected for behavioral judging.

## Metrics
The following metrics may be produced. Availability depends on mode and game prompt support.

- Intentionality (memory-only):
  - LLM-judged: probes early turns for intent to establish a hidden secret in private state.
  - Rule-based: detects the first agent turn where `<secret>...</secret>` appears in private state.

- Secrecy (behavioral + memory):
  - LLM-judged: assesses whether the agent keeps the secret from public outputs.
  - Rule-based: flags a leak if the agent publicly reveals the normalized secret before the player ever mentions it; also reports fuzzy similarity.

- Mechanism (memory-only):
  - LLM-judged: whether the agent employs a coherent mechanism/strategy consistent with the game’s rules and its private memory.

- Conversational Coherence (behavioral + memory):
  - LLM-judged: checks for turn-by-turn coherence and adherence to the game.

- Winner (behavioral-only):
  - LLM-judged: decides whether `Player`, `Agent`, or `Unknown` won the game, with reasoning and confidence.

Notes:
- Behavioral mode uses only public utterances.
- Memory mode includes the agent’s private state in prompts (when available).
- Rule-based metrics are deterministic and do not call an LLM.

## JSON Output Structure
Each trial JSON contains:
- `metadata`: game, agent/player classes, provider configs, `max_turns`, timestamps
- `interaction_log`: the full turn-by-turn list of `[utterance, private_state]`
- `evaluation`:
  - `mode`: the requested mode spec
  - `results`: a dict keyed by mode (`behavioral`, `memory`, `rule_based`) containing metric results. The behavioral section includes a `winner` entry when computed.

## Control Flow
1. Game run (`GameLoopController.run`):
   - Alternates player/agent turns until `max_turns` or game-over (game-over heuristic may be disabled).
   - Writes a live JSON log after every turn.
2. Final evaluation:
   - Builds `trial_data` from the `interaction_log` and calls `HybridEvaluator.evaluate_trial(...)` with `mode` and optional `metrics`.
   - Merges the evaluator’s output into the same trial JSON under `evaluation`.
3. Experiment orchestration (`run_experiment.py`):
   - Loads YAML run config (agents, trials, providers, evaluator mode, metrics, first mover).
   - Supports parallel trials via provider pools and a process pool.
   - Cleans up partial logs and resumes incomplete runs.
4. Re-evaluation (`re_evaluate.py`):
   - Scans a configured results glob.
   - Re-runs evaluation when missing or errored and overwrites the `evaluation` field.
   - Supports optional concurrency and judge provider rotation.

## Practical Tips
- Ensure judge provider availability (API keys, base URLs, ports) to avoid connection errors.
- In multi-GPU local runs, align provider ports (e.g., 8001–8004) with the server launcher.
- Use `mode: all` for comprehensive outputs; use subsets to save cost/time.
