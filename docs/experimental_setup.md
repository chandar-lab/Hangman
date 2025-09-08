# Experimental Setup for PSITs Paper

This document outlines the experimental design for evaluating Private Space Interactive Tasks (PSITs) using text-based private memory agents. The setup tests claims that LLMs and existing agentic frameworks struggle with PSITs, and our proposed memory update strategies (overwrite, patch_and_replace, append_and_delete) across ReActMemAgent and WorkflowAgent improve performance.

## Overview
- **Tasks (Games)**: Hangman, Twenty Questions, Diagnosis Simulator (3 core; optionally Zendo as 4th for diversity, but consider removing to reduce compute).
- **Models**: Qwen3 14B and GPT-OSS 20B for main experiments; GPT-OSS 120B for model scale ablation.
- **Agents**: WorkflowAgent and ReActMemAgent (with 3 memory strategies each); Baselines: VanillaLLMAgent, PrivateCoTAgent, ReActAgent (no PublicCoT).
- **Scale**: 500-1000 trials per condition (agent x strategy x model x game) for mains; smaller (100-200) for ablations.
- **Runs**: Batch via run_experiment.py; use DeepSpeed for efficiency. Estimate costs with notebooks/estimate_cost.ipynb.

## Hyperparameters
- **Games**: 4
- **Matches per game**: 50
- **Turns per Dialogue**: Variable per game (e.g., max 20 for Hangman, 30 for Twenty Questions, 40 for Diagnosis Simulator to allow completion).
- **Max Tokens per Generation**: 512 
- **Reasoning Tokens (Qwen3 14B)**: 256 
- **Temperature**: 0.3 (default for deterministic agent responses; adjustable, e.g., 0.7 for more creative judge evaluations).
- **Experiment Size Calculation**: 2 LLMs * 3 games * 8 agent configs * 50 matches * (avg. 20 turns + 7 evals) ≈ 84k LLM calls (reduced by rule-based evals for some metrics).

## Evaluation Protocol
Hybrid: Rule-based for objective metrics; LLMJudge for subjective. No full ground-truth mechanics (skip is_game_over; terminate at max_turns=20).

### Rule-Based Metrics
- **Secrecy**: Scan public utterances for secret leaks (exact/fuzzy match via regex/Levenshtein >90%). Score 1-5: 5=no leak, 3=partial hint, 1=full leak. Force normalized formatting in agents (e.g., &lt;secret&gt;word&lt;/secret&gt;).
- **Intentionality**: Check if secret appears in private memory by turn 1-2. Score 1-5 based on timing/clarity.
- **Memory Growth**: Track avg. length of working_memory over turns (tokens or chars) per strategy.
- **Additional Metrics**:
  - Leakage Rate: % of trials with any detected leak.
  - Win Rate: Approximate via rule-based winner detection (e.g., regex for "you win" in logs).
  - Update Efficiency: Avg. number of tool calls per turn; edit distance between memory versions.
  - Turn Efficiency: Avg. turns to approximate win (capped at max_turns).

### LLM-Based Metrics
- Coherence, Mechanism: Use existing LLMJudge (1-5 scores with reasoning/confidence) powered by a state-of-the-art model like Kimi K2, Gemini 2.5, or GPT-5.
- Hybrid Aggregation: Average rule-based and LLM scores; report correlations.

### Robustness
- Noise Tests: Perturb prompts/logs.

## Main Experiments
- **Setup**: Same LLM for agent/player to minimize variables.
- **Conditions**: 2 frameworks x 3 strategies x 2 models x 3-4 games.
- **Metrics**: Report means/std devs, p-values (t-tests).
- **Plots**: Win rate vs. strategy; leakage over turns; memory growth curves.

## Ablations
- **Memory Strategy**: Isolate overwrite vs. patch_and_replace vs. append_and_delete on secrecy/memory growth.
- **Framework**: ReActMem vs. Workflow on efficiency/leakage.
- **Model Scale**: Smaller runs with GPT-OSS 120B vs. 20B; plot scaling laws (e.g., win rate improves with size).
- **Same-LLM Bias**: Smaller mixed-LLM experiments (e.g., Qwen agent + GPT player) to check if results change (e.g., due to word preferences in Hangman). Compare to main same-LLM runs.
- **Prompt Sensitivity**: Vary prompts and measure variance.

## Baselines Justification
Baselines cover naive (VanillaLLMAgent), CoT (PrivateCoTAgent), and stateless ReAct. Literature review found no direct SOTA for secretive agents; explain in paper (no need for untested external baselines).

## Reproducibility and Compute
- Seeds for all randomness.
- Full configs in appendices.
- Compute: Limit to feasible runs; prioritize Hangman for depth.

This setup ensures rigor for top-tier submission.
