# TODO v2

A concise plan of next actions to support Objective 1 (PSITs) and the experimental pipeline.

- Implement is_game_over()
  - Phase 1: Judge-based termination using `HangmanJudge` over the current log (fast heuristic) to decide WIN/LOSS/LEAK/INCOHERENCE and stop.
  - Phase 2: (optional) Ground-truth mechanics per game (e.g., Hangman lives/word tracking) for non-LLM termination.

- Remove Zendo

- Add Rule Based Eval
  - Force agent to format in a normalized way the secret 

## New Things to Implement
- Implement rule-based secrecy metric: Scan public utterances for leaks using regex and Levenshtein distance.
- Implement rule-based intentionality metric: Check early private memory for secret presence.
- Add memory growth tracking: Measure avg. working_memory length over turns.
- Additional metrics: Leakage rate (% trials with leaks), win rate approximation (regex/human), update efficiency (tool calls/edit distance), turn efficiency (avg. turns).
- Force normalized secret formatting in agent prompts/tools (e.g., <secret> ... </secret>).
- Mixed-LLM ablation: Smaller experiments with different LLMs for agent/player to check bias (e.g., Qwen agent + GPT player).
- Model scale ablation: Runs with GPT-OSS 120B vs. 20B.
- Hybrid evaluation: Combine rule-based and LLMJudge with aggregation logic.
- Statistical reporting: Add means/std devs, p-values in results.ipynb.
- Noise robustness tests: Perturb prompts/logs.

## Open Points: Design Choices Yet to Finalize
- Include Zendo as 4th game or stick to 3 (Hangman, Twenty Questions, Diagnosis Simulator)?
- Exact thresholds for secrecy (e.g., Levenshtein >90% for 'leak')?
- Scope of human evaluation (e.g., 10% vs. 20% of trials; which metrics)?
- Prompt variations for sensitivity ablation (e.g., how many variants, what changes)?
- Win rate approximation method (regex reliability vs. more human checks)?
- Compute trade-offs: Trial counts per condition (500 vs. 1000) given costs. 