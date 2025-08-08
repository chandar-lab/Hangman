# TODO v2

A concise plan of next actions to support Objective 1 (PSITs) and the experimental pipeline.

- Add new PSIT games (src/hangman/games/ + prompts/ + minimal players/)
  - 20 Questions
    - Agent privately selects a common object/person/concept.
    - Player asks yes/no questions; agent answers consistently from private state.
    - Add prompts, game wrapper, and cooperative player role.
  - Zendo (Text Version)
    - Agent privately defines a simple rule (e.g., "word has >2 vowels").
    - Player proposes examples; agent replies Yes/No based on hidden rule.
    - Add prompts, game wrapper, and player role description.
  - Medical Diagnosis Simulator
    - Agent privately selects a condition; acts as patient.
    - Player (doctor) asks questions; agent answers consistently with condition.
    - Add prompts, game wrapper, and doctor player role.

- Implement is_game_over()
  - Phase 1: Judge-based termination using `HangmanJudge` over the current log (fast heuristic) to decide WIN/LOSS/LEAK/INCOHERENCE and stop.
  - Phase 2: (optional) Ground-truth mechanics per game (e.g., Hangman lives/word tracking) for non-LLM termination.

- Implement run_experiment.py sweep from config.yaml
  - Parse agents, games, players, models, seeds, and trials.
  - Run cross-product sweeps; support parallelism (e.g., process/thread pool with rate limiting).
  - Save logs under results/{game}/{agent}/timestamp.json.
  - After each trial, run `HangmanJudge` (or game-specific judge) and store evaluations alongside logs.
  - Produce a summary CSV/JSON of metrics per setting.

- Refactor agents
  - Add vanilla stateless LLMs
    - Vanilla (no CoT), CoT, ToT (tree-style), ReAct (no private state), PublicMemory (memory appended to chat only).
  - Simplify naming
    - Use concise names and consistent suffixes (e.g., ReAct, ReaDisPat, ReaDisOve, ReaDisUpd, ReaKee).
  - Add more complex baselines
    - Reflexion-style self-reflection (public trace), Cognitive Tools (tool-augmented planning without private state), StateAct (explicit private-state tool usage baseline).
  - Ensure every agent implements: `invoke()`, `get_private_state()`, `reset()`; align on `ModelOutput` and thread configs.

- Refactor prompts
  - Centralize agent prompts; document how/when to read/write private memory.
  - Add clear instructions for secrecy vs. public utterances per game.
  - Provide standardized system prompts for player roles (cooperative/adversarial) per game.

- Housekeeping (optional but recommended)
  - Add config entries for new games/agents; validate on startup.
  - Add small unit tests for provider parsing, tool update logic, and judge JSON parsing.
  - Add notebooks or scripts to aggregate and visualize evaluation metrics.
