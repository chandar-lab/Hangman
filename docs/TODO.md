# TODO v2

A concise plan of next actions to support Objective 1 (PSITs) and the experimental pipeline.

- Implement is_game_over()
  - Phase 1: Judge-based termination using `HangmanJudge` over the current log (fast heuristic) to decide WIN/LOSS/LEAK/INCOHERENCE and stop.
  - Phase 2: (optional) Ground-truth mechanics per game (e.g., Hangman lives/word tracking) for non-LLM termination.

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
  - Provide standardized system prompts for player roles (cooperative/adversarial) per game.

- Housekeeping (optional but recommended)
  - Add config entries for new games/agents; validate on startup.
  - Add small unit tests for provider parsing, tool update logic, and judge JSON parsing.
  - Add notebooks or scripts to aggregate and visualize evaluation metrics.
