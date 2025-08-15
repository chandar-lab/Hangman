# TODO v2

A concise plan of next actions to support Objective 1 (PSITs) and the experimental pipeline.

- Implement is_game_over()
  - Phase 1: Judge-based termination using `HangmanJudge` over the current log (fast heuristic) to decide WIN/LOSS/LEAK/INCOHERENCE and stop.
  - Phase 2: (optional) Ground-truth mechanics per game (e.g., Hangman lives/word tracking) for non-LLM termination.

-- Refactor Agents
  - Remove ProvatCoTAgent mentions
  - Add new patch and ovewrite tools support to ReAct
  - Double check what is the patch algorithm used by copilot or Gemini CLI
  - Implement 

- Refactor prompts
  - Centralize agent prompts; document how/when to read/write private memory, what is working memory, how to interact with it.
  - Provide standardized system prompts for player roles (cooperative/adversarial) per game.

- Housekeeping (optional but recommended)
  - Add config entries for new games/agents; validate on startup.
  - Add small unit tests for provider parsing, tool update logic, and judge JSON parsing.
  - Add notebooks or scripts to aggregate and visualize evaluation metrics.
