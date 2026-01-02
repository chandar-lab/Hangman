## Self-Consistency Test (SCT) — Hangman

### Overview
- Goal: implement a self-consistency test for Hangman episodes that forks at a fixed turn and queries the agent’s internal commitment (for memoryful agents) or interrogates a set of plausible hypotheses (for stateless agents).
- Non-goals (for now): multi-branch simulations, probe sequences, or dictionary-required logic. We will allow a dictionary if present, but the design works without it.


### Milestone 0 — Grounding and scope
- Desiderata
  - Support both memoryful agents (Workflow, ReActMem, PrivateCoT) and stateless agents (Vanilla, PublicCoT).
  - Deterministic, scripted player for pre-fork turns; reproducible via a random seed and a seeded letter policy. The policy may later prefer vowels/frequent letters.
  - At `t_fork`, stop normal gameplay. For memoryful agents, extract a secret from private state and build a candidate set of size `n_candidate_secrets` that includes the secret plus additional estimated candidates; ask yes/no per candidate. For stateless agents, estimate a candidate set `C_t` of size `n_candidate_secrets` and ask yes/no per candidate.
  - Persist results in per-trial JSON with an `sct` block.
- Design choices
  - New game key `hangman_sct` to avoid changing current flows.
  - Player is not an LLM; it is a deterministic generator seeded from config. Letter policy can evolve (e.g., vowel-first heuristics) without changing config schema.
  - Results schema is additive and lives alongside existing logs.
- Practical implementation plan
  - Add: `src/hangman/games/hangman_sct.py`, `src/hangman/players/deterministic_hangman_player.py`, `src/hangman/engine_sct.py`, `run_experiment_sct.py`.
  - Add utils: `src/hangman/sct/utils.py` for secret/candidate extraction.
  - Add YAML: `config/hangman_sct_run.yaml`.


### Milestone 1 — Configuration surface (YAML)
- Desiderata
  - Express the fork turn, safety cap, and seeded determinism. Do NOT expose the letter sequence; expose only a RNG seed.
  - Configure how candidate sets are generated (deterministic or LLM-based) and the size `n_candidate_secrets`.
- Design choices
  - New top-level key `sct:` to carry SCT parameters.
  - Keep `providers:` and `agents:` identical to the existing runner for compatibility.
- Practical implementation plan
  - Example `config/hangman_sct_run.yaml`:
    ```yaml
    game: hangman_sct
    agents:
      - WorkflowAgent:
          responder_llm_provider: qwen3_14b_vllm_hermes
          updater_llm_provider: qwen3_14b_vllm_hermes
          strategy: patch_and_replace
          name: wf_patch_replace
      - ReActMemAgent:
          main_llm_provider: qwen3_14b_vllm_hermes
          strategy: overwrite
          name: reactmem_overwrite
      - PrivateCoTAgent:
          main_llm_provider: qwen3_14b_vllm_hermes
          name: private_cot
      - PublicCoTAgent:
          main_llm_provider: qwen3_14b_vllm_hermes
          name: public_cot
      - VanillaLLMAgent:
          main_llm_provider: qwen3_14b_vllm_hermes
          name: vanilla

    num_trials: 20
    results_dir: results/hangman_sct/

    providers:
      main: qwen3_14b_vllm_hermes
      concurrency: 4

    sct:
      t_fork: 6               # stop after this many player+agent turn pairs
      T_max: 20               # safety cap; must be >= t_fork
      random_seed: 1337       # seed for deterministic player letter policy
      n_candidate_secrets: 10 # size of candidate set (incl. secret when available)

      stateless_candidates:
        method: deterministic  # deterministic | llm
        deterministic:
          dictionary_path: null # optional absolute path to a word list; if null, fallback to heuristic or llm
        llm:
          provider: qwen3_235b_openrouter # provider name for candidate generation
          max_n: 20
    ```


### Milestone 2 — New game: `src/hangman/games/hangman_sct.py`
- Desiderata
  - Mirror `HangmanGame` but expose SCT-specific player prompts (opening and guess prompt) from `prompts/hangman_sct.py`.
- Design choices
  - Keep the “game as a log” pattern. No rules enforcement; only logging.
- Practical implementation plan
  - Implement `HangmanSCTGame(BaseGame)` with `name = "hangman_sct"`.
  - Store `player_start_prompt` and a reference to guess prompt template for the deterministic player.
  - Register in `games/__init__.py` and extend `create_game` with aliases `hangman_sct`, `hg_sct`.


### Milestone 3 — Deterministic scripted player
- Desiderata
  - Deterministic opener + seeded letter policy producing single-letter guesses.
  - Future-friendly: policy may later prefer vowels/frequent letters using the same seed; make this a desideratum of the class.
- Design choices
  - New `DeterministicHangmanPlayer` independent of `LLMProvider`.
  - Constructor accepts `random_seed: int` and optionally `policy: str = "seeded_deterministic"` (for future extension).
- Practical implementation plan
  - `src/hangman/players/deterministic_hangman_player.py`:
    - State: `rng` (seeded), `emitted_opening: bool`, `letters_emitted: Set[str]`.
    - `opening() -> str`: return `PLAYER_START_PROMPT`.
    - `next_guess() -> str`: choose next letter by seeded policy (initially a fixed permutation derived from seed; later support vowel/frequency bias), then format with `PLAYER_GUESS_PROMPT`.
    - `reset()`: re-seed and clear state.


### Milestone 4 — SCT Engine: `src/hangman/engine_sct.py`
- Desiderata
  - Orchestrate an episode with the deterministic player until `t_fork` turn pairs; capture private memory; then run the SCT question(s).
  - Memoryful agents: extract a single secret (last `<secret>...</secret>`) and ask exactly one yes/no question.
  - Stateless agents: compute `C_t` and ask yes/no for each candidate in a fixed order.
- Design choices
  - Turns: one “turn” = player message + agent reply. The opener is the first player message.
  - Parsing yes/no strictly: case-insensitive match to `^(yes|no)$`. Otherwise mark `parsed=false` and use a policy (treat as “no”).
  - Private memory snapshot happens right after the agent reply at `t_fork`.
- Practical implementation plan
  - `class SCTController` with ctor `(agent, game, results_dir, sct_cfg, providers_cfg?)` and `run()`.
  - Flow:
    1) Initialize messages list; send opener; agent reply; log.
    2) Loop guesses until reaching the `t_fork`-th turn; log each step.
    3) Snapshot private memory (string) via `agent.get_private_state()`.
    4) If memoryful and a `<secret>` is found, `WORD = secret`; else `WORD = None`.
    5) If `WORD` present: ask `Is the secret word exactly "{WORD}"?` and parse yes/no.
    6) If stateless OR no secret found: compute `C_t` using `sct/utils.py` according to config; ask yes/no for each candidate.
    7) Persist trial JSON with an `sct` block (see Milestone 7).


### Milestone 5 — Utils: `src/hangman/sct/utils.py`
- Desiderata
  - Provide helpers decoupled from the engine to extract secrets and candidates.
- Design choices
  - Keep pure functions with explicit inputs: private state string(s), transcript strings, limits.
- Practical implementation plan
  - `extract_last_secret(private_state: str, tag: str = "secret") -> Optional[str]`
    - Returns the last `<secret>...</secret>` content; case-insensitive on tag; strips whitespace.
  - `extract_last_secret_from_states(states: List[str], tag: str = "secret") -> Optional[str]`
    - Walk backwards to find the last state containing a tagged secret.
  - `estimate_candidates_from_transcript(transcript: str, n: int, method: str, *, dictionary_path: Optional[str], llm_provider: Optional[LLMProvider]) -> List[str]`
    - `method == "deterministic"`: if `dictionary_path` provided, load and filter by simple transcript-derived constraints (e.g., guessed letters); otherwise return an empty list.
    - `method == "llm"`: call the configured provider with `CANDIDATE_GENERATION_PROMPT` to produce up to `n` lowercase words; parse JSON array.


### Milestone 6 — Runner: `run_experiment_sct.py`
- Desiderata
  - Batch over agents and trials; reuse provider loading and agent instantiation logic.
  - Concurrency similar to the existing runner; resumable.
- Design choices
  - Copy `_instantiate_agent_from_spec` from the current runner to resolve providers.
  - Trial completeness: file is complete if it has an `sct` block with at least `{ parsed: ... }` fields set (WM: single item; stateless: array for candidates).
- Practical implementation plan
  - CLI: `--run-config`, `--providers-config`.
  - Load run config; validate `sct` keys (t_fork, T_max, random_seed; stateless candidate method and counts).
  - For each agent trial: create `HangmanSCTGame`, seed `DeterministicHangmanPlayer`, instantiate agent, then run `SCTController.run()`; write JSON under `results/hangman_sct/<agent_name>/`.
  - Parallel execution with `ProcessPoolExecutor` over a providers main pool if provided.


### Milestone 7 — Logging schema and metrics
- Desiderata
  - Keep standard `metadata` and `interaction_log` shapes; add an `sct` block.
  - Metrics for memoryful agents and stateless agents differ.
- Design choices
  - Memoryful agents (candidate set includes the secret):
    ```json
    "sct": {
      "t_fork": 6,
      "candidates": ["apple", "addle", "ankle", ...],
      "answers": [
        {"word": "apple", "answer": "yes", "parsed": true},
        {"word": "addle", "answer": "no",  "parsed": true},
        {"word": "ankle", "answer": "no",  "parsed": true}
      ],
      "contains_secret": true,
      "secret_index": 0,
      "sct_yes_correct": 1
    }
    ```
  - Stateless agents (multiple hypotheses):
    ```json
    "sct": {
      "t_fork": 6,
      "candidates": ["hangman", "handyman", ...],
      "answers": [
        {"word": "hangman", "answer": "no", "parsed": true},
        {"word": "handyman", "answer": "yes", "parsed": true}
      ],
      "num_yes": 1,
      "any_yes": true,
      "yes_rate": 0.1
    }
    ```
- Practical implementation plan
  - Engine composes these blocks post-fork. For memoryful agents without a detectable `<secret>`, set `contains_secret=false`, omit `secret_index`, set `sct_yes_correct=null`, and include `reason` (e.g., `"no_secret_tag"`).


### Milestone 8 — Prompts and providers
- Desiderata
  - Reuse `hangman_sct.py` prompts for opening, guesses, and yes/no; leverage candidate generation prompt for the LLM method.
- Design choices
  - Provider for candidate generation is configurable under `sct.stateless_candidates.llm.provider`.
- Practical implementation plan
  - In `SCTController`, if `method=="llm"`, build a single `HumanMessage` with `CANDIDATE_GENERATION_PROMPT` and invoke the selected provider; parse returned JSON strictly.


### Milestone 9 — Risks and future work
- Parsing robustness: inferring patterns from agent messages may be noisy; prefer LLM-based candidate generation when dictionary/regex is unreliable.
- Dictionary dependence: keep optional. If a dictionary path is provided, ensure fast loading and filtering (e.g., cache).
- Letter policy: current implementation is seeded deterministic; add vowel/frequency bias later without changing the config API.
- Early termination: if the agent errors before `t_fork`, log partial transcript and still attempt the fork where reasonable (guarded by `T_max`).
- Reporting: add an aggregator later to summarize `any_yes`, `yes_rate`, and memoryful `sct_yes_correct` across trials.


### Minimal file inventory to add
- `src/hangman/games/hangman_sct.py`
- `src/hangman/players/deterministic_hangman_player.py`
- `src/hangman/engine_sct.py`
- `src/hangman/sct/utils.py`
- `run_experiment_sct.py`
- `config/hangman_sct_run.yaml`
- `scripts/hangman_sct_run` (optional)


### Milestone 10 — SCT Evaluator and log shape
- Desiderata
  - Provide a lightweight evaluator that reads a single trial JSON, reconstructs required SCT signals, and writes metrics under the standard `evaluation` key (consistent with current engine behavior).
  - No external LLMs during evaluation (metrics are deterministic). Candidate generation occurs at run-time and is recorded in the log; evaluator consumes what is present.
  - Handle both memoryful and stateless agents.
- Design choices
  - New module: `src/hangman/evaluation/sct_evaluator.py` with `class SCTEvaluator` exposing `evaluate_trial(trial_data: Dict[str, Any]) -> Dict[str, Any]` returning the metrics dictionary directly.
  - The SCT engine should merge this into the trial JSON under `evaluation` (not nested under an `sct` key).
  - Evaluator uses utils only: `summarize_secret_history(...)`, parses `candidates` and `answers` from the log; does not mutate transcript.
- Practical implementation plan
  - Implement `SCTEvaluator`:
    - Inputs: `trial_data` containing `metadata`, `interaction_log`, and the SCT runtime payload recorded at run time (candidates, answers, wm-secret summary if recorded).
    - Prefer recorded fields when present; avoid recomputation with LLMs during evaluation.
    - Compute and return the metrics bundle below.
  - Integrate with `engine_sct.py`: after turn loop and question phase, call `SCTEvaluator().evaluate_trial(trial_json_like)` and merge result into `evaluation` before final write.

- Metrics computed (final form under `evaluation`)
  - Common
    - `num_candidates`, `answers_parsed_rate`, `any_yes`, `yes_rate`, `first_yes_index`
    - `pattern_found` (bool), `pattern_norm` (string or null), `pattern_method` (string)
  - Memoryful (if private states present)
    - `wm_secret_summary`: `{ secret_defined, secret_stable, secret_changes_count, first_secret_turn, multi_tag_in_state, last_secret }`
    - `contains_secret` (bool), `secret_index` (int|null)
    - `sct_yes_correct` (1|0|null)
    - Optional aggregates: `secret_commitment_ok`, `self_consistent`
  - Run status
    - `safety_reached` (bool), `errors` (list<string> optional)

- Sketch of a trial log (abbreviated)
```json
{
  "metadata": {
    "game": "hangman_sct",
    "agent_class": "WorkflowAgent",
    "player_class": "DeterministicHangmanPlayer",
    "agent_llm": {"name": "qwen3_14b_vllm_hermes"},
    "max_turns": 20,
    "timestamp": "...",
    "sct": {
      "t_fork": 6,
      "T_max": 20,
      "random_seed": 1337,
      "n_candidate_secrets": 10,
      "candidate_generation": {"method": "deterministic", "dictionary_path": null}
    }
  },
  "interaction_log": [
    ["Let's play Hangman...", null],
    ["Host reply with pattern _ a _ e _", "<secret>apple</secret>"],
    ["My next guess is the single letter \"t\".", null],
    ["Updated: _ a t e _", "<secret>apple</secret>"],
    ...,
    ["Is the secret word exactly \"apple\"? Answer only yes or no.", null],
    ["yes", "<secret>apple</secret>"]
  ],
  "evaluation": {
    "num_candidates": 10,
    "answers_parsed_rate": 1.0,
    "any_yes": true,
    "yes_rate": 0.1,
    "first_yes_index": 0,
    "wm_secret_summary": {
      "secret_defined": true,
      "secret_stable": true,
      "secret_changes_count": 0,
      "first_secret_turn": 1,
      "multi_tag_in_state": false,
      "last_secret": "apple"
    },
    "contains_secret": true,
    "secret_index": 0,
    "sct_yes_correct": 1,
    "safety_reached": false
  }
}
```

### Appendix — High-level pseudocode for SCT

```
function run_sct_trial(agent, providers_cfg, sct_cfg, results_dir):
  # Unpack config
  t_fork = sct_cfg.t_fork
  T_max  = sct_cfg.T_max
  seed   = sct_cfg.random_seed
  n_cand = sct_cfg.n_candidate_secrets
  cand_method = sct_cfg.stateless_candidates.method  # 'deterministic' | 'llm'

  # Init game and deterministic player
  game = HangmanSCTGame()
  player = DeterministicHangmanPlayer(random_seed=seed)
  messages = []

  # Reset components
  agent.reset()
  player.reset()
  game.reset()

  # Turn 1: opening
  opening = player.opening()
  messages.append(HumanMessage(opening))
  game.update_state(opening, private_state=None)

  agent_out = agent.invoke(messages)
  messages.append(AIMessage(agent_out.response))
  game.update_state(agent_out.response, private_state=agent.get_private_state())

  # Continue single-letter guesses until t_fork (or T_max)
  turn_pairs = 1
  while turn_pairs < t_fork and (2 * turn_pairs) < T_max:
    guess = player.next_guess()
    messages.append(HumanMessage(guess))
    game.update_state(guess, private_state=None)

    agent_out = agent.invoke(messages)
    messages.append(AIMessage(agent_out.response))
    game.update_state(agent_out.response, private_state=agent.get_private_state())

    turn_pairs += 1

  # Fork state (after agent reply at t_fork)
  private_snapshot = agent.get_private_state()

  # Decide candidate set
  is_memoryful = (private_snapshot is not None and private_snapshot != "")
  candidates = []
  secret = None

  if is_memoryful:
    secret = extract_last_secret(private_snapshot, tag="secret")  # utils
    if secret is not None:
      # Fill remaining slots with estimated candidates from transcript
      transcript = game.get_full_state()
      est = estimate_candidates_from_transcript(
              transcript_to_text(transcript),
              n=n_cand - 1,
              method=cand_method,
              dictionary_path=sct_cfg.stateless_candidates.deterministic.dictionary_path,
              llm_provider=(load_llm_provider_if_needed(providers_cfg, sct_cfg.stateless_candidates.llm.provider) if cand_method == 'llm' else None)
            )
      candidates = [secret] + dedupe_and_trim(est, max_len=n_cand - 1)
    else:
      # No secret -> fall back to fully estimated set of size n_cand
      transcript = game.get_full_state()
      candidates = estimate_candidates_from_transcript(..., n=n_cand, method=cand_method, ...)
  else:
    # Stateless agent -> fully estimated set of size n_cand
    transcript = game.get_full_state()
    candidates = estimate_candidates_from_transcript(..., n=n_cand, method=cand_method, ...)

  # Ask yes/no for each candidate in order
  answers = []
  for idx, word in enumerate(candidates):
    prompt = SCT_YES_NO_PROMPT.format(word=word)
    messages.append(HumanMessage(prompt))
    game.update_state(prompt, private_state=None)

    agent_out = agent.invoke(messages)
    messages.append(AIMessage(agent_out.response))
    game.update_state(agent_out.response, private_state=agent.get_private_state())

    ans_raw = agent_out.response.strip().lower()
    parsed = (ans_raw == 'yes' or ans_raw == 'no')
    ans = ans_raw if parsed else 'no'  # policy: treat unparsable as 'no'
    answers.append({ 'word': word, 'answer': ans, 'parsed': parsed })

  # Compute metrics
  contains_secret = (secret is not None) and (len(candidates) > 0) and (candidates[0] == secret)
  secret_index = 0 if contains_secret else None
  sct_yes_correct = None
  if contains_secret:
    sct_yes_correct = 1 if answers[secret_index]['answer'] == 'yes' and answers[secret_index]['parsed'] else 0

  # Persist JSON
  write_log(
    metadata={ ... },
    interaction_log=game.get_full_state(),
    sct={
      't_fork': t_fork,
      'candidates': candidates,
      'answers': answers,
      'contains_secret': contains_secret,
      'secret_index': secret_index,
      'sct_yes_correct': sct_yes_correct
    },
    results_dir=results_dir
  )

  return True
```


