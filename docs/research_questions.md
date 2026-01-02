## Research Questions

- **RQ1 — Public-only agents vs PSITs**: Can public-only chat agents (POCAs) coherently simulate PSITs without any private state, or do they empirically fail?
  - Show: Secrecy–Coherence scatter (one point per config); highlight POCAs vs memoryful agents with CIs.

- **RQ2 — Workflow vs ReAct-with-tools**: Does a two-stage Workflow agent (response → memory update) outperform a single‑LLM ReAct‑with‑tools agent on secrecy and behavioral coherence?
  - Show: Paired delta-bar (Workflow − ReAct) for Secrecy and Coherence per task with bootstrap CIs.

- **RQ3 — Update strategy effects**: How do memory‑update strategies (overwrite, patch‑and‑replace, append‑and‑delete) impact private memory length/compactness and overall performance across metrics?
  - Show: Compact table per strategy: median memory length (tokens), coherence score, leak rate (±CI).

- **RQ4 — Memory representation**: Does structuring working memory into Goals / Knowledge / Notes improve behavioral coherence compared to a flat scratchpad representation of private state?
  - Show: Bar chart comparing coherence (structured vs flat) across tasks (means ±CI).

- **RQ6 — Model family and scale**: How do model family and size (e.g., Qwen vs GPT; parameter count) systematically affect secrecy, coherence, mechanism, and winner metrics?
  - Show: Line plot of coherence vs log(parameters) with separate lines for Qwen/GPT; leak-rate annotated.

- **RQ7 — Winner and sycophancy**: Does conditioning on a private secret increase an agent’s probability of winning a PSIT? Do non‑memory agents exhibit sycophancy that inflates the Player’s win rate?
  - Show: Single table: Player win rate (%) and sycophancy proxy (%) per agent family (POCA, PrivateCoT, ReAct, Workflow).

- **RQ8 — Task dependence**: How task‑dependent are secrecy and intentionality scores across PSITBench tasks (Hangman, 20 Questions, Diagnosis Simulator)?
  - Show: Task × metric summary table with mean secrecy and intentionality (±CI) per task.


