Guiding principles (from Cursor/Copilot/Gemini)
	•	Have a primary, expressive edit path (diff/patch or precise edit).
	•	Provide at least one fallback that is simpler/robuster when the primary fails.
	•	Use context—not line numbers— to target changes and avoid misapplies.
	•	Validate and count occurrences before writing; then write full content atomically.
	•	Add “healing” (auto-corrections for quoting/escaping/mismatch) only if needed.
	•	Keep notebooks separate (not relevant here).

⸻

Representing the memory

Even if you keep it as a string, pick a light structure so tools can target reliably:
	•	Numbered bullets (your current approach): easy to address by index, but brittle to reordering.
	•	Tagged bullets (e.g., - [tag1, tag2] content) to allow selectors.
	•	Section headers (e.g., ## Facts, ## Tasks, ## Hypotheses) to scope edits.
	•	Key–value YAML for critical slots (e.g., high-value facts), with a freeform notes section below.

You can start with “section headers + bullets” and evolve later.

⸻

Core tools (recommended set)

1) patch_memory (Primary; V4A-style patch but fileless)

When: multi-edit updates, rewording across the string, reordering sections.

Input idea:

*** Begin Patch
*** Update Memory
@@ section: Facts
- User’s advisor: TBD
+ User’s advisor: Prof. X
@@ section: Hypotheses
- LLM tends to forget deadlines
+ LLM tends to forget deadlines; pin key dates
*** End Patch

Behavior: Parse, apply hunks by context (“@@ section: NAME” + a few unchanged lines). No line numbers. If context doesn’t match uniquely, fail. Internally compute the new full string, then overwrite.

Why: Mirrors the “apply_patch” expressiveness without file paths; robust for grouped edits.

⸻

2) replace_in_memory (Surgical, literal replace with counts)

When: exact text substitution; low risk.

Args: old_string, new_string, expected_replacements (default 1).

Guards:
	•	Require 3–5 lines (or at least a section header + a couple of neighbors) around the target in old_string.
	•	Compute actual occurrence count; if ≠ expected, fail.
	•	Optionally heal trivial escaping issues.

Why: This is the safest “make this one thing different” tool.

⸻

3) edit_items (Structure-aware list editing)

When: you maintain bullets/sections and want stable addressing.

Args:
	•	target: selector (by section name + item index, or by tag, or “contains”).
	•	op: one of insert|update|delete|reorder|upsert.
	•	payload: text (for insert/update), or new index (for reorder).

Rules:
	•	Disallow ambiguous selectors (must resolve to exactly one section/item unless op=insert).
	•	Support append to section without supplying indices.
	•	For delete/update, require neighbor context in payload or a unique tag to avoid wrong matches.

Why: Line numbers are brittle; selectors + minimal context make small operations predictable.

⸻

4) rewrite_memory (Summarize/compact/normalize)

When: memory is too long or needs normalization.

Args: instructions (e.g., “dedupe repetitive facts; keep pinned items; enforce 1 screen length”), optional budget_tokens.

Behavior: Produce a compact version that preserves pinned/critical items, then replace the entire memory atomically.

Why: Token budget control & hygiene—crucial for long-running agents.

⸻

Optional helpers (nice to have)
	•	pin_items / set_ttl: annotate entries as pinned or with expiry hints ([ttl: 7d]) so the summarizer keeps or drops them.
	•	rename_section / merge_sections: structural reshaping.
	•	validate_memory: check invariants (max size, required sections present).
	•	diff_preview (internal only): compute a unified diff for logs/telemetry, though you said no UX is needed.

⸻

How these map to SoTA patterns

SoTA Pattern	Your Tool	Notes
Contextual patch (V4A)	patch_memory	Primary path for multi-spot edits; no line numbers.
Literal replace	replace_in_memory	With expected_replacements and context requirement.
Hinted edit fallback	edit_items	Selector + minimal surrounding context instead of ellipses.
Full-file rewrite	All tools write full string	Compute new content, then replace atomically.
Healing	(Optional) in replace/patch	Unescape quotes, fix trivial mismatches; still fail if ambiguous.
Notebook special-case	N/A	Notebooks irrelevant here.


⸻

Failure & safety rules (crucial)
	•	Ambiguity = failure: if a selector or context matches 0 or >1 places, fail with a diagnostic that invites a more specific call.
	•	Atomic write with revision: require a revision_id (or hash) to guard against concurrent edits; reject if stale.
	•	Size constraints: enforce a max length and have rewrite_memory to compact automatically when near the limit.
	•	Namespaces: reserve a top “System/Policy” section the agent cannot edit; tools must not touch it.
	•	Telemetry: log the before/after size, changed sections, and counts—not the full content if privacy-sensitive.

⸻

Choosing how many tools
	•	Minimum viable: 2 tools
	•	patch_memory (primary) + replace_in_memory (surgical fallback).
	•	Practical sweet spot: 3–4 tools
	•	Add edit_items for structure-aware ops and rewrite_memory for compaction.
	•	Avoid: line-number edits; they’re brittle as memory shifts.

⸻

Example decision policy (what the agent should try first)
	1.	If multiple edits or structure changes across sections → patch_memory.
	2.	If “change exactly this phrase/snippet” → replace_in_memory.
	3.	If “insert/delete/move one bullet in a known section” → edit_items.
	4.	If memory near limit or messy → rewrite_memory (then proceed).

⸻

Implementation notes (non-code)
	•	Context anchors: For patch hunks, use anchors like @@ section: NAME and optionally item headers like @@ item: startsWith("…").
	•	Occurrence counting: In replace_in_memory, compute exact match count before mutation; compare to expected_replacements.
	•	Healing scope: Keep it narrow (e.g., fix \" vs "), otherwise you risk unintended matches.
	•	Determinism: Avoid stochastic steps in core application; use LLMs only to generate candidates, not to apply them.
	•	Testing: Fuzz tests against reorder, whitespace, UTF-8, and duplicate items.

⸻

Bottom line

Adopt a hybrid edit stack:
	•	patch_memory (contextual, diff-like, no line numbers) as primary.
	•	replace_in_memory as surgical fallback with expected_replacements.
	•	edit_items for structured, selector-based micro-edits.
	•	rewrite_memory for compaction and hygiene.

This mirrors the proven patterns (Copilot/Cursor/Gemini), but simplified for a single in-memory string, giving you robustness without UI complexity.