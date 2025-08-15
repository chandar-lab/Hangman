# patch_memory.py
# Working-Memory Patch Tool (section-anchored, V4A-inspired)
# ----------------------------------------------------------
# This module provides a LangChain @tool named `patch_memory` that applies a
# context-based patch to a single "working memory" string. It does not use line
# numbers; instead each hunk anchors to a memory section by title, and uses `-`
# and `+` lines to describe deletions and insertions. All writes are atomic:
# the final "new_memory" is computed and returned together with metadata.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from hangman.tools.memory_utils import (
    _Document,
    _Options,
    PatchError,
    _Section,
    _Hunk,
    _Patch,
)

# =========================
# Public tool: patch_memory
# =========================

@tool
def patch_memory(
    patch: str,
    explanation: str,
    expected_hunks: Optional[int] = None,
    expected_changes: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    current_memory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply a context-based patch to the working memory string (no line numbers).

    Each patch contains one or more hunks, *anchored to a memory section* using
    a header of the form: `@@ section: <Title>`.
    Within a section, specify deleted lines with '-' and added lines with '+',
    optionally including 1–3 unchanged context lines before/after for disambiguation.

    - If a section or target is ambiguous/missing, the tool FAILS with a diagnostic.
    - The patch should be IDEMPOTENT: re-applying it should not corrupt memory.
    - Returns the updated memory plus metadata (applied hunks, changed lines, etc.).

    Required args:
      - patch (string): The patch text with the required headers:
          ```
          *** Begin Patch
          *** Update Memory
          @@ section: Goals and Plans
          - Old line
          + New line
          *** End Patch
          ```
      - explanation (string): one-sentence high-level intent for the change.

    Optional safety args:
      - expected_hunks (int): how many hunks you intend to apply.
      - expected_changes (int): total +/- line operations expected.
      - options (dict):
          * strict_context (bool, default False): require clear context or exact old block.
          * normalize_whitespace (bool, default True): collapse repeated spaces when matching.
          * case_sensitive (bool, default True): literal, case sensitive matching when True.

    Integration arg (injected by agent):
      - current_memory (string): the current working-memory string to patch.

    Returns:
      {
        "new_memory": <updated string>,
        "meta": {
          "applied_hunks": <int>,
          "changed_lines": <int>,
          "sections_touched": [<str>...],
          "warnings": [<str>...]
        }
      }

    ----------------
    Few-shot examples
    ----------------

    Example A: simple replacement within a section
    ```
    *** Begin Patch
    *** Update Memory
    @@ section: Goals and Plans
    - Planned steps or strategies for achieving them.
    + Planned steps: finish MVP, write docs, ship v1 by Sept 15.
    *** End Patch
    ```

    Example B: multi-line update with light context
    ```
    *** Begin Patch
    *** Update Memory
    @@ section: Active Knowledge
    [Investigation notes]
    - Hypothesis: caching is the bottleneck
    - Evidence: slow Redis reads
    + Hypothesis (confirmed): caching is the bottleneck
    + Evidence: slow Redis reads; profiler shows 45% time in cache layer
    *** End Patch
    ```

    Example C: pure insertion (anchor with a nearby context line)
    ```
    *** Begin Patch
    *** Update Memory
    @@ section: Goals and Plans
    [Milestones]
    + - Milestone: pass integration tests in CI
    *** End Patch
    ```
    """
    print(f"---TOOL: patch_memory---")

    memory = current_memory or ""
    opts = _Options.from_dict(options or {})

    # Parse memory into sections
    doc = _Document.parse(memory)

    # Parse patch into hunks
    patch_obj = _Patch.parse(patch)

    # Optional safety expectation: number of hunks
    if expected_hunks is not None and expected_hunks != len(patch_obj.hunks):
        raise PatchError(
            f"expected_hunks={expected_hunks} but parsed {len(patch_obj.hunks)} hunks."
        )

    # Apply hunks
    total_changes = 0
    sections_touched: List[str] = []
    warnings: List[str] = []

    for hunk in patch_obj.hunks:
        # Find the target section by normalized title
        section = doc.find_section_by_title(hunk.section_title)
        if section is None:
            raise PatchError(
                f"Section not found: '{hunk.section_title}'. "
                f"Available sections: {', '.join(doc.list_titles())}"
            )

        # Apply hunk to the section body (excludes the '## n. Title' header line)
        new_body, changed_lines, hunk_warnings = _apply_hunk_to_section(
            section.body,
            hunk,
            opts,
        )

        if changed_lines > 0:
            section.body = new_body
            total_changes += changed_lines
            if section.title not in sections_touched:
                sections_touched.append(section.title)
        else:
            # Idempotent/no-op case (already up-to-date)
            warnings.extend(hunk_warnings or ["Hunk produced no change (idempotent)."])

    # Reassemble the updated memory
    new_memory = doc.render()

    # Optional safety expectation: number of total +/- operations applied
    if expected_changes is not None and expected_changes != total_changes:
        raise PatchError(
            f"expected_changes={expected_changes} but applied {total_changes} +/- line ops."
        )

    return {
        "new_memory": new_memory,
        "meta": {
            "applied_hunks": len(patch_obj.hunks),
            "changed_lines": total_changes,
            "sections_touched": sections_touched,
            "warnings": warnings,
        },
    }


# =======================
# Matching & application
# =======================

def _apply_hunk_to_section(
    section_body: str,
    hunk: _Hunk,
    opts: _Options,
) -> Tuple[str, int, List[str]]:
    """
    Apply a single hunk to the given section body.
    Returns: (new_section_body, changed_lines_count, warnings[])
    """
    text = section_body
    warnings: List[str] = []

    # Normalization functions for matching
    def norm(s: str) -> str:
        if not opts.case_sensitive:
            s = s.lower()
        if opts.normalize_whitespace:
            s = re.sub(r"[ \t]+", " ", s)
        return s

    def find_unique(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
        """
        Return (start, end) of a unique occurrence in `haystack`, or None if 0 or >1.
        - If normalize_whitespace=True, build a regex that treats any run of spaces/tabs
          in `needle` as `[ \t]+` and match directly on the ORIGINAL haystack to get
          correct spans. No approximate back-mapping.
        - Otherwise, do a literal search respecting case sensitivity.
        """
        if needle == "":
            return None

        if opts.normalize_whitespace:
            # Build a regex that replaces any run of [ \t]+ in the needle with [ \t]+
            def _build_ws_relaxed_pattern(s: str) -> str:
                parts = re.split(r"[ \t]+", s)
                # Keep empty parts out to avoid accidental anchors on nothing
                parts = [p for p in parts if p != ""]
                if not parts:
                    # The needle is only whitespace; reject
                    return r"(?!x)x"  # never matches
                return r"[ \t]+".join(re.escape(p) for p in parts)

            pattern = _build_ws_relaxed_pattern(needle)
            flags = 0 if opts.case_sensitive else re.IGNORECASE
            matches = list(re.finditer(pattern, haystack, flags))
            if len(matches) == 1:
                return matches[0].span()
            return None

        # Literal path (no normalization)
        hay = haystack if opts.case_sensitive else haystack.lower()
        ned = needle if opts.case_sensitive else needle.lower()

        # Find all occurrences; we only accept a unique one
        idxs = []
        start = 0
        while True:
            pos = hay.find(ned, start)
            if pos == -1:
                break
            idxs.append(pos)
            start = pos + 1
            if len(idxs) > 1:
                # Early exit if more than one occurrence
                return None

        if len(idxs) == 1:
            start = idxs[0]
            end = start + len(needle)
            return (start, end)

        return None

    # Build anchors
    minus_block = "\n".join(hunk.minus).strip("\n")
    plus_block = "\n".join(hunk.plus).strip("\n")
    pre_ctx_block = "\n".join(hunk.pre_context).strip("\n")
    post_ctx_block = "\n".join(hunk.post_context).strip("\n")

    changed_lines = 0

    # Case 1: Replacement / Deletion (minus present)
    if minus_block:
        # Try strongest anchor first: pre + minus + post (when available)
        candidates = []
        if pre_ctx_block and post_ctx_block:
            candidates.append(pre_ctx_block + "\n" + minus_block + "\n" + post_ctx_block)
        if pre_ctx_block and not post_ctx_block:
            candidates.append(pre_ctx_block + "\n" + minus_block)
        if post_ctx_block and not pre_ctx_block:
            candidates.append(minus_block + "\n" + post_ctx_block)
        # Always include the bare minus block
        candidates.append(minus_block)

        start_end = None
        for needle in candidates:
            start_end = find_unique(text, needle)
            if start_end:
                # If needle included context, shrink to the minus span within it
                if needle != minus_block:
                    # Find minus span inside the matched window
                    window = text[start_end[0]:start_end[1]]
                    inner = find_unique(window, minus_block)
                    if inner:
                        start = start_end[0] + inner[0]
                        end = start_end[0] + inner[1]
                        start_end = (start, end)
                break

        if start_end is None:
            # Idempotency check: maybe plus already present (replacement already made)
            if plus_block and find_unique(text, plus_block):
                warnings.append("Minus block not found, plus block already present (idempotent).")
                return text, 0, warnings

            msg = "Target for replacement not found uniquely in section."
            if opts.strict_context:
                msg += " Provide more context lines (pre/post) to disambiguate."
            raise PatchError(msg)

        start, end = start_end
        before = text[:start]
        after = text[end:]

        # Replace with plus (may be empty for deletion)
        new_text = before
        new_text += plus_block if plus_block else ""
        # Ensure newline integrity when removing/adding blocks
        if (plus_block and not plus_block.endswith("\n")) and (after.startswith("\n")):
            # keep the existing newline from the remainder
            pass
        new_text += after

        # Count changes as +/- line count
        changed_lines += (len(hunk.minus) + len(hunk.plus))
        return new_text, changed_lines, warnings

    # Case 2: Insertion-only (no minus; must have some pre-context anchor)
    if not minus_block and plus_block:
        # Allow insertion at start if section is empty
        if not text.strip():
            text = "\n" 
            insertion_point = len(text)
        elif not pre_ctx_block:
            if opts.strict_context:
                raise PatchError(
                    "Insertion-only hunk requires pre-context to anchor (strict_context=True)."
                )
            # non-strict: append at end of section
            insertion_point = len(text)
        else:
            # Allow inserting right after the section header
            header_line = f"## {section_title}"
            if len(pre_ctx_block) == 1 and pre_ctx_block[0].strip() == header_line.strip():
                # Insert immediately after header
                insertion_point = text.index("\n") + 1  # first newline after header
            else:
                anchor = find_unique(text, pre_ctx_block)
                if anchor is None:
                    if not text.strip():
                        text = "\n"
                        insertion_point = len(text)
                    else:
                        raise PatchError(
                            "Insertion anchor (pre-context) not found uniquely in section."
                        )
                else:
                    insertion_point = anchor[1]

        # Compose insertion
        prefix = text[:insertion_point]
        suffix = text[insertion_point:]
        needs_nl_before = (
            len(prefix) > 0 and not prefix.endswith("\n") and (suffix.startswith("\n") or suffix == "")
        )

        new_text = prefix
        if needs_nl_before:
            new_text += "\n"
        new_text += plus_block

        # If inserting before existing content, ensure the inserted block ends with \n
        if suffix and not plus_block.endswith("\n"):
            new_text += "\n"

        new_text += suffix

        # --- FIX: when appending at end (suffix == ""), ensure final newline ---
        if not suffix and not new_text.endswith("\n"):
            new_text += "\n"

        changed_lines += len(hunk.plus)
        return new_text, changed_lines, warnings

    # Case 3: No minus and no plus (noop hunk)
    warnings.append("Empty hunk (no +/-) had no effect.")
    return text, 0, warnings


# ==========================
# Optional: simple self-test
# ==========================

if __name__ == "__main__":
    """
    Ad-hoc test runner for the `patch_memory` tool.
    These tests exercise success paths, idempotency, ambiguity handling,
    options (case/whitespace/strictness), and safety expectations.
    """

    def hr(title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def run_test(name: str, patch_text: str, memory: str, **kwargs):
        hr(f"TEST: {name}")
        try:
            args = {
                "patch": patch_text,
                "explanation": kwargs.pop("explanation", name),
                "current_memory": memory,
            }
            args.update(kwargs)
            result = patch_memory.invoke(args)
            print("RESULT new_memory:\n", result["new_memory"])
            print("RESULT meta:\n", result["meta"])
            return result
        except PatchError as e:
            print("PATCH ERROR:", str(e))
            return None
        except Exception as e:
            print("UNEXPECTED ERROR:", type(e).__name__, str(e))
            return None

    # ----------------------------------------------------------------------------------
    # Baseline working memory
    # ----------------------------------------------------------------------------------
    initial_memory = """## 1. Goals and Plans

- Current overarching goal.
- Subgoals or milestones.
- Planned steps or strategies for achieving them.

## 3. Active Knowledge

[Investigation notes]
- Hypothesis: caching is the bottleneck
- Evidence: slow Redis reads
"""

    # ----------------------------------------------------------------------------------
    # 1) Simple single-hunk replacement inside Goals and Plans
    # ----------------------------------------------------------------------------------
    patch_1 = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
- Planned steps or strategies for achieving them.
+ Planned steps: finish MVP, write docs, ship v1 by Sept 15.
*** End Patch"""
    r1 = run_test(
        "simple replacement in Goals and Plans",
        patch_1,
        initial_memory,
        expected_hunks=1,
        expected_changes=2,
    )
    mem_1 = r1["new_memory"] if r1 else initial_memory

    # ----------------------------------------------------------------------------------
    # 2) Idempotency: re-apply the same patch (should warn, 0 changes)
    # ----------------------------------------------------------------------------------
    r2 = run_test(
        "idempotent reapply",
        patch_1,
        mem_1,
        expected_hunks=1,
    )

    # ----------------------------------------------------------------------------------
    # 3) Multi-hunk: update hypothesis + insert new milestone with context
    # ----------------------------------------------------------------------------------
    patch_2 = """*** Begin Patch
*** Update Memory
@@ section: Active Knowledge
[Investigation notes]
- Hypothesis: caching is the bottleneck
+ Hypothesis (confirmed): caching is the bottleneck

@@ section: Goals and Plans
- Subgoals or milestones.
+ Subgoals or milestones.
+ - Milestone: pass integration tests in CI
*** End Patch"""
    r3 = run_test(
        "multi-hunk: confirm hypothesis + insert milestone",
        patch_2,
        mem_1,
        expected_hunks=2,
    )
    mem_2 = r3["new_memory"] if r3 else mem_1

    # ----------------------------------------------------------------------------------
    # 4) Ambiguity failure: minus block too generic (appears twice)
    #    Here we try to delete the word 'goal' (too generic), expecting a failure.
    # ----------------------------------------------------------------------------------
    patch_ambiguous = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
- goal
+ objective
*** End Patch"""
    _ = run_test(
        "ambiguity failure (too generic minus)",
        patch_ambiguous,
        mem_2,
    )

    # ----------------------------------------------------------------------------------
    # 5) Section not found: typo in section title
    # ----------------------------------------------------------------------------------
    patch_bad_section = """*** Begin Patch
*** Update Memory
@@ section: Goalz / Plans
- Current overarching goal.
+ Current overarching goal (updated).
*** End Patch"""
    _ = run_test(
        "section not found",
        patch_bad_section,
        mem_2,
    )

    # ----------------------------------------------------------------------------------
    # 6) Insertion-only with strict_context=True (missing anchor) -> expect failure
    # ----------------------------------------------------------------------------------
    patch_insert_no_anchor = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
+ - Milestone: write end-to-end tests
*** End Patch"""
    _ = run_test(
        "insertion-only without anchor (strict) -> fail",
        patch_insert_no_anchor,
        mem_2,
        options={"strict_context": True},
    )

    # ----------------------------------------------------------------------------------
    # 7) Insertion-only with strict_context=False (append to section)
    # ----------------------------------------------------------------------------------
    _ = run_test(
        "insertion-only append (non-strict)",
        patch_insert_no_anchor,
        mem_2,
        options={"strict_context": False},
    )

    # ----------------------------------------------------------------------------------
    # 8) Expected_hunks mismatch -> expect failure
    # ----------------------------------------------------------------------------------
    _ = run_test(
        "expected_hunks mismatch",
        patch_2,
        mem_2,
        expected_hunks=99,  # incorrect on purpose
    )

    # ----------------------------------------------------------------------------------
    # 9) Expected_changes mismatch -> expect failure
    # ----------------------------------------------------------------------------------
    _ = run_test(
        "expected_changes mismatch",
        patch_1,
        mem_2,
        expected_hunks=1,
        expected_changes=999,  # incorrect on purpose
    )

    # ----------------------------------------------------------------------------------
    # 11) Case-insensitive match (options.case_sensitive=False)
    # ----------------------------------------------------------------------------------
    patch_case_insensitive = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
- Current overarching goal.
+ Current overarching goal: grow user base to 1k MAU.
*** End Patch"""
    _ = run_test(
        "case-insensitive section matching",
        patch_case_insensitive,
        mem_2,
        options={"case_sensitive": False},
    )

    # ----------------------------------------------------------------------------------
    # 12) Whitespace-normalized match (options.normalize_whitespace=True)
    #     Simulate minus/plus with extra spaces or tabs
    # ----------------------------------------------------------------------------------
    mem_ws = mem_2.replace("Subgoals or milestones.", "Subgoals    or\tmilestones.")
    patch_ws = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
- Subgoals or milestones.
+ Subgoals or milestones.
*** End Patch"""
    _ = run_test(
        "normalize whitespace matching",
        patch_ws,
        mem_ws,
        options={"normalize_whitespace": True},
    )

    # ----------------------------------------------------------------------------------
    # 13) Deletion-only: remove an evidence line
    # ----------------------------------------------------------------------------------
    patch_delete_only = """*** Begin Patch
*** Update Memory
@@ section: Active Knowledge
[Investigation notes]
- Evidence: slow Redis reads
*** End Patch"""
    _ = run_test(
        "deletion-only (minus without plus)",
        patch_delete_only,
        mem_2,
    )

    # ----------------------------------------------------------------------------------
    # 14) Empty hunk (no +/- and no context) -> warn/no-op
    # ----------------------------------------------------------------------------------
    patch_empty = """*** Begin Patch
*** Update Memory
@@ section: Goals and Plans
*** End Patch"""
    _ = run_test(
        "empty hunk noop",
        patch_empty,
        mem_2,
    )

    hr("DONE")