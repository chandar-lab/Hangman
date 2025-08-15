# replace_in_memory.py
# Working-Memory Replace Tool (surgical, section-scoped, line-number–free)
# -----------------------------------------------------------------------
# This module provides a LangChain @tool named `replace_in_memory` that performs
# precise, literal replacements inside a single "working memory" string.
# It is designed to be fully compatible with the helpers and conventions used by
# `patch_memory` (section anchoring, optimistic concurrency, return shape, options).

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

# Reuse internals from patch_memory for consistent parsing, errors, options.
from hangman.tools.memory_utils import (
    _Document,
    _Options,
    PatchError,
)

# ----------------------
# Public tool definition
# ----------------------

@tool
def replace_in_memory(
    old_string: str,
    new_string: str,
    explanation: str,
    section_title: Optional[str] = None,
    expected_replacements: Optional[int] = 1,
    pre_context: Optional[str] = None,
    post_context: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    current_memory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Replace an exact text span in the working memory. Prefer scoping by `section_title`
    and include light `pre_context` and/or `post_context` to uniquely anchor the target.
    By default the tool replaces exactly ONE occurrence (set `expected_replacements` when
    intentionally replacing multiple). Fails if the target is ambiguous or not found.
    Returns the updated memory and metadata.

    Arguments:
      - old_string (str, required): exact literal text to replace.
      - new_string (str, required): exact literal replacement.
      - explanation (str, required): one-sentence intent for telemetry.
      - section_title (str, optional): limit search to a single section; matches titles
        like '## n. <Title>' ignoring the numeric prefix (case-insensitive on the title text).
      - expected_replacements (int, optional; default=1): number of replacements intended.
      - pre_context (str, optional): unchanged text that must appear immediately before each target.
      - post_context (str, optional): unchanged text that must appear immediately after each target.
      - options (dict, optional): same as patch_memory:
          * strict_context (bool, default True): require unambiguous anchors; otherwise fail.
          * normalize_whitespace (bool, default False): collapse repeated spaces when matching (matching only).
          * case_sensitive (bool, default True): literal case-sensitive matching.

      - current_memory (str, injected by agent): the working-memory string to edit.

    Returns:
      {
        "new_memory": <updated string>,
        "meta": {
          "applied_hunks": 1,
          "changed_lines": <int>,
          "sections_touched": [<str>...],
          "warnings": [<str>...]
        }
      }

    Examples:

    # Replace a single bullet inside “Goals and Plans”
    {
      "section_title": "Goals and Plans",
      "old_string": "- Current overarching goal.",
      "new_string": "- Current overarching goal: ship v1 by Sept 15.",
      "expected_replacements": 1,
      "explanation": "Clarify primary goal"
    }

    # Multi-occurrence replacement in a section, with context
    {
      "section_title": "Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
      "pre_context": "[Investigation notes]",
      "old_string": "Hypothesis: caching is the bottleneck",
      "new_string": "Hypothesis (confirmed): caching is the bottleneck",
      "expected_replacements": 2,
      "explanation": "Mark hypothesis as confirmed"
    }
    """
    print(f"---TOOL: replace_in_memory---")
    memory = current_memory or ""
    opts = _Options.from_dict(options or {})

    # Select target text (whole memory or a specific section body)
    doc = _Document.parse(memory)
    target_label = None
    if section_title:
        section = doc.find_section_by_title(section_title)
        if section is None:
            raise PatchError(
                f"Section not found: '{section_title}'. "
                f"Available sections: {', '.join(doc.list_titles())}"
            )
        target_text = section.body
        target_label = section.title
    else:
        target_text = memory

    def _looks_like_header(s: str) -> bool:
        return s.strip().startswith("## ")

    def _strip_leading_header(s: str, header_line: str) -> str:
        # If new_string starts with the exact header line, drop it and leading newlines
        s_norm = s.replace("\r\n", "\n").replace("\r", "\n")
        header_norm = header_line.replace("\r\n", "\n").replace("\r", "\n")
        if s_norm.startswith(header_norm):
            s_norm = s_norm[len(header_norm):]
            # drop a single leading newline if it’s there
            if s_norm.startswith("\n"):
                s_norm = s_norm[1:]
        return s_norm

    def _contains_line(body: str, line: str) -> bool:
        # naive but sufficient: checks exact line presence in section body
        body_lines = [ln.rstrip("\n") for ln in body.splitlines()]
        return line.rstrip("\n") in body_lines

    # ---------------- Header-aware APPEND mode ----------------
    if section_title:
        header_line = section.header  # e.g., "## 2. Facts and Knowledge"
        header_line = header_line.replace("\r\n", "\n").replace("\r", "\n")

        wants_append = False
        if old_string.strip() == "":
            wants_append = True
        elif _looks_like_header(old_string):
            wants_append = True
        elif new_string.replace("\r\n", "\n").replace("\r", "\n").startswith(header_line):
            wants_append = True

        if wants_append:
            body = target_text  # section.body
            # Remove any header the model copied into new_string
            content_to_add = _strip_leading_header(new_string, header_line).strip("\n").strip()

            if not content_to_add:
                # Nothing to add (e.g., model only supplied the header) -> no-op
                return {
                    "new_memory": memory,
                    "meta": {
                        "applied_hunks": 0,
                        "changed_lines": 0,
                        "sections_touched": [],
                        "warnings": ["No-op: new_string contained no body content to append."],
                    },
                }

            # If it looks like a multi-line block, we append as-is; if it’s a single line, dedupe.
            if "\n" not in content_to_add:
                # Single line: avoid duplicate insertion
                if _contains_line(body, content_to_add):
                    return {
                        "new_memory": memory,
                        "meta": {
                            "applied_hunks": 0,
                            "changed_lines": 0,
                            "sections_touched": [],
                            "warnings": ["No-op: line already present in section."],
                        },
                    }

            # Compose new body with clean spacing: one blank line between existing body and new content
            if not body.strip():
                new_body = "\n" + content_to_add + "\n"
            else:
                new_body = body.rstrip() + "\n\n" + content_to_add + "\n"

            section.body = new_body
            new_memory = doc.render()
            return {
                "new_memory": new_memory,
                "meta": {
                    "applied_hunks": 1,
                    "changed_lines": content_to_add.count("\n") + 1,
                    "sections_touched": [target_label] if target_label else [],
                    "warnings": [],
                },
            }
    # --------------- end of header-aware APPEND mode ----------

    #  Allow insertion when old_string is empty
    if old_string.strip() == "":
        if not target_text.strip():
            # Section (or whole memory) empty — append
            new_body = "\n" + new_string.strip() + "\n"
        else:
            # Default: append at end
            new_body = target_text.rstrip() + "\n" + "\n" + new_string.strip() + "\n"

        if section_title:
            section.body = new_body
            new_memory = doc.render()
            sections_touched = [target_label] if target_label else []
        else:
            new_memory = new_body
            sections_touched = []

        return {
            "new_memory": new_memory,
            "meta": {
                "applied_hunks": 1,
                "changed_lines": new_string.count("\n") + 1,
                "sections_touched": sections_touched,
                "warnings": [],
            },
        }

    # Build the matching "needle"
    needle = _build_needle(pre_context, old_string, post_context)

    # Find matches according to options
    matches = _find_matches(
        target_text=target_text,
        needle=needle,
        case_sensitive=opts.case_sensitive,
        normalize_whitespace=bool(opts.normalize_whitespace),
    )

    # If expected_replacements is None, default to 1
    intended = 1 if expected_replacements in (None, 0) else int(expected_replacements)

    # Enforce match count vs expected
    if intended == 1:
        if len(matches) == 0:
            # Idempotency hint: maybe already replaced?
            if _already_replaced(target_text, pre_context, new_string, post_context, opts):
                new_memory = memory  # no change
                return _result(new_memory, warnings=["No-op: target already equals new_string at anchor."])
            raise PatchError(
                "Target not found. Provide section_title and/or pre/post context to disambiguate."
            )
        if len(matches) > 1:
            raise PatchError(
                f"Ambiguous target: found {len(matches)} possible matches. "
                "Add pre_context/post_context or set the correct expected_replacements."
            )
    else:
        if len(matches) != intended:
            # No-op check for the multi case (rare)
            if len(matches) == 0 and _already_replaced_multi(target_text, pre_context, new_string, post_context, opts, intended):
                new_memory = memory
                return _result(new_memory, warnings=[f"No-op: found {intended} new_string matches already present."])
            raise PatchError(
                f"expected_replacements={intended} but found {len(matches)} matches. "
                "Refine anchors or adjust expected_replacements."
            )

    # Compute replacement ranges for just the old_string within each match
    ranges = _extract_old_ranges_within_matches(
        target_text,
        matches,
        pre_context or "",
        old_string,
        post_context or "",
        opts,
    )

    # Idempotency: check if all already equal to new_string
    if _all_ranges_already_new(target_text, ranges, new_string):
        new_memory = memory
        return _result(new_memory, warnings=["No-op: all target ranges already equal new_string."])

    # Apply replacements left-to-right
    new_target_text = _apply_replacements(target_text, ranges, new_string)

    # Reassemble the full memory if we edited a section
    if section_title:
        section.body = new_target_text
        new_memory = doc.render()
        sections_touched = [target_label] if target_label else []
    else:
        new_memory = new_target_text
        sections_touched = []

    # Compute changed_lines (approximate, consistent with patch_memory telemetry spirit)
    count = len(ranges)
    changed_lines = (old_string.count("\n") + 1) * count + (new_string.count("\n") + 1) * count

    return {
        "new_memory": new_memory,
        "meta": {
            "applied_hunks": 1,  # one logical replace operation
            "changed_lines": changed_lines,
            "sections_touched": sections_touched,
            "warnings": [],
        },
    }


# -----------------
# Helper functions
# -----------------

def _build_needle(pre: Optional[str], old: str, post: Optional[str]) -> str:
    """Construct the matching needle with required adjacency: pre + old + post."""
    parts: List[str] = []
    if pre:
        parts.append(pre)
    parts.append(old)
    if post:
        parts.append(post)
    return "".join(parts)


def _find_matches(
    target_text: str,
    needle: str,
    *,
    case_sensitive: bool,
    normalize_whitespace: bool,
) -> List[Tuple[int, int]]:
    """
    Find all non-overlapping matches of `needle` in `target_text`.
    When normalize_whitespace=True, sequences of spaces/tabs in the NEEDLE are matched
    against one-or-more spaces/tabs in the target (matching only; indices refer to the
    original target_text span).
    Returns list of (start, end) spans covering the entire needle match.
    """
    if not needle:
        return []

    if not normalize_whitespace:
        hay = target_text if case_sensitive else target_text.lower()
        ned = needle if case_sensitive else needle.lower()
        matches: List[Tuple[int, int]] = []
        idx = 0
        while True:
            f = hay.find(ned, idx)
            if f == -1:
                break
            matches.append((f, f + len(ned)))
            idx = f + len(ned)
        return matches

    # Build a regex that collapses runs of spaces/tabs in the NEEDLE into [ \t]+
    pattern = _needle_to_regex(needle)
    flags = 0 if case_sensitive else re.IGNORECASE
    matches = []
    for m in re.finditer(pattern, target_text, flags):
        matches.append((m.start(), m.end()))
    return matches


def _needle_to_regex(needle: str) -> str:
    """Escape needle into a regex, collapsing consecutive spaces/tabs into [ \t]+ (not across newlines)."""
    out: List[str] = []
    prev_ws = False
    for ch in needle:
        if ch in (" ", "\t"):
            if not prev_ws:
                out.append(r"[ \t]+")
                prev_ws = True
            # else skip repeated spaces/tabs
        else:
            out.append(re.escape(ch))
            prev_ws = False
    return "".join(out)


def _already_replaced(
    target_text: str,
    pre_context: Optional[str],
    new_string: str,
    post_context: Optional[str],
    opts: _Options,
) -> bool:
    """Heuristic no-op check for the single-replacement case."""
    needle_new = _build_needle(pre_context, new_string, post_context)
    m = _find_matches(
        target_text,
        needle_new,
        case_sensitive=opts.case_sensitive,
        normalize_whitespace=bool(opts.normalize_whitespace),
    )
    return len(m) >= 1


def _already_replaced_multi(
    target_text: str,
    pre_context: Optional[str],
    new_string: str,
    post_context: Optional[str],
    opts: _Options,
    expected: int,
) -> bool:
    """Heuristic no-op check for multi-replacement case."""
    needle_new = _build_needle(pre_context, new_string, post_context)
    m = _find_matches(
        target_text,
        needle_new,
        case_sensitive=opts.case_sensitive,
        normalize_whitespace=bool(opts.normalize_whitespace),
    )
    return len(m) == expected


def _extract_old_ranges_within_matches(
    target_text: str,
    matches: List[Tuple[int, int]],
    pre_context: str,
    old_string: str,
    post_context: str,
    opts: _Options,
) -> List[Tuple[int, int]]:
    """
    From each full (pre+old+post) match, extract the (start,end) range for just the old_string.
    If no pre/post provided, the match already equals old_string and we return it as-is.
    If normalization allowed the match to succeed but the exact old_string literal isn't found
    in the window, fail (we don't attempt fuzzy rewrite of the center piece).
    """
    ranges: List[Tuple[int, int]] = []
    if not pre_context and not post_context:
        # Each match span equals the old_string span directly
        return matches

    for (start, end) in matches:
        window = target_text[start:end]
        # Find old_string inside the window with literal (case-insensitive if requested)
        if opts.case_sensitive:
            inner_idx = window.find(old_string)
        else:
            inner_idx = window.lower().find(old_string.lower())
        if inner_idx < 0:
            # Could happen when normalize_whitespace=True and window spacing differs.
            # We keep behavior strict: require old_string literal present inside the window.
            raise PatchError(
                "Anchor matched, but exact old_string not found inside the anchored window. "
                "Either disable normalize_whitespace, or include exact literal old_string."
            )
        old_start = start + inner_idx
        old_end = old_start + len(old_string)
        ranges.append((old_start, old_end))
    return ranges


def _all_ranges_already_new(target_text: str, ranges: List[Tuple[int, int]], new_string: str) -> bool:
    """True if every target range already equals new_string."""
    for (s, e) in ranges:
        if target_text[s:e] != new_string:
            return False
    return True


def _apply_replacements(text: str, ranges: List[Tuple[int, int]], new_string: str) -> str:
    """Apply replacements left-to-right using the provided spans."""
    if not ranges:
        return text
    # Ensure left-to-right and non-overlapping
    ranges = sorted(ranges, key=lambda p: p[0])
    out_parts: List[str] = []
    cursor = 0
    for (s, e) in ranges:
        out_parts.append(text[cursor:s])
        out_parts.append(new_string)
        cursor = e
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _result(new_memory: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Assemble the standard return envelope."""
    return {
        "new_memory": new_memory,
        "meta": {
            "applied_hunks": 1,
            "changed_lines": 0,
            "sections_touched": [],
            "warnings": warnings or [],
        },
    }




if __name__ == "__main__":
    

    def hr(title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)


    def run_test(
        name: str,
        memory: str,
        *,
        old_string: str,
        new_string: str,
        explanation: Optional[str] = None,
        section_title: Optional[str] = None,
        expected_replacements: Optional[int] = 1,
        pre_context: Optional[str] = None,
        post_context: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        hr(f"TEST: {name}")
        try:
            args = {
                "old_string": old_string,
                "new_string": new_string,
                "explanation": explanation or name,
                "section_title": section_title,
                "expected_replacements": expected_replacements,
                "pre_context": pre_context,
                "post_context": post_context,
                "options": options or {},
                "current_memory": memory,
            }
            result = replace_in_memory.invoke(args)
            print("RESULT new_memory:\n", result["new_memory"])
            print("RESULT meta:\n", result["meta"])
            return result
        except PatchError as e:
            print("REPLACE ERROR:", str(e))
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

## 3. Active Knowledge (Reasoning Outputs & Ephemeral Notes)

[Investigation notes]
- Hypothesis: caching is the bottleneck
- Evidence: slow Redis reads
"""

    # ----------------------------------------------------------------------------------
    # 1) Single replacement in a section (basic)
    # ----------------------------------------------------------------------------------
    r1 = run_test(
        "single replacement in Goals and Plans",
        initial_memory,
        section_title="Goals and Plans",
        old_string="- Current overarching goal.",
        new_string="- Current overarching goal: ship v1 by Sept 15.",
        expected_replacements=1,
    )
    mem_1 = r1["new_memory"] if r1 else initial_memory

    # ----------------------------------------------------------------------------------
    # 2) Idempotent no-op (apply same change again)
    # ----------------------------------------------------------------------------------
    r2 = run_test(
        "idempotent no-op on repeated replacement",
        mem_1,
        section_title="Goals and Plans",
        old_string="- Current overarching goal.",
        new_string="- Current overarching goal: ship v1 by Sept 15.",
        expected_replacements=1,
    )
    mem_2 = r2["new_memory"] if r2 else mem_1

    # ----------------------------------------------------------------------------------
    # 3) Ambiguous without expected_replacements (old_string occurs twice)
    #    We inject a second identical hypothesis line to force ambiguity.
    # ----------------------------------------------------------------------------------
    memory_with_dup = mem_2.replace(
        "- Hypothesis: caching is the bottleneck",
        "- Hypothesis: caching is the bottleneck\n- Hypothesis: caching is the bottleneck",
        1,
    )
    r3 = run_test(
        "ambiguous (two matches) without expected_replacements",
        memory_with_dup,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        old_string="- Hypothesis: caching is the bottleneck",
        new_string="- Hypothesis (confirmed): caching is the bottleneck",
        expected_replacements=1,  # default behavior
    )

    # ----------------------------------------------------------------------------------
    # 4) Multi-replacement with explicit expected_replacements=2 (success)
    # ----------------------------------------------------------------------------------
    r4 = run_test(
        "multi-replacement with expected_replacements=2",
        memory_with_dup,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        old_string="- Hypothesis: caching is the bottleneck",
        new_string="- Hypothesis (confirmed): caching is the bottleneck",
        expected_replacements=2,
    )
    mem_4 = r4["new_memory"] if r4 else memory_with_dup

    # ----------------------------------------------------------------------------------
    # 5) Case-insensitive replacement (options.case_sensitive=False)
    # ----------------------------------------------------------------------------------
    mem_case = mem_4.replace("slow Redis reads", "SLOW REDIS READS")
    r5 = run_test(
        "case-insensitive replace",
        mem_case,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        old_string="- Evidence: slow Redis reads",
        new_string="- Evidence: slow Redis reads (profiled)",
        expected_replacements=1,
        options={"case_sensitive": False},
    )
    mem_5 = r5["new_memory"] if r5 else mem_case

    # ----------------------------------------------------------------------------------
    # 6) Whitespace-normalized matching (options.normalize_whitespace=True)
    #    Create a target with irregular spacing and replace with normalized text.
    # ----------------------------------------------------------------------------------
    mem_ws = mem_5.replace(
        "- Evidence: slow Redis reads (profiled)",
        "-   Evidence:    slow    Redis   reads    (profiled)",
    )
    r6 = run_test(
        "normalize whitespace for matching",
        mem_ws,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        old_string="- Evidence: slow Redis reads (profiled)",
        new_string="- Evidence: slow Redis reads (profiled, p99=1.2s)",
        expected_replacements=1,
        options={"normalize_whitespace": True},
    )
    mem_6 = r6["new_memory"] if r6 else mem_ws

    # ----------------------------------------------------------------------------------
    # 7) Pre-context anchoring
    # ----------------------------------------------------------------------------------
    r7 = run_test(
        "pre-context anchoring inside section",
        mem_6,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        pre_context="[Investigation notes]\n",
        old_string="- Hypothesis (confirmed): caching is the bottleneck",
        new_string="- Hypothesis (confirmed): caching is the bottleneck [cached keys: 1.2M]",
        expected_replacements=1,
    )
    mem_7 = r7["new_memory"] if r7 else mem_6

    # ----------------------------------------------------------------------------------
    # 8) Post-context anchoring (ensure target followed by a known line)
    # ----------------------------------------------------------------------------------
    r8 = run_test(
        "post-context anchoring inside section",
        mem_7,
        section_title="Active Knowledge (Reasoning Outputs & Ephemeral Notes)",
        old_string="- Hypothesis (confirmed): caching is the bottleneck [cached keys: 1.2M]",
        post_context="\n- Evidence:",
        new_string="- Hypothesis (confirmed): caching is the bottleneck [cached keys: 1.3M]",
        expected_replacements=1,
    )
    mem_8 = r8["new_memory"] if r8 else mem_7

    # ----------------------------------------------------------------------------------
    # 9) Global replacement (no section_title) — remove a temporary marker
    # ----------------------------------------------------------------------------------
    mem_global = mem_8 + "\n[TEMP]"
    r9 = run_test(
        "global replacement (no section) to remove marker",
        mem_global,
        old_string="[TEMP]",
        new_string="",
        expected_replacements=1,
    )
    mem_9 = r9["new_memory"] if r9 else mem_global

    # ----------------------------------------------------------------------------------
    # 10) Section not found error
    # ----------------------------------------------------------------------------------
    r10 = run_test(
        "section not found diagnostic",
        mem_9,
        section_title="Nonexistent Section",
        old_string="- Foo",
        new_string="- Bar",
        expected_replacements=1,
    )

    # ----------------------------------------------------------------------------------
    # 12) Ambiguity resolved by expected_replacements (3 copies)
    # ----------------------------------------------------------------------------------
    mem_many = mem_9 + (
        "\n- Subgoals or milestones.\n- Subgoals or milestones.\n- Subgoals or milestones.\n"
    )
    r12a = run_test(
        "ambiguous three matches with default expected_replacements=1 (should fail)",
        mem_many,
        section_title="Goals and Plans",
        old_string="- Subgoals or milestones.",
        new_string="- Subgoals or milestones. (tracked)",
    )
    r12b = run_test(
        "resolve ambiguity with expected_replacements=4",
        mem_many,
        section_title="Goals and Plans",
        old_string="- Subgoals or milestones.",
        new_string="- Subgoals or milestones. (tracked)",
        expected_replacements=4,
    )

    # ----------------------------------------------------------------------------------
    # 13) No-op when new_string already present at anchor but old_string not found
    # ----------------------------------------------------------------------------------
    mem_noop = mem_9.replace(
        "- Subgoals or milestones.",
        "- Subgoals or milestones. (weekly)",
    )
    r13 = run_test(
        "no-op when already replaced (anchor present)",
        mem_noop,
        section_title="Goals and Plans",
        pre_context="## 1. Goals and Plans\n\n",
        old_string="- Subgoals or milestones.",
        post_context="\n- Planned steps",
        new_string="- Subgoals or milestones. (weekly)",
        expected_replacements=1,
    )

    # ----------------------------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------------------------
    hr("ALL TESTS COMPLETE")