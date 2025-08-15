# delete_from_memory.py
# Delete lines from a specific section of the working memory using robust matching.

from __future__ import annotations

import re
from typing import List, Optional
from langchain_core.tools import tool

from hangman.tools.memory_utils import _Document, PatchError


def _canon_line(s: str) -> str:
    """
    Canonicalize a line for robust matching:
    - strip leading/trailing whitespace
    - drop common bullet prefixes (-, *, •) + one following space if present
    - collapse internal runs of whitespace to a single space
    - lowercase
    """
    s = s.strip()
    s = re.sub(r"^[\-\*\u2022]\s+", "", s)  # -, *, • + space
    s = re.sub(r"[ \t]+", " ", s)
    return s.lower()


@tool
def delete_from_memory(
    section_title: str,
    lines: List[str],
    current_memory: Optional[str] = None,
) -> str:
    """
    Deletes lines from the given section using robust, plain-text matching.

    Args:
        section_title: The section to edit (e.g., "Goals and Plans", "Facts and Knowledge", "Active Notes").
        lines: A list of target lines to remove.

    Returns:
        The updated working memory string.
    """
    print(f"---TOOL: delete_from_memory---")
    print(f"Section Title: {section_title}")
    print(f"Lines: {lines}")

    memory = current_memory or ""
    doc = _Document.parse(memory)

    section = doc.find_section_by_title(section_title)
    if section is None:
        raise PatchError(
            f"Section not found: '{section_title}'. "
            f"Available sections: {', '.join(doc.list_titles())}"
        )

    # Build canonical targets, drop empties
    targets_raw = [t for t in (lines or []) if t is not None and t.strip() != ""]
    if not targets_raw:
        return doc.render()

    targets_canon = [(_canon_line(t), len(_canon_line(t))) for t in targets_raw]

    # Walk each line in the section and keep those that do NOT match any target.
    kept: List[str] = []
    for line in section.body.splitlines():
        cl = _canon_line(line)
        remove = False
        for tc, ln in targets_canon:
            if not tc:
                continue
            if ln >= 8:
                # Long target → allow substring to tolerate small differences
                if tc in cl:
                    remove = True
                    break
            else:
                # Short target → require exact equality to avoid accidental broad matches
                if tc == cl:
                    remove = True
                    break
        if not remove:
            kept.append(line)

    # Reassemble body: preserve original newline style; ensure ending newline if section had any content
    if kept:
        new_body = "\n".join(kept)
        if not new_body.endswith("\n"):
            new_body += "\n"
        section.body = new_body
    else:
        # Section becomes empty → keep a single trailing newline so future appends are clean
        section.body = "\n"

    return doc.render()