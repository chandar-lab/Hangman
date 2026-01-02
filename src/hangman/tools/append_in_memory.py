# append_in_memory.py
# Append lines to a specific section of the working memory (paragraph-style, no numbering).

from __future__ import annotations

from typing import List, Optional
from langchain_core.tools import tool

# Reuse the section parser to stay consistent with other tools (e.g., patch_memory/replace_in_memory).
from hangman.tools.memory_utils import _Document, PatchError


@tool
def append_in_memory(
    section_title: str,
    lines: List[str],
    current_memory: Optional[str] = None,
) -> str:
    """
    Appends one or more lines to the end of a given section.

    Args:
        section_title: The section to append to (e.g., "Goals and Plans", "Facts and Knowledge", "Active Notes").
        lines: A list of strings to append as individual lines.

    Returns:
        The updated working memory string.
    """
    print(f"---TOOL: append_in_memory---")
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

    # Normalize incoming lines (strip only trailing newlines; keep internal spaces as-is).
    new_block = "\n".join(line.rstrip("\n") for line in lines if line is not None and line != "")

    if not new_block:
        # Nothing to do
        return doc.render()

    body = section.body
    if not body.strip():
        # Empty section → just place lines with leading/trailing NL
        section.body = ("\n" + new_block + "\n")
    else:
        # Non-empty → ensure a blank line separation, then append, then newline
        section.body = body.rstrip() + "\n\n" + new_block + "\n"

    return doc.render()