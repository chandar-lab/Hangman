# memory_utils.py
# Utility functions and data models for working with memory strings.
# This module provides helper functions and data structures for working with
# memory strings, including parsing, normalization, and error handling.
# It is used by the patch_memory and replace_in_memory tools.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============
# Data models
# ============

@dataclass
class _Options:
    strict_context: bool = True
    normalize_whitespace: bool = False
    case_sensitive: bool = True

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "_Options":
        return _Options(
            strict_context=bool(d.get("strict_context", False)),
            normalize_whitespace=bool(d.get("normalize_whitespace", True)),
            case_sensitive=bool(d.get("case_sensitive", True)),
        )


@dataclass
class _Section:
    header: str         # full header line as in memory, e.g., "## 1. Goals and Plans"
    title: str          # normalized title text, e.g., "Goals and Plans"
    body: str           # text from after header to before next header / EOF


@dataclass
class _Document:
    preface: str                # any text before the first '## n. Title'
    sections: List[_Section]

    SECTION_RE = re.compile(r"^##\s+\d+\.\s+(.*)$", re.MULTILINE)

    @staticmethod
    def parse(memory: str) -> "_Document":
        """Split memory into preface + ordered sections."""
        sections: List[_Section] = []
        matches = list(_Document.SECTION_RE.finditer(memory))
        if not matches:
            # Treat entire memory as preface (no sections)
            return _Document(preface=memory, sections=[])

        preface_start = 0
        preface_end = matches[0].start()
        preface = memory[preface_start:preface_end]

        for i, m in enumerate(matches):
            header_start = m.start()
            header_end = m.end()
            header_line = memory[header_start:header_end]
            title_text = m.group(1).strip()

            body_start = header_end
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(memory)
            body = memory[body_start:body_end]

            sections.append(_Section(header=header_line, title=_norm_title(title_text), body=body))

        return _Document(preface=preface, sections=sections)

    def find_section_by_title(self, query_title: str) -> Optional[_Section]:
        qt = _norm_title(query_title)
        for s in self.sections:
            if _norm_title(s.title) == qt:
                return s
        return None

    def list_titles(self) -> List[str]:
        return [s.title for s in self.sections]

    def render(self) -> str:
        # Reassemble in original order
        parts: List[str] = [self.preface] if self.preface else []
        for s in self.sections:
            parts.append(s.header)
            parts.append(s.body)
        return "".join(parts)


@dataclass
class _Hunk:
    section_title: str
    pre_context: List[str]
    minus: List[str]
    plus: List[str]
    post_context: List[str]


@dataclass
class _Patch:
    hunks: List[_Hunk]

    @staticmethod
    def parse(patch: str) -> "_Patch":
        """Parse a minimal V4A-like, section-anchored patch format."""
        # Normalize newlines
        text = patch.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Allow accidental leading spaces before the '*** Begin Patch' header
        text = text.lstrip()

        # Basic header checks
        if not text.startswith("*** Begin Patch"):
            raise PatchError("Patch must start with '*** Begin Patch'.")
        if not text.endswith("*** End Patch"):
            raise PatchError("Patch must end with '*** End Patch'.")
        if "*** Update Memory" not in text:
            raise PatchError("Missing '*** Update Memory' header.")

        # Split into body lines between the two headers
        begin_idx = text.index("*** Begin Patch") + len("*** Begin Patch")
        end_idx = text.rindex("*** End Patch")
        body = text[begin_idx:end_idx].strip("\n")

        # Tokenize hunks by '@@ section:'
        hunk_blocks = re.split(r"(?m)^(?=@@\s*section:\s*)", body)
        hunks: List[_Hunk] = []

        for block in hunk_blocks:
            block = block.strip()
            if not block:
                continue
            m = re.match(r"^@@\s*section:\s*(.+?)\s*$", block.splitlines()[0])
            if not m:
                # allow *** Update Memory line to precede hunks; skip if this is that line
                if block.startswith("*** Update Memory"):
                    continue
                raise PatchError("Each hunk must start with '@@ section: <Title>'.")
            section_title = m.group(1).strip()

            # Remaining lines are the hunk body
            lines = block.splitlines()[1:]

            # Classify lines into pre_context (before first +/-), minus/plus, post_context (after last +/-)
            first_edit = None
            last_edit = None
            for idx, line in enumerate(lines):
                if line.startswith("-") or line.startswith("+"):
                    first_edit = idx
                    break
            for idx in range(len(lines) - 1, -1, -1):
                if lines[idx].startswith("-") or lines[idx].startswith("+"):
                    last_edit = idx
                    break

            pre_context: List[str] = []
            post_context: List[str] = []
            minus: List[str] = []
            plus: List[str] = []

            if first_edit is None:
                # No +/- found: treat entire block as pure pre_context (for insert-only use case)
                pre_context = [ln for ln in lines if ln.strip() != ""]
            else:
                pre_context = [ln for ln in lines[:first_edit] if ln.strip() != ""]
                post_context = [ln for ln in lines[last_edit + 1:] if ln.strip() != ""]
                for ln in lines[first_edit:last_edit + 1]:
                    if ln.startswith("-"):
                        minus.append(ln[1:])
                    elif ln.startswith("+"):
                        plus.append(ln[1:])
                    elif ln.strip() == "":
                        # ignore blank lines inside the edit block
                        continue
                    else:
                        # context in the middle of +/- window isn't supported; advise simpler shape
                        raise PatchError(
                            "Unsupported context inside edit block. Put context lines "
                            "before the first or after the last +/- line."
                        )

            hunks.append(_Hunk(
                section_title=section_title,
                pre_context=pre_context,
                minus=minus,
                plus=plus,
                post_context=post_context,
            ))

        if not hunks:
            raise PatchError("No hunks parsed. Ensure you included '@@ section:' blocks.")
        return _Patch(hunks=hunks)


# =============
# Util helpers
# =============

class PatchError(RuntimeError):
    """Raised when the patch cannot be applied safely/uniquely."""


def _norm_title(title: str) -> str:
    # Normalize just for matching titles (strip numbering if present elsewhere)
    return re.sub(r"\s+", " ", title.strip()).lower()


def _approx_index(haystack: str, needle: str, *, case_sensitive: bool = True) -> int:
    """
    Very light fallback to find a literal needle when normalization made us lose offsets.
    This tries a naive search ignoring consecutive spaces differences.
    """
    if not case_sensitive:
        haystack = haystack.lower()
        needle = needle.lower()
    # Collapse multiple spaces for both
    h = re.sub(r"[ \t]+", " ", haystack)
    n = re.sub(r"[ \t]+", " ", needle)
    idx = h.find(n)
    if idx < 0:
        return -1
    # This is approximate; return best-effort original index
    # (good enough for rare normalize_whitespace=True paths)
    return max(0, min(len(haystack) - 1, idx))