"""Progressive-disclosure helpers for ``IndexHandle``.

Every provider that implements the 4-step read chain (``list_categories``
→ ``list_notes`` → ``read_outline`` → ``read_section``) needs the same
markdown summary / outline / section logic. Centralising here keeps the
file / SQL / ephemeral implementations bit-for-bit identical so hosts
can swap backends without seeing layout drift.

Public surface:

- ``make_summary`` — assemble a ``NoteSummary`` from raw note fields.
- ``parse_outline`` — produce a ``NoteOutline`` (heading tree) from
  the body of a single note.
- ``extract_section`` — return the body slice associated with a
  specific heading (case-insensitive, exact-text match).
"""

from __future__ import annotations

import re
from typing import List, Optional

from geny_executor.memory.provider import (
    NoteOutline,
    NoteSummary,
    OutlineNode,
)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", flags=re.DOTALL)
_FIRST_PARA_LIMIT = 200


def _strip_frontmatter(body: str) -> str:
    """Drop a leading ``---`` YAML frontmatter block, if present.

    Provider note bodies on disk include frontmatter; the
    progressive-disclosure surface returns prose only, so summary
    previews and outline parsing skip past it.
    """
    if body.startswith("---"):
        match = _FRONTMATTER_RE.match(body)
        if match is not None:
            return body[match.end() :]
    return body


def _first_paragraph(body: str, *, limit: int = _FIRST_PARA_LIMIT) -> str:
    """Return the first non-empty, non-heading paragraph trimmed to
    ``limit`` characters. Used as the listing preview.
    """
    text = _strip_frontmatter(body)
    for para in text.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        # Skip pure heading paragraphs.
        non_heading = [line for line in stripped.splitlines() if not line.lstrip().startswith("#")]
        cleaned = "\n".join(line.rstrip() for line in non_heading).strip()
        if not cleaned:
            continue
        return cleaned[:limit]
    return ""


def make_summary(
    *,
    filename: str,
    title: str,
    category: str,
    tags: List[str],
    importance: str,
    body: str,
    modified: str,
) -> NoteSummary:
    """Assemble a ``NoteSummary`` from raw note fields."""
    char_count = len(body or "")
    return NoteSummary(
        filename=filename,
        title=title or filename,
        category=category or "",
        tags=list(tags or []),
        importance=str(importance or "medium").lower(),
        char_count=char_count,
        modified=modified or "",
        first_paragraph=_first_paragraph(body or ""),
    )


def parse_outline(
    *,
    filename: str,
    title: str,
    body: str,
) -> NoteOutline:
    """Build the heading tree of ``body`` as a ``NoteOutline``.

    Frontmatter is skipped before parsing. ``line_start`` and
    ``line_end`` reference 1-indexed lines in the **post-frontmatter**
    body — that's what ``extract_section`` slices against, so callers
    that round-trip through both helpers see consistent line ranges.

    The outline is flat (every heading appended in source order) plus
    nested children: each ``OutlineNode``'s ``children`` list contains
    every heading at strictly greater depth that appears between this
    heading and the next heading at the same or shallower depth.
    """
    text = _strip_frontmatter(body or "")
    lines = text.splitlines()

    # First pass — flat list of (level, heading, line_start) for
    # every line that matches the heading regex (top-level only —
    # we don't recurse into code-fenced ``` blocks for now since
    # the markdown surface doesn't currently emit them inside notes).
    flat: List[OutlineNode] = []
    in_code_fence = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        if not heading_text:
            continue
        flat.append(
            OutlineNode(
                level=level,
                heading=heading_text,
                line_start=idx + 1,  # 1-indexed
                line_end=len(lines),  # patched below
            )
        )

    # Second pass — patch line_end of each node so the section runs
    # up to (but not including) the next heading at the same or
    # shallower depth. Last heading runs to the end of the body.
    for i, node in enumerate(flat):
        end = len(lines)
        for follower in flat[i + 1 :]:
            if follower.level <= node.level:
                end = follower.line_start - 1
                break
        node.line_end = end

    # Third pass — fold flat list into a tree. Use a stack of
    # (last_node, depth_floor); each new node is the child of the
    # nearest ancestor with strictly smaller level.
    root: List[OutlineNode] = []
    stack: List[OutlineNode] = []
    for node in flat:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if not stack:
            root.append(node)
        else:
            stack[-1].children.append(node)
        stack.append(node)

    return NoteOutline(filename=filename, title=title or filename, headings=root)


def extract_section(body: str, heading: str) -> Optional[str]:
    """Return the body slice for the section whose heading text
    equals ``heading`` (case-insensitive). Returns ``None`` when the
    heading isn't found.

    The returned text excludes the heading line itself — only the
    section body. Trailing whitespace is stripped.
    """
    if not heading or not body:
        return None
    text = _strip_frontmatter(body)
    lines = text.splitlines()
    needle = heading.strip().lower()

    in_code_fence = False
    section_start: Optional[int] = None
    section_level: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            if section_start is None:
                continue
            continue
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        if section_start is None:
            if heading_text.lower() == needle:
                section_start = idx + 1
                section_level = level
            continue
        # Already inside a section — terminate at sibling/parent heading.
        if section_level is not None and level <= section_level:
            return "\n".join(lines[section_start:idx]).rstrip("\n").rstrip()

    if section_start is None:
        return None
    return "\n".join(lines[section_start:]).rstrip("\n").rstrip()


__all__ = [
    "make_summary",
    "parse_outline",
    "extract_section",
]
