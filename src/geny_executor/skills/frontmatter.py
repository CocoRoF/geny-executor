"""YAML frontmatter parser for SKILL.md files.

Cycle 20260424 executor uplift — Phase 4 Week 7.

A SKILL.md file looks like::

    ---
    name: refactor-ts
    description: Plan and execute a TypeScript refactor
    allowed_tools:
      - Read
      - Grep
      - Edit
    model_override: claude-opus-4-7
    execution_mode: inline
    ---

    # body starts here

    When the user asks to refactor TypeScript...

``parse_frontmatter`` splits the file into ``(metadata_dict, body)``.
The frontmatter MUST:

* Start on the first non-empty line with a bare ``---`` delimiter.
* Contain valid YAML.
* End with another ``---`` on its own line.

If any of those fails the whole file is treated as skill body with an
empty metadata dict — the loader will reject it because ``name`` /
``description`` are missing, which surfaces the error at parse time
instead of letting a malformed skill masquerade as a valid one.

Safety: we use ``yaml.safe_load`` — SKILL.md files come from the
project tree, but they may come from third-party plugin bundles we
don't want to let execute arbitrary Python objects.

See ``executor_uplift/08_design_skills.md`` §3.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import yaml

_DELIMITER = "---"


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split ``text`` into ``(metadata, body)``.

    Returns ``({}, text)`` unchanged if no frontmatter block is
    present or if parsing fails — the caller decides whether to treat
    that as a hard error.

    When the frontmatter block is present but malformed (invalid YAML
    or missing closing ``---``), this still returns ``({}, original
    text)`` and logs at DEBUG — we prefer "skill rejected by the
    loader for missing fields" over "skill silently loaded with
    partial metadata".
    """
    if not text:
        return {}, ""

    # Find the opening delimiter (must be the first non-empty line).
    lines = text.splitlines(keepends=True)
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != _DELIMITER:
        return {}, text

    open_idx = idx
    close_idx = -1
    for j in range(open_idx + 1, len(lines)):
        if lines[j].strip() == _DELIMITER:
            close_idx = j
            break
    if close_idx == -1:
        # Opening delimiter without a matching closer.
        return {}, text

    frontmatter_text = "".join(lines[open_idx + 1 : close_idx])
    body = "".join(lines[close_idx + 1 :])
    # Strip the leading blank line that usually follows the frontmatter
    # block so the rendered prompt starts clean.
    if body.startswith("\n"):
        body = body[1:]

    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return {}, text

    if parsed is None:
        # Empty frontmatter block — valid YAML, no fields.
        return {}, body
    if not isinstance(parsed, dict):
        # Top-level YAML value is a list / string / etc. — not a
        # valid frontmatter block.
        return {}, text

    return parsed, body
