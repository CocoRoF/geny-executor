"""One short-term memory row, as text a model can read.

Providers render their short-term layer as text for hosts whose retriever
does not read memory itself. They used to write ``f"[{role}] {content!r}"``
— for any row holding content blocks that is the Python ``repr`` of a list
of dicts: ``[assistant] [{'type': 'tool_use', 'id': 'toolu_…', 'input':
{…}, 'cache_control': …}]``. The model received it verbatim, next to the
same turns replayed as real messages.
"""

from __future__ import annotations

import json
from typing import Any

#: A tool result is a reminder that the call happened and roughly what came
#: back, not the result itself — the full thing is in the replay window.
RESULT_CHARS = 300
ARGS_CHARS = 160


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _args(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return _clip(json.dumps(value, ensure_ascii=False), ARGS_CHARS)
        except (TypeError, ValueError):
            return _clip(str(value), ARGS_CHARS)
    return _clip(str(value or ""), ARGS_CHARS)


def _result(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        content = (
            " ".join(
                str(b.get("text") or "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            or "[non-text result]"
        )
    return _clip(str(content or ""), RESULT_CHARS)


def turn_text(role: str, content: Any) -> str:
    """``[role] text`` — content blocks rendered as prose, never ``repr``."""
    if isinstance(content, str):
        return f"[{role}] {content}"
    parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = str(block.get("text") or "").strip()
            if text:
                parts.append(text)
        elif kind == "tool_use":
            parts.append(f"[called {block.get('name') or '?'}({_args(block.get('input'))})]")
        elif kind == "tool_result":
            status = "failed" if block.get("is_error") else "returned"
            parts.append(f"[tool {status}: {_result(block)}]")
        elif kind in ("image", "image_url"):
            parts.append("[image]")
        elif kind == "document":
            parts.append("[document]")
    return f"[{role}] {' '.join(parts)}" if parts else f"[{role}]"


def turn_to_text(turn: Any) -> str:
    """:func:`turn_text` for a ``Turn`` row."""
    return turn_text(str(getattr(turn, "role", "") or "user"), getattr(turn, "content", ""))
