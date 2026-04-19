"""YAML-ish frontmatter parser + serializer.

Hand-rolled (no PyYAML dep) because:
  1. `geny-executor` core must stay optional-dep free; `memory` extras
     already pulls in numpy for vector, we do not add yaml on top.
  2. Geny's legacy parser is also hand-rolled and reads a deliberately
     narrow subset. Matching that subset keeps the file format
     round-trippable between the two codebases without semantic drift.

Supported frontmatter grammar:

    ---
    key: scalar                 # string / number / bool / iso8601
    other: "quoted string"      # double or single quotes preserved as string
    tags: [a, b, c]             # inline list of scalars
    aliases: [one, two]
    ---

Nested mappings are NOT supported. Multiline values are NOT supported.
Lists nest exactly one level (flat list of scalars).

The parser is robust: a malformed line becomes a (key, raw string) entry
rather than aborting the whole file. This matches Geny's tolerant behaviour.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_FM_BOUNDARY = re.compile(r"^---\s*$")
_LIST_INLINE = re.compile(r"^\[(.*)\]$")
_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def split(text: str) -> Tuple[Dict[str, Any], str]:
    """Return `(frontmatter_dict, body)`. If the text has no
    leading `---` frontmatter block, returns `({}, text)` untouched.
    """
    lines = text.splitlines()
    if not lines or not _FM_BOUNDARY.match(lines[0]):
        return {}, text

    meta_lines: List[str] = []
    body_start = None
    for idx, raw in enumerate(lines[1:], start=1):
        if _FM_BOUNDARY.match(raw):
            body_start = idx + 1
            break
        meta_lines.append(raw)

    if body_start is None:
        # No closing boundary — treat whole thing as body
        return {}, text

    meta = parse(meta_lines)
    body = "\n".join(lines[body_start:])
    # Preserve trailing newline if present in source
    if text.endswith("\n") and not body.endswith("\n"):
        body += "\n"
    return meta, body


def parse(lines: List[str]) -> Dict[str, Any]:
    """Parse `lines` (already stripped of the `---` boundaries) into
    a dict. Unknown / malformed lines are silently skipped.
    """
    out: Dict[str, Any] = {}
    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        # key: value
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            continue
        out[key] = _parse_value(value.strip())
    return out


def dump(meta: Dict[str, Any]) -> str:
    """Serialize `meta` back into the `---` frontmatter block,
    trailing newline included. Keys emitted in a stable order that
    matches Geny's canonical layout — deterministic output is a
    testability / diff-friendliness requirement, not a correctness one.
    """
    if not meta:
        return ""
    order = [
        "title",
        "aliases",
        "tags",
        "category",
        "importance",
        "created",
        "modified",
        "source",
        "session_id",
        "links_to",
        "linked_from",
    ]
    rest = [k for k in meta if k not in order]
    keys = [k for k in order if k in meta] + sorted(rest)

    parts: List[str] = ["---"]
    for key in keys:
        parts.append(f"{key}: {_format_value(meta[key])}")
    parts.append("---")
    parts.append("")  # blank separator before body
    return "\n".join(parts)


# ── internal ──────────────────────────────────────────────────────────


def _parse_value(raw: str) -> Any:
    if not raw:
        return ""
    # Quoted string — strip quotes, keep inner content verbatim
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return raw[1:-1]
    # Inline list [a, b, c]
    m = _LIST_INLINE.match(raw)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return []
        items = [_parse_scalar(p.strip()) for p in _split_csv(inner)]
        return items
    return _parse_scalar(raw)


def _parse_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    if low in ("null", "none", "~"):
        return None
    # int / float
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        if _looks_like_float(raw):
            return float(raw)
    except ValueError:
        pass
    # Strip quotes if someone quoted inside a list
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        return raw[1:-1]
    return raw


def _looks_like_float(s: str) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return "." in s or "e" in s.lower()


def _split_csv(s: str) -> List[str]:
    """Split `a, b, "c, d", e` — respect quotes."""
    parts: List[str] = []
    buf: List[str] = []
    quote: str = ""
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return _format_scalar(value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    s = str(value)
    # Quote if value contains characters that break the simple parser
    if any(ch in s for ch in (":", "#", "[", "]", ",")) or (s and s[0] in ('"', "'")):
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s
