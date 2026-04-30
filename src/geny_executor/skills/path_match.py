"""Gitignore-style path pattern matching for skill `paths` activation.

Phase 10.2 (Skills uplift) — skills declare `paths: ["src/**/*.ts"]` to
activate only when one of the named patterns matches a path the
session is currently working with. Stdlib only, no third-party
dependency.

Supported syntax:

* ``*``  — within a single path segment (does *not* cross ``/``).
* ``?``  — single character within a segment.
* ``**`` — across path segments (zero or more, including the slashes).
* leading ``/`` — anchors at path root (else floating).
* trailing ``/`` — directory-only match (matches the dir and anything under it).

Patterns are matched against forward-slash-normalised paths. Windows
backslashes are normalised before matching so authors don't write
two patterns. Empty / missing patterns return ``True`` for every path
(no narrowing).

Not supported (deliberate scope cuts vs full gitignore):
* Negation (``!pattern``) — skills can't *un*-activate themselves.
* Character classes (``[...]``) — uncommon in this domain; use ``?``.

If you need full gitignore behaviour, switch to ``pathspec`` later —
the public API (``match_any``, ``compile_patterns``) is deliberately
tiny so the swap is a one-liner.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence


def _normalise(path: str) -> str:
    """Forward-slash + strip leading ``./``. Lower-case is *not*
    applied — most ecosystems are case-sensitive."""
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _translate_segment(seg: str) -> str:
    """Translate a single path segment into a regex fragment.

    ``*`` and ``?`` are single-segment wildcards (do not match ``/``).
    Literal characters are regex-escaped.
    """
    out: List[str] = []
    for ch in seg:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _pattern_to_regex(pattern: str) -> "re.Pattern[str]":
    """Translate a single gitignore-ish pattern into a compiled regex.

    Algorithm: split on ``/``; translate each segment; ``**`` becomes
    a multi-segment wildcard; stitch back with explicit ``/``
    separators. Leading ``/`` anchors at path root, trailing ``/``
    means dir-only.
    """
    p = pattern.strip()
    anchored = p.startswith("/")
    if anchored:
        p = p[1:]
    dir_only = p.endswith("/")
    if dir_only:
        p = p[:-1]

    segments = p.split("/")
    parts: List[str] = []
    for seg in segments:
        if seg == "**":
            parts.append(".*")
        else:
            parts.append(_translate_segment(seg))

    body = "/".join(parts)
    # Collapse ``.*/`` / ``/.*`` into a single piece that allows zero
    # or more intervening slashes — keeps adjacent ``a/**/b`` matching
    # both ``a/b`` and ``a/x/b`` correctly.
    body = re.sub(r"\.\*/", "(?:.*/)?", body)
    body = re.sub(r"/\.\*", "(?:/.*)?", body)
    # Bare ``**`` pattern (whole path) → match anything.
    if body == "(?:.*/)?" or body == "(?:/.*)?":
        body = ".*"

    if anchored:
        prefix = r"\A"
    else:
        # Floating: match anywhere on a path-segment boundary.
        prefix = r"(?:\A|.*/)"

    if dir_only:
        # Match the directory itself (``docs``) or anything under it
        # (``docs/...``).
        suffix = r"(?:/.*)?\Z"
    else:
        suffix = r"\Z"

    return re.compile(prefix + body + suffix)


def compile_patterns(patterns: Sequence[str]) -> List["re.Pattern[str]"]:
    """Pre-compile a list of patterns. Skills compile once at load
    time and reuse the regexes for every active-paths check."""
    return [_pattern_to_regex(p) for p in patterns if p and p.strip()]


def match_any(paths: Iterable[str], compiled: Sequence["re.Pattern[str]"]) -> bool:
    """Return True if any of ``paths`` matches any of ``compiled``.

    Empty ``compiled`` returns False — caller decides what to do with
    that (skills with empty `paths` are usually treated as
    unconditional, so they short-circuit before reaching here).
    """
    if not compiled:
        return False
    for raw in paths:
        if not raw:
            continue
        path = _normalise(raw)
        for regex in compiled:
            if regex.match(path):
                return True
    return False
