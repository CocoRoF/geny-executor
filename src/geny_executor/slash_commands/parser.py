"""Slash command parser — detect & split user input."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import List, Optional


_COMMAND_NAME_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


@dataclass
class ParsedSlash:
    """Result of parsing a user input line.

    ``command`` is the command name without the leading slash.
    ``args`` are positional tokens after the command (shlex-split).
    ``remaining_prompt`` is anything after the first newline — the
    host typically sends this to the LLM as user prompt after the
    command's response.
    """

    command: str
    args: List[str]
    remaining_prompt: str


def parse_slash(input_text: str) -> Optional[ParsedSlash]:
    """Detect a slash prefix and split.

    Returns ``None`` when the input doesn't start with ``/`` or when
    the would-be command name doesn't match
    ``[a-zA-Z][a-zA-Z0-9_-]*``. Whitespace at the start is tolerated.

    Examples::

        parse_slash("/cost")
            → ParsedSlash("cost", [], "")
        parse_slash("/skill-foo arg1\\nplease run")
            → ParsedSlash("skill-foo", ["arg1"], "please run")
        parse_slash("regular text")
            → None
        parse_slash("/cost 'with quoted'")
            → ParsedSlash("cost", ["with quoted"], "")
    """
    if not input_text:
        return None
    text = input_text.lstrip()
    if not text.startswith("/"):
        return None

    first_line, _, rest = text.partition("\n")

    # shlex.split tolerates quoted args; on bad shell syntax (unmatched
    # quote etc.) we treat the input as not-a-slash so the host falls
    # back to treating it as literal user input.
    try:
        parts = shlex.split(first_line[1:])
    except ValueError:
        return None
    if not parts:
        return None

    cmd = parts[0]
    if not _COMMAND_NAME_RE.fullmatch(cmd):
        return None
    return ParsedSlash(
        command=cmd,
        args=parts[1:],
        remaining_prompt=rest.lstrip(),
    )


__all__ = ["ParsedSlash", "parse_slash"]
