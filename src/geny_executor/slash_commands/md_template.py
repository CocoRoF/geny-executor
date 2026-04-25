"""Markdown-template slash commands (PR-A.2.4).

Hosts let end users (or themselves) drop ``foo.md`` files into a
discovery directory (e.g. ``~/.geny/commands/``) and have those
become slash commands automatically. Each file's frontmatter
declares metadata; the body is treated as a prompt template — when
the user runs ``/foo arg1 arg2``, the body is substituted (``$ARG_1``
``$ARG_2`` ``$ARGS``) and emitted as a ``follow_up_prompt`` so the
host sends it to the LLM.

File format::

    ---
    description: Run the test suite for a single module
    category: control
    aliases: [t]
    ---
    Please run the tests for the module at $ARG_1 and report any
    failures. Use the Bash tool. Then summarize.

The handler:

1. Reads ``args`` (already shlex-split by :func:`parse_slash`).
2. Substitutes ``$ARG_N`` (1-indexed) and ``$ARGS`` (joined string).
3. Returns a :class:`SlashResult` whose ``content`` is a short
   notice ("running command from foo.md") and ``follow_up_prompt``
   is the substituted body — the host then sends that as the next
   user turn to the LLM.

Safety: the body is **not executed** — it's prompt text that goes to
the LLM. So a malicious .md file can attempt prompt-injection but
cannot run code on the host. Loaders skip files larger than
:data:`_MAX_BODY_BYTES` (64 KiB) to bound memory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)

logger = logging.getLogger(__name__)


_MAX_BODY_BYTES = 64 * 1024
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<front>.*?)---\s*\n(?P<body>.*)\Z", re.DOTALL)
_ARG_PLACEHOLDER_RE = re.compile(r"\$ARG_(\d+)")


@dataclass
class _Frontmatter:
    description: str = ""
    category: SlashCategory = SlashCategory.DOMAIN
    aliases: List[str] = None  # type: ignore[assignment]


def _parse_frontmatter(raw: str) -> _Frontmatter:
    """Tolerant frontmatter parser. Accepts simple ``key: value`` lines.
    Lists in flow form (``[a, b]``) handled for ``aliases``. Unknown
    keys ignored."""
    fm = _Frontmatter(aliases=[])
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "description":
            fm.description = value
        elif key == "category":
            try:
                fm.category = SlashCategory(value.lower())
            except ValueError:
                fm.category = SlashCategory.DOMAIN
        elif key == "aliases":
            fm.aliases = _parse_aliases(value)
    return fm


def _parse_aliases(value: str) -> List[str]:
    if not value:
        return []
    if value.startswith("["):
        # Flow list — strip braces, split on commas.
        inner = value.strip("[]")
        return [tok.strip().strip("'\"") for tok in inner.split(",") if tok.strip()]
    # Single token.
    return [value.strip("'\"")]


class MdTemplateCommand(SlashCommand):
    """Slash command synthesised from a markdown template file."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        category: SlashCategory,
        body_template: str,
        source_path: Path,
        aliases: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description or f"Template command from {source_path.name}"
        self.category = category
        self.aliases = list(aliases or [])
        self._template = body_template
        self._source = source_path

    async def execute(self, args: List[str], ctx: SlashContext) -> SlashResult:
        body = _ARG_PLACEHOLDER_RE.sub(
            lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) <= len(args) else m.group(0),
            self._template,
        )
        body = body.replace("$ARGS", " ".join(args))
        return SlashResult(
            content=f"_Running template command from `{self._source.name}`_",
            follow_up_prompt=body,
            metadata={"source": str(self._source)},
        )


def load_md_command(path: Path) -> Optional[MdTemplateCommand]:
    """Load one markdown file into a :class:`MdTemplateCommand`. Returns
    ``None`` for files that don't parse / exceed the size cap / lack a
    frontmatter block."""
    try:
        if path.stat().st_size > _MAX_BODY_BYTES:
            logger.warning("md_command_too_large", extra={"path": str(path)})
            return None
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("md_command_read_failed", extra={"path": str(path), "error": str(exc)})
        return None
    match = _FRONTMATTER_RE.match(text)
    if not match:
        logger.warning("md_command_missing_frontmatter", extra={"path": str(path)})
        return None
    fm = _parse_frontmatter(match.group("front"))
    body = match.group("body").strip()
    if not body:
        logger.warning("md_command_empty_body", extra={"path": str(path)})
        return None
    name = path.stem
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
        logger.warning("md_command_invalid_name", extra={"path": str(path), "command_name": name})
        return None
    return MdTemplateCommand(
        name=name,
        description=fm.description,
        category=fm.category,
        body_template=body,
        source_path=path,
        aliases=fm.aliases,
    )


def load_md_commands_into(registry, directory: Path) -> int:
    """Walk ``directory`` for ``*.md`` files and register each into
    ``registry``. Returns the number of commands loaded."""
    if not directory.exists() or not directory.is_dir():
        return 0
    loaded = 0
    for md in sorted(directory.glob("*.md")):
        cmd = load_md_command(md)
        if cmd is not None:
            registry.register(cmd)
            loaded += 1
    return loaded


__all__ = [
    "MdTemplateCommand",
    "load_md_command",
    "load_md_commands_into",
]
