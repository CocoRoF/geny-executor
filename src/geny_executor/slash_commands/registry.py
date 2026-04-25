"""Slash command registry — register / resolve / list / discover."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
)

logger = logging.getLogger(__name__)


class SlashCommandRegistry:
    """Process-singleton-ish registry. A fresh instance can be created
    for tests; production code uses :func:`get_default_registry`.

    Discovery hierarchy (last-registered wins on name collision):

      1. Built-in commands (registered by their module imports).
      2. Service commands (host calls ``register`` at startup).
      3. Project / user paths (``discover_paths``) — walks each path
         for ``*.md`` files with frontmatter and registers them as
         :class:`MdTemplateCommand` (added in PR-A.2.4).
    """

    def __init__(self) -> None:
        self._commands: Dict[str, SlashCommand] = {}
        self._discovery_paths: List[Path] = []

    def register(self, cmd: SlashCommand) -> None:
        if cmd.name in self._commands:
            logger.warning(
                "slash_command_overwritten",
                extra={"command_name": cmd.name, "old_class": type(self._commands[cmd.name]).__name__},
            )
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases or []:
            if alias in self._commands and self._commands[alias] is not cmd:
                logger.warning("slash_command_alias_collision", extra={"alias": alias})
            self._commands[alias] = cmd

    def deregister(self, name: str) -> bool:
        cmd = self._commands.pop(name, None)
        if cmd is None:
            return False
        # Drop aliases too — but only ones that point to *this* command.
        for alias in list(cmd.aliases or []):
            if self._commands.get(alias) is cmd:
                self._commands.pop(alias, None)
        return True

    def resolve(self, name: str) -> Optional[SlashCommand]:
        return self._commands.get(name)

    def list_all(self) -> List[SlashCommand]:
        # Deduplicate aliases pointing at the same instance.
        seen = set()
        out: List[SlashCommand] = []
        for cmd in self._commands.values():
            if id(cmd) in seen:
                continue
            seen.add(id(cmd))
            out.append(cmd)
        return sorted(out, key=lambda c: c.name)

    def list_by_category(self, category: SlashCategory) -> List[SlashCommand]:
        return [c for c in self.list_all() if c.category == category]

    def discover_paths(self, path: Path) -> int:
        """Register every ``*.md`` slash command found under ``path``.

        Returns the count loaded. Missing / non-directory ``path`` is
        recorded but does not raise — it's normal for a host to wire
        ``~/.geny/commands/`` even when the user hasn't created it.

        The actual markdown loader lands in PR-A.2.4
        (:class:`MdTemplateCommand`); for now this method records the
        path and returns 0.
        """
        self._discovery_paths.append(path)
        if not path.exists() or not path.is_dir():
            return 0
        # MdTemplateCommand loader is added in PR-A.2.4.
        try:
            from geny_executor.slash_commands.md_template import (
                load_md_commands_into,
            )
        except ImportError:
            return 0
        return load_md_commands_into(self, path)

    @property
    def discovery_paths(self) -> List[Path]:
        return list(self._discovery_paths)


_DEFAULT: Optional[SlashCommandRegistry] = None


def get_default_registry() -> SlashCommandRegistry:
    """Process-wide singleton. Tests get a fresh instance via
    :func:`reset_default_registry` (or by constructing one directly)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SlashCommandRegistry()
    return _DEFAULT


def reset_default_registry() -> SlashCommandRegistry:
    """Test helper. Replaces the singleton with a fresh registry and
    returns it. Production code should never call this."""
    global _DEFAULT
    _DEFAULT = SlashCommandRegistry()
    return _DEFAULT


__all__ = [
    "SlashCommandRegistry",
    "get_default_registry",
    "reset_default_registry",
]
