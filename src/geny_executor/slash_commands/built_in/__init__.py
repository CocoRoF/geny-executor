"""Built-in slash commands (PR-A.2.2 / PR-A.2.3).

Importing this module registers each shipped command into the default
registry. Hosts that build a fresh registry (tests, multi-tenant
scenarios) call :func:`install_built_in_commands` against their own.

Categories:

* INTROSPECTION — read-only state queries
  (cost / status / help / memory / context).
* CONTROL — mutating operations
  (clear / cancel / compact / config / model / preset_info / tasks).
"""

from __future__ import annotations

from geny_executor.slash_commands.built_in.clear import ClearCommand
from geny_executor.slash_commands.built_in.context import ContextCommand
from geny_executor.slash_commands.built_in.cost import CostCommand
from geny_executor.slash_commands.built_in.help import HelpCommand
from geny_executor.slash_commands.built_in.memory import MemoryCommand
from geny_executor.slash_commands.built_in.status import StatusCommand
from geny_executor.slash_commands.registry import (
    SlashCommandRegistry,
    get_default_registry,
)


def install_built_in_commands(registry: SlashCommandRegistry) -> int:
    """Register every shipped built-in into ``registry``. Returns the
    count registered. Idempotent — re-registering the same command
    overwrites silently (logged at WARNING by the registry)."""
    classes = [
        CostCommand,
        ClearCommand,
        StatusCommand,
        HelpCommand,
        MemoryCommand,
        ContextCommand,
    ]
    for cls in classes:
        registry.register(cls())
    return len(classes)


# Auto-install into the default registry on import. Tests that want
# isolation use SlashCommandRegistry() + install_built_in_commands.
install_built_in_commands(get_default_registry())


__all__ = [
    "ClearCommand",
    "ContextCommand",
    "CostCommand",
    "HelpCommand",
    "MemoryCommand",
    "StatusCommand",
    "install_built_in_commands",
]
