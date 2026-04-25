"""``/help`` — list every registered slash command."""

from __future__ import annotations

from geny_executor.slash_commands.registry import get_default_registry
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class HelpCommand(SlashCommand):
    name = "help"
    description = "List all available slash commands grouped by category."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        # ``ctx.extras["slash_registry"]`` lets the host pass an
        # alternate registry (multi-tenant / per-session). Falls back
        # to the default singleton.
        registry = ctx.extras.get("slash_registry") or get_default_registry()
        all_cmds = registry.list_all()
        if not all_cmds:
            return SlashResult(content="No slash commands registered.")
        lines = ["**Available slash commands**", ""]
        for cat in SlashCategory:
            cmds = [c for c in all_cmds if c.category == cat]
            if not cmds:
                continue
            lines.append(f"### {cat.value.title()}")
            for cmd in cmds:
                aliases = ""
                if cmd.aliases:
                    aliases = f" (aliases: {', '.join('/' + a for a in cmd.aliases)})"
                lines.append(f"- `/{cmd.name}` — {cmd.description}{aliases}")
            lines.append("")
        return SlashResult(content="\n".join(lines).rstrip())


__all__ = ["HelpCommand"]
