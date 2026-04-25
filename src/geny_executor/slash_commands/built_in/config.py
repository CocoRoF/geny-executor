"""``/config`` — show the active manifest's strategy slot map."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class ConfigCommand(SlashCommand):
    name = "config"
    description = "Show the active strategy slot map (one row per stage)."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        stages = getattr(ctx.pipeline, "stages", None) or []
        if not stages:
            return SlashResult(content="Pipeline has no stages.", success=False)
        lines = ["**Active strategy slots**", ""]
        for stage in stages:
            stage_name = getattr(stage, "name", None) or type(stage).__name__
            slots_fn = getattr(stage, "get_strategy_slots", None)
            if not callable(slots_fn):
                continue
            try:
                slots = slots_fn() or {}
            except Exception:
                continue
            if not slots:
                continue
            lines.append(f"### {stage_name}")
            for slot_name, slot in slots.items():
                # Slots may carry a ``strategy`` attribute, or be the
                # strategy themselves. Best effort.
                strategy = getattr(slot, "strategy", slot)
                impl = type(strategy).__name__
                lines.append(f"- `{slot_name}` → `{impl}`")
            lines.append("")
        return SlashResult(content="\n".join(lines).rstrip())


__all__ = ["ConfigCommand"]
