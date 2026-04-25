"""``/status`` — dump session info (preset / model / active strategies)."""

from __future__ import annotations

from typing import List

from geny_executor.slash_commands.built_in._helpers import need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class StatusCommand(SlashCommand):
    name = "status"
    description = "Show session info — preset, model, active strategies."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        pipeline = ctx.pipeline
        manifest = getattr(pipeline, "manifest", None)
        lines: List[str] = ["**Session status**", ""]
        if ctx.session_id:
            lines.append(f"- Session: `{ctx.session_id}`")
        if manifest is not None:
            preset = getattr(manifest, "preset_name", None)
            if preset:
                lines.append(f"- Preset: `{preset}`")
            model = getattr(manifest, "model", None)
            if model:
                lines.append(f"- Model: `{model}`")
        stages = getattr(pipeline, "stages", None) or []
        active_stages = [
            getattr(s, "name", None) or type(s).__name__
            for s in stages
            if getattr(s, "is_active", lambda: True)()
        ]
        lines.append(f"- Active stages: {len(active_stages)}")
        if active_stages:
            preview = ", ".join(active_stages[:8])
            suffix = ", ..." if len(active_stages) > 8 else ""
            lines.append(f"  ({preview}{suffix})")
        return SlashResult(content="\n".join(lines))


__all__ = ["StatusCommand"]
