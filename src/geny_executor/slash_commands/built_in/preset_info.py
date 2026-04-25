"""``/preset-info`` — show preset metadata. Mutation is host-specific (e.g. Geny `/preset`)."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class PresetInfoCommand(SlashCommand):
    name = "preset-info"
    description = "Show the active preset's metadata (name + any preset_metadata dict)."
    category = SlashCategory.INTROSPECTION
    aliases = ["preset_info"]

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        manifest = getattr(ctx.pipeline, "manifest", None)
        if manifest is None:
            return SlashResult(content="Pipeline has no manifest.", success=False)
        preset = getattr(manifest, "preset_name", None) or "(unnamed)"
        meta = getattr(manifest, "preset_metadata", None) or {}
        lines = [f"**Preset:** `{preset}`"]
        if meta:
            lines.append("")
            for k, v in sorted(meta.items()):
                lines.append(f"- {k}: {v}")
        return SlashResult(content="\n".join(lines))


__all__ = ["PresetInfoCommand"]
