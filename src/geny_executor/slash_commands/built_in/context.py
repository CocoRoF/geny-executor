"""``/context`` — show files currently loaded by the context loader."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import find_strategy, need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class ContextCommand(SlashCommand):
    name = "context"
    description = "Show files currently loaded by the context loader (CLAUDE.md / AGENTS.md / etc.)."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        loader = (
            find_strategy(ctx.pipeline, "context_loader")
            or find_strategy(ctx.pipeline, "context_provider")
        )
        if loader is None:
            return SlashResult(content="No context loader strategy configured.", success=False)
        # Try common method names for "what did you load last".
        for method in ("last_loaded_paths", "loaded_paths", "list_loaded"):
            fn = getattr(loader, method, None)
            if callable(fn):
                try:
                    paths = fn()
                except Exception as exc:  # noqa: BLE001
                    return SlashResult(content=f"Context query failed: {exc}", success=False)
                paths = list(paths or [])
                if not paths:
                    return SlashResult(content="No context files loaded.")
                lines = [f"**Context files** (loaded by `{method}`)", ""]
                for p in paths:
                    lines.append(f"- `{p}`")
                return SlashResult(content="\n".join(lines))
        return SlashResult(
            content="Context loader does not expose a list-loaded method.",
            success=False,
        )


__all__ = ["ContextCommand"]
