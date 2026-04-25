"""``/cancel`` — cancel the active pipeline run."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class CancelCommand(SlashCommand):
    name = "cancel"
    description = "Cancel the in-flight pipeline run (if any)."
    category = SlashCategory.CONTROL

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        # Try the most common shapes for "ask the pipeline to stop".
        for method in ("stop", "cancel", "request_stop", "abort"):
            fn = getattr(ctx.pipeline, method, None)
            if callable(fn):
                try:
                    result = fn()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:  # noqa: BLE001
                    return SlashResult(content=f"Cancel failed: {exc}", success=False)
                return SlashResult(content=f"✓ Pipeline cancel requested via {method}().")
        return SlashResult(
            content="Pipeline does not expose stop / cancel / request_stop / abort.",
            success=False,
        )


__all__ = ["CancelCommand"]
