"""``/clear`` — reset message history (preserves session)."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import find_strategy, need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class ClearCommand(SlashCommand):
    name = "clear"
    description = "Clear message history (preserves session). New messages start fresh."
    category = SlashCategory.CONTROL

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        history = (
            find_strategy(ctx.pipeline, "history_provider")
            or find_strategy(ctx.pipeline, "history")
            or find_strategy(ctx.pipeline, "message_history")
        )
        if history is None:
            return SlashResult(
                content="No history provider strategy configured.",
                success=False,
            )
        clear_fn = getattr(history, "clear", None) or getattr(history, "reset", None)
        if not callable(clear_fn):
            return SlashResult(
                content="History provider does not expose clear().",
                success=False,
            )
        try:
            result = clear_fn()
            # Support both sync + async.
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:  # noqa: BLE001
            return SlashResult(
                content=f"History clear failed: {exc}",
                success=False,
            )
        return SlashResult(content="✓ Cleared message history.")


__all__ = ["ClearCommand"]
