"""``/compact`` — manually trigger context summarization (Stage 19)."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import find_strategy, need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class CompactCommand(SlashCommand):
    name = "compact"
    description = "Manually trigger context summarization (Stage 19)."
    category = SlashCategory.CONTROL

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        summarizer = (
            find_strategy(ctx.pipeline, "summarize_strategy")
            or find_strategy(ctx.pipeline, "summarizer")
            or find_strategy(ctx.pipeline, "summary_provider")
        )
        if summarizer is None:
            return SlashResult(
                content="No summarize strategy configured (Stage 19 inactive).",
                success=False,
            )
        for method in ("summarize_now", "compact", "run"):
            fn = getattr(summarizer, method, None)
            if callable(fn):
                try:
                    result = fn()
                    if hasattr(result, "__await__"):
                        result = await result
                except Exception as exc:  # noqa: BLE001
                    return SlashResult(content=f"Compaction failed: {exc}", success=False)
                tokens_saved = getattr(result, "tokens_compressed", None) if result else None
                msg = "✓ Context summarized."
                if tokens_saved:
                    msg += f" ({tokens_saved} tokens compressed)"
                return SlashResult(content=msg)
        return SlashResult(
            content="Summarize strategy does not expose summarize_now / compact / run.",
            success=False,
        )


__all__ = ["CompactCommand"]
