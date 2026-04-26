"""``/memory`` — show recent memory notes from the active provider."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import find_strategy, need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class MemoryCommand(SlashCommand):
    name = "memory"
    description = "Show recent memory notes from the active memory provider."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        memory = find_strategy(ctx.pipeline, "memory_provider") or find_strategy(
            ctx.pipeline, "memory"
        )
        if memory is None:
            return SlashResult(content="No memory provider configured.", success=False)
        recent_fn = (
            getattr(memory, "recent", None)
            or getattr(memory, "list_recent", None)
            or getattr(memory, "tail", None)
        )
        if not callable(recent_fn):
            return SlashResult(
                content="Memory provider does not expose a recent() / list_recent() method.",
                success=False,
            )
        try:
            limit = int(args[0]) if args else 10
        except ValueError:
            limit = 10
        try:
            result = recent_fn(limit=limit) if _accepts_kw(recent_fn, "limit") else recent_fn(limit)
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:  # noqa: BLE001
            return SlashResult(content=f"Memory query failed: {exc}", success=False)
        notes = list(result or [])
        if not notes:
            return SlashResult(content="No memory notes.")
        lines = [f"**Recent memory** (last {len(notes)})", ""]
        for note in notes[:limit]:
            summary = getattr(note, "summary", None) or getattr(note, "text", None) or str(note)
            lines.append(f"- {summary}")
        return SlashResult(content="\n".join(lines))


def _accepts_kw(fn, name: str) -> bool:
    try:
        import inspect

        sig = inspect.signature(fn)
        return name in sig.parameters
    except (TypeError, ValueError):
        return False


__all__ = ["MemoryCommand"]
