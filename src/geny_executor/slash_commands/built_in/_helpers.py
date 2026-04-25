"""Shared helpers for built-in slash commands."""

from __future__ import annotations

from typing import Any, Optional

from geny_executor.slash_commands.types import SlashContext, SlashResult


def need_pipeline(ctx: SlashContext) -> Optional[SlashResult]:
    """Return an error SlashResult when the context has no pipeline.
    Caller short-circuits on non-None return."""
    if ctx.pipeline is None:
        return SlashResult(content="No active pipeline.", success=False)
    return None


def find_strategy(pipeline: Any, slot_name: str) -> Optional[Any]:
    """Best-effort strategy lookup. Pipeline implementations vary in
    their public surface; we try the most common shapes:

    1. ``pipeline.get_strategy(slot_name)``
    2. ``pipeline.<slot_name>``
    3. ``pipeline._strategies.get(slot_name)``
    4. Walk ``pipeline.stages`` and ask each ``stage.get_strategy_slots()``.

    Returns ``None`` when nothing is found — callers print an
    informative "not configured" message rather than raising.
    """
    if pipeline is None:
        return None

    # Pattern 1: explicit accessor.
    getter = getattr(pipeline, "get_strategy", None)
    if callable(getter):
        try:
            value = getter(slot_name)
        except Exception:
            value = None
        if value is not None:
            return value

    # Pattern 2: attribute.
    value = getattr(pipeline, slot_name, None)
    if value is not None and not callable(value):
        return value

    # Pattern 3: private dict.
    private = getattr(pipeline, "_strategies", None)
    if isinstance(private, dict):
        value = private.get(slot_name)
        if value is not None:
            return value

    # Pattern 4: walk stages.
    stages = getattr(pipeline, "stages", None)
    if stages:
        for stage in stages:
            slots_fn = getattr(stage, "get_strategy_slots", None)
            if callable(slots_fn):
                try:
                    slots = slots_fn()
                except Exception:
                    continue
                if isinstance(slots, dict) and slot_name in slots:
                    return slots[slot_name]
    return None


__all__ = ["find_strategy", "need_pipeline"]
