"""Default artifact strategies for Stage 15: Memory."""

from __future__ import annotations

from typing import List

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Insight
from geny_executor.stages.s15_memory.insight import (
    INSIGHTS_KEY,
    drain_pending_insights,
)
from geny_executor.stages.s15_memory.interface import MemoryUpdateStrategy


class AppendOnlyStrategy(MemoryUpdateStrategy):
    """Append conversation to history only."""

    @property
    def name(self) -> str:
        return "append_only"

    @property
    def description(self) -> str:
        return "Appends conversation to history"

    async def update(self, state: PipelineState) -> None:
        pass


class NoMemoryStrategy(MemoryUpdateStrategy):
    """No memory updates — fully stateless."""

    @property
    def name(self) -> str:
        return "no_memory"

    @property
    def description(self) -> str:
        return "No memory updates (stateless)"

    async def update(self, state: PipelineState) -> None:
        pass


class ReflectiveStrategy(MemoryUpdateStrategy):
    """Reflective — extract key information for long-term storage."""

    @property
    def name(self) -> str:
        return "reflective"

    @property
    def description(self) -> str:
        return "Extracts and stores key information from conversation"

    async def update(self, state: PipelineState) -> None:
        state.metadata["needs_reflection"] = True
        state.add_event(
            "memory.reflection_queued",
            {"message_count": len(state.messages), "iteration": state.iteration},
        )


class StructuredReflectiveStrategy(MemoryUpdateStrategy):
    """Drain queued :class:`Insight` payloads into the recorded collection.

    Companion to :class:`ReflectiveStrategy`. Where ``ReflectiveStrategy``
    only flags ``needs_reflection`` and waits for someone else to act,
    this strategy assumes a *producer* (host code, a sub-agent, a parsing
    stage) has already pushed concrete insights into
    ``state.metadata[PENDING_INSIGHTS_KEY]`` and turns each one into a
    validated :class:`Insight` record on
    ``state.metadata[INSIGHTS_KEY]``. Each successful record emits a
    ``memory.insight_recorded`` event with the same shape as
    :meth:`Insight.to_event`; coercion failures emit
    ``memory.insight_invalid`` and the queue is cleared either way so a
    bad payload cannot wedge subsequent runs.

    The strategy also clears ``needs_reflection`` once it processes the
    queue, signalling that the structured pass has consumed whatever the
    producer left behind.
    """

    @property
    def name(self) -> str:
        return "structured_reflective"

    @property
    def description(self) -> str:
        return "Validates queued InsightRecords and appends them to state.metadata"

    async def update(self, state: PipelineState) -> None:
        try:
            drained = list(drain_pending_insights(state))
        except (TypeError, ValueError) as exc:
            state.add_event(
                "memory.insight_invalid",
                {"error": str(exc), "iteration": state.iteration},
            )
            state.metadata["needs_reflection"] = False
            return

        recorded: List[Insight] = state.metadata.setdefault(INSIGHTS_KEY, [])
        for insight in drained:
            recorded.append(insight)
            state.add_event("memory.insight_recorded", insight.to_event())

        state.add_event(
            "memory.structured_reflection_done",
            {
                "recorded": len(drained),
                "total": len(recorded),
                "iteration": state.iteration,
            },
        )
        state.metadata["needs_reflection"] = False
