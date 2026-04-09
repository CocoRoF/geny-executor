"""Default artifact strategies for Stage 15: Memory."""

from __future__ import annotations

from geny_executor.core.state import PipelineState
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
