"""Memory update strategies — Level 2 strategies for post-execution memory."""

from __future__ import annotations

from abc import abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class MemoryUpdateStrategy(Strategy):
    """Base interface for memory update logic."""

    @abstractmethod
    async def update(self, state: PipelineState) -> None:
        """Update memory based on execution results."""
        ...


class AppendOnlyStrategy(MemoryUpdateStrategy):
    """Append conversation to history only."""

    @property
    def name(self) -> str:
        return "append_only"

    @property
    def description(self) -> str:
        return "Appends conversation to history"

    async def update(self, state: PipelineState) -> None:
        # Messages are already accumulated in state.messages
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
    """Reflective — extract key information for long-term storage.

    This is a structural placeholder. Full implementation would use
    a lightweight API call to summarize/extract.
    """

    @property
    def name(self) -> str:
        return "reflective"

    @property
    def description(self) -> str:
        return "Extracts and stores key information from conversation"

    async def update(self, state: PipelineState) -> None:
        # Mark that reflection should happen
        state.metadata["needs_reflection"] = True
        state.add_event(
            "memory.reflection_queued",
            {
                "message_count": len(state.messages),
                "iteration": state.iteration,
            },
        )
