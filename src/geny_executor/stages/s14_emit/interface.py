"""Stage 14: Emit — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_emit.types import EmitResult


class Emitter(Strategy, ABC):
    """Level 2 strategy: how to emit results to external consumers."""

    @abstractmethod
    async def emit(self, state: PipelineState) -> EmitResult:
        """Emit pipeline results. Return emission result."""
        ...
