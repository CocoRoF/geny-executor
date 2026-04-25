"""Stage 14: Emit — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s17_emit.types import EmitResult


class Emitter(Strategy, ABC):
    """Level 2 strategy: how to emit results to external consumers.

    Class-level scheduling hints (S7.11). All optional; the legacy
    :class:`EmitterChain` ignores them and runs in declared order.
    :class:`OrderedEmitterChain` honours both:

    * ``requires`` — names of emitters that must succeed first. The
      chain topologically sorts emitters before emit. If a required
      emitter failed or was skipped, the dependent is skipped too.
    * ``timeout_seconds`` — per-emit wall-clock budget. Exceeding it
      counts toward the chain's backpressure threshold.
    """

    requires: Tuple[str, ...] = ()
    timeout_seconds: Optional[float] = None

    @abstractmethod
    async def emit(self, state: PipelineState) -> EmitResult:
        """Emit pipeline results. Return emission result."""
        ...
