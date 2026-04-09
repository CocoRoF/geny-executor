"""Emit stage data types."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from geny_executor.stages.s14_emit.interface import Emitter
    from geny_executor.core.state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class EmitResult:
    """Result of emission."""

    emitted: bool = True
    channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmitterChain:
    """Chain of emitters — runs all in sequence."""

    def __init__(self, emitters: Optional[List[Emitter]] = None):
        self._emitters = emitters or []

    def add(self, emitter: Emitter) -> None:
        self._emitters.append(emitter)

    async def emit_all(self, state: PipelineState) -> List[EmitResult]:
        results = []
        for emitter in self._emitters:
            try:
                result = await emitter.emit(state)
                results.append(result)
            except Exception as e:
                logger.warning("Emitter %s failed: %s", emitter.name, e)
                results.append(
                    EmitResult(emitted=False, channels=[emitter.name], metadata={"error": str(e)})
                )
        return results

    @property
    def emitters(self) -> List[Emitter]:
        return list(self._emitters)
