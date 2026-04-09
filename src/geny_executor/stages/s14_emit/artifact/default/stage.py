"""Default implementation of Stage 14: Emit."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_emit.interface import Emitter
from geny_executor.stages.s14_emit.types import EmitterChain


class EmitStage(Stage[Any, Any]):
    """Stage 14: Emit.

    Delivers pipeline results to external consumers via emitter chain.
    """

    def __init__(self, emitters: Optional[List[Emitter]] = None):
        self._chain = EmitterChain(emitters or [])

    @property
    def name(self) -> str:
        return "emit"

    @property
    def order(self) -> int:
        return 14

    @property
    def category(self) -> str:
        return "egress"

    def add_emitter(self, emitter: Emitter) -> None:
        self._chain.add(emitter)

    def should_bypass(self, state: PipelineState) -> bool:
        return len(self._chain.emitters) == 0

    async def execute(self, input: Any, state: PipelineState) -> Any:
        if not self._chain.emitters:
            return input

        state.add_event(
            "emit.start",
            {
                "emitter_count": len(self._chain.emitters),
                "channels": [e.name for e in self._chain.emitters],
            },
        )

        results = await self._chain.emit_all(state)

        channels = []
        for r in results:
            channels.extend(r.channels)

        state.add_event(
            "emit.complete",
            {
                "channels_emitted": channels,
                "all_emitted": all(r.emitted for r in results),
            },
        )

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="emitters",
                current_impl=", ".join(type(e).__name__ for e in self._chain.emitters) or "none",
                available_impls=[
                    "TextEmitter",
                    "CallbackEmitter",
                    "VTuberEmitter",
                    "TTSEmitter",
                ],
            ),
        ]
