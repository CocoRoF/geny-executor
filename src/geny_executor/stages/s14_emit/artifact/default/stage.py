"""Default implementation of Stage 14: Emit."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from geny_executor.core.slot import SlotChain
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_emit.interface import Emitter
from geny_executor.stages.s14_emit.types import EmitterChain
from geny_executor.stages.s14_emit.artifact.default.emitters import (
    CallbackEmitter,
    TextEmitter,
    TTSEmitter,
    VTuberEmitter,
)


class EmitStage(Stage[Any, Any]):
    """Stage 14: Emit.

    Delivers pipeline results to external consumers via emitter chain.
    """

    def __init__(self, emitters: Optional[List[Emitter]] = None):
        self._chains: Dict[str, SlotChain] = {
            "emitters": SlotChain(
                name="emitters",
                items=list(emitters or []),
                registry={
                    "text": TextEmitter,
                    "callback": CallbackEmitter,
                    "vtuber": VTuberEmitter,
                    "tts": TTSEmitter,
                },
                description="Ordered chain of output emitters",
            ),
        }

    @property
    def emitters(self) -> SlotChain:
        """Public handle on the emitter chain (list/mutate items directly)."""
        return self._chains["emitters"]

    @property
    def _emitter_chain(self) -> SlotChain:
        return self._chains["emitters"]

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
        """Append an emitter to the chain.

        .. deprecated::
            Prefer :meth:`add_to_chain("emitters", impl_name)` for
            hot-swappable configuration. Retained for backward compatibility
            with builders that pre-construct Emitter instances.
        """
        warnings.warn(
            "EmitStage.add_emitter() is deprecated; use "
            "stage.add_to_chain('emitters', impl_name) or pre-populate via "
            "EmitStage(emitters=[...]).",
            DeprecationWarning,
            stacklevel=2,
        )
        self._emitter_chain.add(emitter)

    def get_strategy_chains(self) -> Dict[str, SlotChain]:
        return self._chains

    def should_bypass(self, state: PipelineState) -> bool:
        return len(self._emitter_chain.items) == 0

    async def execute(self, input: Any, state: PipelineState) -> Any:
        emitters = self._emitter_chain.items
        if not emitters:
            return input

        chain = EmitterChain(emitters)

        state.add_event(
            "emit.start",
            {
                "emitter_count": len(emitters),
                "channels": [e.name for e in emitters],
            },
        )

        results = await chain.emit_all(state)

        channels: List[str] = []
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
