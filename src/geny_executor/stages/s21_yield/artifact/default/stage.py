"""Default implementation of Stage 16: Yield."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s21_yield.interface import ResultFormatter
from geny_executor.stages.s21_yield.artifact.default.formatters import (
    DefaultFormatter,
    StreamingFormatter,
    StructuredFormatter,
)
from geny_executor.stages.s21_yield.artifact.default.multi_format import (
    MultiFormatFormatter,
)


class YieldStage(Stage[Any, Any]):
    """Stage 16: Yield.

    Dual abstraction:
      - Level 2 formatter: final output formatting
    """

    def __init__(self, formatter: Optional[ResultFormatter] = None):
        self._slots: Dict[str, StrategySlot] = {
            "formatter": StrategySlot(
                name="formatter",
                strategy=formatter or DefaultFormatter(),
                registry={
                    "default": DefaultFormatter,
                    "structured": StructuredFormatter,
                    "streaming": StreamingFormatter,
                    "multi_format": MultiFormatFormatter,
                },
                description="Final result formatting strategy",
            ),
        }

    @property
    def _formatter(self) -> ResultFormatter:
        return self._slots["formatter"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "yield"

    @property
    def order(self) -> int:
        return 16

    @property
    def category(self) -> str:
        return "egress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    async def execute(self, input: Any, state: PipelineState) -> Any:
        self._formatter.format(state)
        state.add_event(
            "yield.complete",
            {
                "text_length": len(state.final_text),
                "iterations": state.iteration,
                "total_cost_usd": state.total_cost_usd,
            },
        )
        return state.final_output if state.final_output is not None else state.final_text
