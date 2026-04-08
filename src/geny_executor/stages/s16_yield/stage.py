"""Stage 16: Yield — final result packaging."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s16_yield.formatters import DefaultFormatter, ResultFormatter


class YieldStage(Stage[Any, Any]):
    """Stage 16: Yield.

    Dual abstraction:
      - Level 2 formatter: final output formatting
    """

    def __init__(self, formatter: Optional[ResultFormatter] = None):
        self._formatter = formatter or DefaultFormatter()

    @property
    def name(self) -> str:
        return "yield"

    @property
    def order(self) -> int:
        return 16

    @property
    def category(self) -> str:
        return "egress"

    async def execute(self, input: Any, state: PipelineState) -> Any:
        self._formatter.format(state)
        state.add_event("yield.complete", {
            "text_length": len(state.final_text),
            "iterations": state.iteration,
            "total_cost_usd": state.total_cost_usd,
        })
        return state.final_output if state.final_output is not None else state.final_text

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="formatter",
                current_impl=type(self._formatter).__name__,
                available_impls=["DefaultFormatter", "StructuredFormatter", "StreamingFormatter"],
            ),
        ]
