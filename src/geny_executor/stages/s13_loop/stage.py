"""Stage 13: Loop — agent loop control."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_loop.controllers import (
    LoopController,
    LoopDecision,
    StandardLoopController,
)


class LoopStage(Stage[Any, Any]):
    """Stage 13: Loop.

    Dual abstraction:
      - Level 2 controller: decides continue/complete/error/escalate
    """

    def __init__(self, controller: Optional[LoopController] = None):
        self._controller = controller or StandardLoopController()

    @property
    def name(self) -> str:
        return "loop"

    @property
    def order(self) -> int:
        return 13

    @property
    def category(self) -> str:
        return "decision"

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Respect upstream decision from Stage 12 (Evaluate) if it's terminal.
        # Only override with controller if upstream said "continue".
        upstream = state.loop_decision
        if upstream in ("complete", "error", "escalate"):
            decision = upstream
        else:
            decision = self._controller.decide(state)

        state.loop_decision = decision

        state.add_event(f"loop.{decision}", {
            "iteration": state.iteration,
            "signal": state.completion_signal,
            "pending_tools": len(state.pending_tool_calls),
            "has_tool_results": bool(state.tool_results),
            "upstream_decision": upstream,
        })

        # Always clear tool_results after decision (they've been consumed)
        state.tool_results = []

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="controller",
                current_impl=type(self._controller).__name__,
                available_impls=[
                    "StandardLoopController",
                    "SingleTurnController",
                    "BudgetAwareLoopController",
                ],
            ),
        ]
