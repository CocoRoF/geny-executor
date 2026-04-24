"""Default implementation of Stage 13: Loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_loop.interface import LoopController
from geny_executor.stages.s13_loop.artifact.default.controllers import (
    BudgetAwareLoopController,
    MultiDimensionalBudgetController,
    SingleTurnController,
    StandardLoopController,
)


class LoopStage(Stage[Any, Any]):
    """Stage 13: Loop.

    Dual abstraction:
      - Level 2 controller: decides continue/complete/error/escalate
    """

    def __init__(
        self,
        controller: Optional[LoopController] = None,
        *,
        max_turns: Optional[int] = None,
        early_stop_on: Optional[List[str]] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "controller": StrategySlot(
                name="controller",
                strategy=controller or StandardLoopController(max_turns=max_turns),
                registry={
                    "standard": StandardLoopController,
                    "single_turn": SingleTurnController,
                    "budget_aware": BudgetAwareLoopController,
                    # Phase 7 S7.7 — pluggable multi-dimensional
                    # budget. Dimensions arrive via constructor; the
                    # zero-arg slot-swap path produces an empty
                    # dimension list (acts like StandardLoopController).
                    "multi_dim_budget": MultiDimensionalBudgetController,
                },
                description="Loop decision strategy",
            ),
        }
        self._max_turns = max_turns
        self._early_stop_on: List[str] = list(early_stop_on or [])

    @property
    def _controller(self) -> LoopController:
        return self._slots["controller"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "loop"

    @property
    def order(self) -> int:
        return 13

    @property
    def category(self) -> str:
        return "decision"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="loop",
            fields=[
                ConfigField(
                    name="max_turns",
                    type="integer",
                    label="Max Turns",
                    description="Hard cap on loop iterations. Blank to defer to state.max_iterations.",
                    default=0,
                    min_value=0,
                ),
                ConfigField(
                    name="early_stop_on",
                    type="array",
                    label="Early Stop Signals",
                    description="Completion signals that should abort the loop immediately.",
                    default=[],
                    item_type="string",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_turns": self._max_turns or 0,
            "early_stop_on": list(self._early_stop_on),
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_turns" in config:
            value = int(config["max_turns"])
            self._max_turns = value if value > 0 else None
            controller = self._slots["controller"].strategy
            if hasattr(controller, "_max_turns"):
                controller._max_turns = self._max_turns  # type: ignore[attr-defined]
        if "early_stop_on" in config:
            self._early_stop_on = list(config["early_stop_on"] or [])

    async def execute(self, input: Any, state: PipelineState) -> Any:
        upstream = state.loop_decision
        if upstream in ("complete", "error", "escalate"):
            decision = upstream
        elif self._early_stop_on and state.completion_signal in self._early_stop_on:
            decision = "complete"
        else:
            decision = self._controller.decide(state)

        state.loop_decision = decision

        state.add_event(
            f"loop.{decision}",
            {
                "iteration": state.iteration,
                "signal": state.completion_signal,
                "pending_tools": len(state.pending_tool_calls),
                "has_tool_results": bool(state.tool_results),
                "upstream_decision": upstream,
            },
        )

        state.tool_results = []
        return input
