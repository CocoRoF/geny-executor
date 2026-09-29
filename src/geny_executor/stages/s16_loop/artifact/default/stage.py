"""Default implementation of Stage 13: Loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s16_loop.interface import LoopController
from geny_executor.stages.s16_loop.repeat_stop import DEFAULT_STOP_AFTER, apply_repeat_stop
from geny_executor.stages.s16_loop.turn_budget import (
    DEFAULT_HARD_TOKENS,
    DEFAULT_SOFT_TOKENS,
    apply_step_limit,
    apply_turn_budget,
)
from geny_executor.stages.s16_loop.artifact.default.controllers import (
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
        repeat_stop_after: int = DEFAULT_STOP_AFTER,
        turn_soft_input_tokens: int = DEFAULT_SOFT_TOKENS,
        turn_hard_input_tokens: int = DEFAULT_HARD_TOKENS,
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
        self._repeat_stop_after = max(0, int(repeat_stop_after))
        self._turn_soft_input_tokens = max(0, int(turn_soft_input_tokens))
        self._turn_hard_input_tokens = max(0, int(turn_hard_input_tokens))

    @property
    def _controller(self) -> LoopController:
        return self._slots["controller"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "loop"

    @property
    def order(self) -> int:
        return 16

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
                    name="repeat_stop_after",
                    type="integer",
                    label="End the turn after refused calls",
                    description=(
                        "Tool calls refused in one turn (repeating a call that returned the "
                        "same result, or kept failing the same way) before the model is told "
                        "to report and stop. 0 turns this off."
                    ),
                    default=DEFAULT_STOP_AFTER,
                    min_value=0,
                ),
                ConfigField(
                    name="turn_soft_input_tokens",
                    type="integer",
                    label="Ask to wrap up after (input tokens per turn)",
                    description=(
                        "Cumulative prompt tokens one turn may read before the model is "
                        "told to finish the most valuable remaining step and report. "
                        "0 turns this note off."
                    ),
                    default=DEFAULT_SOFT_TOKENS,
                    min_value=0,
                ),
                ConfigField(
                    name="turn_hard_input_tokens",
                    type="integer",
                    label="End the turn after (input tokens per turn)",
                    description=(
                        "Cumulative prompt tokens after which the model is told to stop "
                        "calling tools and report; the turn ends with that response. "
                        "0 turns the budget off."
                    ),
                    default=DEFAULT_HARD_TOKENS,
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
            "repeat_stop_after": self._repeat_stop_after,
            "turn_soft_input_tokens": self._turn_soft_input_tokens,
            "turn_hard_input_tokens": self._turn_hard_input_tokens,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_turns" in config:
            value = int(config["max_turns"])
            self._max_turns = value if value > 0 else None
            controller = self._slots["controller"].strategy
            if self._controller_declares_max_turns(controller):
                # 2026-06-09 audit §2.1: the old hasattr('_max_turns')
                # poke silently skipped MultiDimensionalBudgetController
                # (it has no such attribute), so a manifest-level
                # max_turns was inert for exactly the controller Geny
                # prod runs. configure() is the contract now — the
                # controller decides what max_turns means for it.
                controller.configure({"max_turns": value})
            elif hasattr(controller, "_max_turns"):
                # Legacy fallback for host-supplied controllers that
                # predate the configure() contract.
                controller._max_turns = self._max_turns  # type: ignore[attr-defined]
        if "early_stop_on" in config:
            self._early_stop_on = list(config["early_stop_on"] or [])
        if "repeat_stop_after" in config:
            self._repeat_stop_after = max(0, int(config["repeat_stop_after"] or 0))
        if "turn_soft_input_tokens" in config:
            self._turn_soft_input_tokens = max(0, int(config["turn_soft_input_tokens"] or 0))
        if "turn_hard_input_tokens" in config:
            self._turn_hard_input_tokens = max(0, int(config["turn_hard_input_tokens"] or 0))

    @staticmethod
    def _controller_declares_max_turns(controller: LoopController) -> bool:
        """True when the controller's config_schema() exposes ``max_turns``."""
        try:
            schema = controller.config_schema()
        except Exception:
            return False
        if schema is None:
            return False
        return any(getattr(f, "name", "") == "max_turns" for f in getattr(schema, "fields", []))

    async def execute(self, input: Any, state: PipelineState) -> Any:
        upstream = state.loop_decision
        if upstream in ("complete", "error", "escalate"):
            decision = upstream
        elif self._early_stop_on and state.completion_signal in self._early_stop_on:
            decision = "complete"
        else:
            decision = self._controller.decide(state)
        decision = apply_repeat_stop(state, decision, self._repeat_stop_after)
        decision = apply_turn_budget(
            state, decision, self._turn_soft_input_tokens, self._turn_hard_input_tokens
        )
        decision = apply_step_limit(state, decision)

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
