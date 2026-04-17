"""Stage 7: Token — tracks usage and calculates cost."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s07_token.interface import TokenTracker, CostCalculator
from geny_executor.stages.s07_token.artifact.default.trackers import (
    DefaultTracker,
    DetailedTracker,
)
from geny_executor.stages.s07_token.artifact.default.pricing import (
    AnthropicPricingCalculator,
    CustomPricingCalculator,
    UnifiedPricingCalculator,
)


class TokenStage(Stage[Any, Any]):
    """Stage 7: Token.

    Dual abstraction:
      - Level 2 tracker: token usage accumulation
      - Level 2 calculator: cost computation
    """

    def __init__(
        self,
        tracker: Optional[TokenTracker] = None,
        calculator: Optional[CostCalculator] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "tracker": StrategySlot(
                name="tracker",
                strategy=tracker or DefaultTracker(),
                registry={
                    "default": DefaultTracker,
                    "detailed": DetailedTracker,
                },
                description="Token usage tracking strategy",
            ),
            "calculator": StrategySlot(
                name="calculator",
                strategy=calculator or AnthropicPricingCalculator(),
                registry={
                    "anthropic_pricing": AnthropicPricingCalculator,
                    "custom_pricing": CustomPricingCalculator,
                    "unified_pricing": UnifiedPricingCalculator,
                },
                description="Cost calculation strategy",
            ),
        }

    @property
    def _tracker(self) -> TokenTracker:
        return self._slots["tracker"].strategy  # type: ignore[return-value]

    @property
    def _calculator(self) -> CostCalculator:
        return self._slots["calculator"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "token"

    @property
    def order(self) -> int:
        return 7

    @property
    def category(self) -> str:
        return "execution"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Get API response from state
        response = state.last_api_response
        if not isinstance(response, APIResponse):
            return input

        # Track usage
        usage = self._tracker.track(response, state)

        # Calculate cost
        cost = self._calculator.calculate(usage, state.model)
        state.accumulate_cost(cost)

        # Update cache metrics
        if usage.cache_creation_input_tokens > 0:
            state.cache_metrics.total_cache_writes += 1
        if usage.cache_read_input_tokens > 0:
            state.cache_metrics.total_cache_reads += 1

        state.add_event(
            "token.tracked",
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_write": usage.cache_creation_input_tokens,
                "cache_read": usage.cache_read_input_tokens,
                "cost_usd": cost,
                "total_cost_usd": state.total_cost_usd,
            },
        )

        return input
