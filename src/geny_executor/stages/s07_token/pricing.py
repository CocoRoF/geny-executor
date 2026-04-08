"""Cost calculators — Level 2 strategies for pricing."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState, TokenUsage


# Anthropic pricing per million tokens (as of 2026)
ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    # Aliases
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3},
}


class CostCalculator(Strategy):
    """Base interface for cost calculation."""

    @abstractmethod
    def calculate(self, usage: TokenUsage, model: str) -> float:
        """Calculate cost in USD from token usage."""
        ...


class AnthropicPricingCalculator(CostCalculator):
    """Anthropic official pricing calculator."""

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self._pricing = {**ANTHROPIC_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

    @property
    def name(self) -> str:
        return "anthropic_pricing"

    @property
    def description(self) -> str:
        return "Anthropic official pricing calculator"

    def calculate(self, usage: TokenUsage, model: str) -> float:
        prices = self._get_prices(model)
        if not prices:
            return 0.0

        cost = 0.0
        # Regular input tokens (excluding cached)
        regular_input = usage.input_tokens - usage.cache_read_input_tokens
        cost += (regular_input / 1_000_000) * prices["input"]

        # Output tokens
        cost += (usage.output_tokens / 1_000_000) * prices["output"]

        # Cache write
        cost += (usage.cache_creation_input_tokens / 1_000_000) * prices.get("cache_write", prices["input"] * 1.25)

        # Cache read
        cost += (usage.cache_read_input_tokens / 1_000_000) * prices.get("cache_read", prices["input"] * 0.1)

        return cost

    def _get_prices(self, model: str) -> Optional[Dict[str, float]]:
        """Look up pricing, trying exact match then prefix match."""
        if model in self._pricing:
            return self._pricing[model]
        # Prefix match
        for key in self._pricing:
            if model.startswith(key.rsplit("-", 1)[0]):
                return self._pricing[key]
        return None


class CustomPricingCalculator(CostCalculator):
    """Custom flat-rate pricing."""

    def __init__(self, input_per_million: float = 3.0, output_per_million: float = 15.0):
        self._input_rate = input_per_million
        self._output_rate = output_per_million

    @property
    def name(self) -> str:
        return "custom_pricing"

    @property
    def description(self) -> str:
        return "Custom flat-rate pricing"

    def calculate(self, usage: TokenUsage, model: str) -> float:
        cost = (usage.input_tokens / 1_000_000) * self._input_rate
        cost += (usage.output_tokens / 1_000_000) * self._output_rate
        return cost
