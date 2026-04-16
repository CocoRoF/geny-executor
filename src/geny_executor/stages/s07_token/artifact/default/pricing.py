"""Cost calculators — concrete implementations for pricing."""

from __future__ import annotations

from typing import Dict, Optional

from geny_executor.core.state import TokenUsage
from geny_executor.stages.s07_token.interface import CostCalculator


# Anthropic pricing per million tokens (as of 2026-04)
# Source: https://docs.anthropic.com/en/docs/about-claude/pricing
# Cache write = 1.25x input, Cache read = 0.1x input (5-minute TTL)
ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    # ── Current models ──
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3},
    "claude-haiku-4-5-20251001": {
        "input": 1.0,
        "output": 5.0,
        "cache_write": 1.25,
        "cache_read": 0.1,
    },
    # ── Legacy models (still active) ──
    "claude-sonnet-4-5-20250929": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.3,
    },
    "claude-opus-4-5-20251101": {
        "input": 5.0,
        "output": 25.0,
        "cache_write": 6.25,
        "cache_read": 0.5,
    },
    "claude-opus-4-1-20250805": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.5,
    },
    # ── Deprecated (retiring 2026-06-15) ──
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.3,
    },
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.5,
    },
    # ── Aliases for prefix matching ──
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.1},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.5},
    "claude-opus-4-1": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.5},
    # ── Older legacy ──
    "claude-haiku-3-5-20241022": {
        "input": 0.80,
        "output": 4.0,
        "cache_write": 1.0,
        "cache_read": 0.08,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
    },
}

# OpenAI pricing per million tokens (as of 2026-04)
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 2.0, "output": 8.0},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Google Gemini pricing per million tokens (as of 2026-04)
# Source: https://ai.google.dev/pricing
GOOGLE_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-3.1-pro": {"input": 2.0, "output": 12.0},
    "gemini-3-flash": {"input": 0.50, "output": 3.0},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

# Unified pricing table
ALL_PRICING: Dict[str, Dict[str, float]] = {
    **ANTHROPIC_PRICING,
    **OPENAI_PRICING,
    **GOOGLE_PRICING,
}


class AnthropicPricingCalculator(CostCalculator):
    """Anthropic official pricing calculator.

    Kept for backward compatibility — use UnifiedPricingCalculator for
    multi-provider pipelines.
    """

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
        cost += (usage.cache_creation_input_tokens / 1_000_000) * prices.get(
            "cache_write", prices["input"] * 1.25
        )

        # Cache read
        cost += (usage.cache_read_input_tokens / 1_000_000) * prices.get(
            "cache_read", prices["input"] * 0.1
        )

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


class UnifiedPricingCalculator(CostCalculator):
    """Multi-provider pricing calculator.

    Covers Anthropic, OpenAI, and Google Gemini models.
    Uses cache pricing when available (Anthropic), falls back to
    simple input/output pricing for other providers.
    """

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self._pricing = {**ALL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

    @property
    def name(self) -> str:
        return "unified_pricing"

    @property
    def description(self) -> str:
        return "Multi-provider pricing (Anthropic + OpenAI + Google)"

    def calculate(self, usage: TokenUsage, model: str) -> float:
        prices = self._get_prices(model)
        if not prices:
            return 0.0

        cost = 0.0
        has_cache_pricing = "cache_write" in prices

        if has_cache_pricing:
            # Anthropic-style: separate cache write/read pricing
            regular_input = usage.input_tokens - usage.cache_read_input_tokens
            cost += (regular_input / 1_000_000) * prices["input"]
            cost += (usage.cache_creation_input_tokens / 1_000_000) * prices["cache_write"]
            cost += (usage.cache_read_input_tokens / 1_000_000) * prices["cache_read"]
        else:
            # Simple input pricing (OpenAI, Google)
            cost += (usage.input_tokens / 1_000_000) * prices["input"]

        cost += (usage.output_tokens / 1_000_000) * prices["output"]
        return cost

    def _get_prices(self, model: str) -> Optional[Dict[str, float]]:
        """Look up pricing: exact match → prefix match."""
        if model in self._pricing:
            return self._pricing[model]
        for key in self._pricing:
            if model.startswith(key.rsplit("-", 1)[0]):
                return self._pricing[key]
        return None
