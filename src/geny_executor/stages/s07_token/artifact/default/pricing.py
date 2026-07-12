"""Cost calculators — concrete implementations for pricing."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s07_token.interface import CostCalculator


def _lookup_prices(
    pricing: Dict[str, Dict[str, float]], model: str
) -> Optional[Dict[str, float]]:
    """Resolve a model id to its price row: exact match → longest key prefix.

    2.51.0 (audit C4): the old ``model.startswith(key.rsplit("-",1)[0])``
    truncated every key by one segment and returned the FIRST dict entry
    that matched by insertion order — so an unlisted ``claude-opus-4-1-<new>``
    could bind to ``opus-4-6`` rates (a 3x error). Now we match the full
    key as a prefix and prefer the LONGEST matching known key, which is
    unambiguous: ``claude-opus-4-1-<new>`` binds to ``claude-opus-4-1``,
    never to ``claude-opus-4-6``.
    """
    if model in pricing:
        return pricing[model]
    best_key: Optional[str] = None
    for key in pricing:
        if model.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return pricing[best_key] if best_key is not None else None


def _price_usage(usage: TokenUsage, prices: Dict[str, float]) -> float:
    """Cost of ``usage`` under a price row, provider-semantics aware.

    The load-bearing distinction (audit D1 — the negative-cost bug): the
    two provider families report ``input_tokens`` differently.

    - **Anthropic** rows carry a ``cache_write`` rate; their
      ``input_tokens`` is the UNCACHED input only (cache reads/creations
      are separate additive fields). So the three token buckets are
      disjoint and are priced independently — subtracting ``cache_read``
      from ``input_tokens`` (as the pre-2.51 code did) double-counted the
      discount and drove cache-heavy turns negative once aggressive
      caching made ``cache_read`` routinely exceed ``input_tokens``.
    - **OpenAI / Google** rows have no ``cache_write``; their
      ``input_tokens`` (``prompt_tokens`` / ``prompt_token_count``)
      already INCLUDES the cached portion. Here ``cache_read`` IS a
      subset to be discounted — but only when the row supplies a
      ``cache_read`` rate; otherwise the full ``input_tokens`` is priced
      at the input rate (a slight overcount, never negative).

    Always clamped to ``>= 0`` as a belt-and-suspenders guard.
    """
    input_rate = prices["input"]
    cost = usage.output_tokens / 1_000_000 * prices["output"]

    if "cache_write" in prices:  # Anthropic semantics: input_tokens is uncached
        cost += usage.input_tokens / 1_000_000 * input_rate
        cost += (
            usage.cache_creation_input_tokens
            / 1_000_000
            * prices.get("cache_write", input_rate * 1.25)
        )
        cost += (
            usage.cache_read_input_tokens
            / 1_000_000
            * prices.get("cache_read", input_rate * 0.1)
        )
    else:  # OpenAI / Google semantics: input_tokens already includes cache reads
        cache_read_rate = prices.get("cache_read")
        if cache_read_rate is not None and usage.cache_read_input_tokens:
            billable = max(0, usage.input_tokens - usage.cache_read_input_tokens)
            cost += billable / 1_000_000 * input_rate
            cost += usage.cache_read_input_tokens / 1_000_000 * cache_read_rate
        else:
            cost += usage.input_tokens / 1_000_000 * input_rate

    return max(0.0, cost)


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
        return _price_usage(usage, prices)

    def _get_prices(self, model: str) -> Optional[Dict[str, float]]:
        """Look up pricing, trying exact match then longest-prefix match."""
        return _lookup_prices(self._pricing, model)


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

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="custom_pricing",
            fields=[
                ConfigField(
                    name="input_per_million",
                    type="number",
                    label="Input ($ / 1M tokens)",
                    description="Flat input-token rate per million.",
                    default=3.0,
                    min_value=0,
                ),
                ConfigField(
                    name="output_per_million",
                    type="number",
                    label="Output ($ / 1M tokens)",
                    description="Flat output-token rate per million.",
                    default=15.0,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        v = config.get("input_per_million")
        if isinstance(v, (int, float)) and v >= 0:
            self._input_rate = float(v)
        v = config.get("output_per_million")
        if isinstance(v, (int, float)) and v >= 0:
            self._output_rate = float(v)

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_per_million": self._input_rate,
            "output_per_million": self._output_rate,
        }

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
        return _price_usage(usage, prices)

    def _get_prices(self, model: str) -> Optional[Dict[str, float]]:
        """Look up pricing: exact match → longest-prefix match."""
        return _lookup_prices(self._pricing, model)
