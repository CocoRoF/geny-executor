"""Retry strategies — concrete implementations for API error recovery."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from geny_executor.core.errors import ErrorCategory
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.stages.s06_api.interface import RetryStrategy


def _require_int(strategy: str, key: str, value: Any, *, minimum: int = 0) -> int:
    """configure() validation: integer >= minimum, bools rejected explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{strategy}: {key!r} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{strategy}: {key!r} must be >= {minimum}, got {value!r}")
    return value


def _require_number(strategy: str, key: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{strategy}: {key!r} must be a number >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{strategy}: {key!r} must be >= {minimum}, got {value!r}")
    return float(value)


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff with jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter

    @property
    def name(self) -> str:
        return "exponential_backoff"

    @property
    def description(self) -> str:
        return f"Exponential backoff (max {self._max_retries} retries)"

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="exponential_backoff",
            fields=[
                ConfigField(
                    name="max_retries",
                    type="integer",
                    label="Max retries",
                    description="Give up after this many retry attempts.",
                    default=3,
                    min_value=0,
                ),
                ConfigField(
                    name="base_delay",
                    type="number",
                    label="Base delay (s)",
                    description="First retry delay; doubles each attempt.",
                    default=1.0,
                    min_value=0,
                ),
                ConfigField(
                    name="max_delay",
                    type="number",
                    label="Max delay (s)",
                    description="Upper bound for any single backoff delay.",
                    default=60.0,
                    min_value=0,
                ),
                ConfigField(
                    name="jitter",
                    type="number",
                    label="Jitter ratio",
                    description="Random +/- fraction applied to each delay.",
                    default=0.1,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        # Validate everything before applying anything: a rejected
        # configure must leave the previous (working) config live.
        retries = self._max_retries
        base = self._base_delay
        max_d = self._max_delay
        jitter = self._jitter
        if "max_retries" in config:
            retries = _require_int("exponential_backoff", "max_retries", config["max_retries"])
        if "base_delay" in config:
            base = _require_number("exponential_backoff", "base_delay", config["base_delay"])
        if "max_delay" in config:
            max_d = _require_number("exponential_backoff", "max_delay", config["max_delay"])
        # Cross-field check on the merged result so a partial update can't
        # invert the pair (max_delay < base_delay would clamp every delay
        # to max_delay — silent loss of the backoff curve).
        if max_d < base:
            raise ValueError(
                f"exponential_backoff: 'max_delay' must be >= 'base_delay' "
                f"(got {max_d} < {base})"
            )
        if "jitter" in config:
            jitter = _require_number("exponential_backoff", "jitter", config["jitter"])
            if jitter > 1.0:
                raise ValueError(
                    f"exponential_backoff: 'jitter' must be in [0.0, 1.0], got {jitter!r}"
                )
        self._max_retries = retries
        self._base_delay = base
        self._max_delay = max_d
        self._jitter = jitter

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_retries": self._max_retries,
            "base_delay": self._base_delay,
            "max_delay": self._max_delay,
            "jitter": self._jitter,
        }

    def should_retry(self, category: ErrorCategory, attempt: int) -> bool:
        if attempt >= self._max_retries:
            return False
        return category.is_recoverable

    def get_delay(self, attempt: int) -> float:
        delay = min(self._base_delay * (2**attempt), self._max_delay)
        jitter_amount = delay * self._jitter
        delay += random.uniform(-jitter_amount, jitter_amount)
        return max(0, delay)


class NoRetry(RetryStrategy):
    """No retry — fail immediately."""

    @property
    def name(self) -> str:
        return "no_retry"

    @property
    def description(self) -> str:
        return "No retry, fail immediately"

    def should_retry(self, category: ErrorCategory, attempt: int) -> bool:
        return False

    def get_delay(self, attempt: int) -> float:
        return 0.0


class RateLimitAwareRetry(RetryStrategy):
    """Respects rate limit headers."""

    def __init__(self, max_retries: int = 5, fallback_delay: float = 5.0):
        self._max_retries = max_retries
        self._fallback_delay = fallback_delay
        self._retry_after: Optional[float] = None

    @property
    def name(self) -> str:
        return "rate_limit_aware"

    @property
    def description(self) -> str:
        return "Respects rate limit retry-after headers"

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="rate_limit_aware",
            fields=[
                ConfigField(
                    name="max_retries",
                    type="integer",
                    label="Max retries",
                    description="Give up after this many retry attempts.",
                    default=5,
                    min_value=0,
                ),
                ConfigField(
                    name="fallback_delay",
                    type="number",
                    label="Fallback delay (s)",
                    description="Delay used when no retry-after header was captured.",
                    default=5.0,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        retries = self._max_retries
        fallback = self._fallback_delay
        if "max_retries" in config:
            retries = _require_int("rate_limit_aware", "max_retries", config["max_retries"])
        if "fallback_delay" in config:
            fallback = _require_number(
                "rate_limit_aware", "fallback_delay", config["fallback_delay"]
            )
        self._max_retries = retries
        self._fallback_delay = fallback

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_retries": self._max_retries,
            "fallback_delay": self._fallback_delay,
        }

    def set_retry_after(self, seconds: float) -> None:
        """Set the retry-after delay from headers."""
        self._retry_after = seconds

    def should_retry(self, category: ErrorCategory, attempt: int) -> bool:
        if attempt >= self._max_retries:
            return False
        return category in {
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.SERVER_ERROR,
        }

    def get_delay(self, attempt: int) -> float:
        if self._retry_after is not None:
            delay = self._retry_after
            self._retry_after = None
            return delay
        return self._fallback_delay
