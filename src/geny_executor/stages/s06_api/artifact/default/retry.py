"""Retry strategies — concrete implementations for API error recovery."""

from __future__ import annotations

import random
from typing import Optional

from geny_executor.core.errors import ErrorCategory
from geny_executor.stages.s06_api.interface import RetryStrategy


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
