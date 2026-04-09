"""Cache strategies — backward-compatible re-exports.

Concrete implementations have moved to:
  geny_executor.stages.s05_cache.artifact.default.strategies

ABCs and constants live in:
  geny_executor.stages.s05_cache.interface
"""

from geny_executor.stages.s05_cache.interface import CacheStrategy, EPHEMERAL_CACHE
from geny_executor.stages.s05_cache.artifact.default.strategies import (
    NoCacheStrategy,
    SystemCacheStrategy,
    AggressiveCacheStrategy,
)

__all__ = [
    "EPHEMERAL_CACHE",
    "CacheStrategy",
    "NoCacheStrategy",
    "SystemCacheStrategy",
    "AggressiveCacheStrategy",
]
