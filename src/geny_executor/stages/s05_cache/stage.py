"""Stage 5: Cache — applies prompt caching strategy."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s05_cache.strategies import CacheStrategy, NoCacheStrategy


class CacheStage(Stage[Any, Any]):
    """Stage 5: Cache.

    Dual abstraction:
      - Level 2 strategy: where to place cache breakpoints
    """

    def __init__(self, strategy: Optional[CacheStrategy] = None):
        self._strategy = strategy or NoCacheStrategy()

    @property
    def name(self) -> str:
        return "cache"

    @property
    def order(self) -> int:
        return 5

    @property
    def category(self) -> str:
        return "pre_flight"

    def should_bypass(self, state: PipelineState) -> bool:
        return isinstance(self._strategy, NoCacheStrategy)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        self._strategy.apply_cache_markers(state)

        state.add_event("cache.applied", {
            "strategy": type(self._strategy).__name__,
            "system_is_blocks": isinstance(state.system, list),
        })

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="strategy",
                current_impl=type(self._strategy).__name__,
                available_impls=[
                    "NoCacheStrategy",
                    "SystemCacheStrategy",
                    "AggressiveCacheStrategy",
                ],
            ),
        ]
