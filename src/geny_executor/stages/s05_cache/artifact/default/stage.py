"""Stage 5: Cache — applies prompt caching strategy."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s05_cache.interface import CacheStrategy
from geny_executor.stages.s05_cache.artifact.default.strategies import (
    AggressiveCacheStrategy,
    NoCacheStrategy,
    SystemCacheStrategy,
)


class CacheStage(Stage[Any, Any]):
    """Stage 5: Cache.

    Dual abstraction:
      - Level 2 strategy: where to place cache breakpoints
    """

    def __init__(
        self,
        strategy: Optional[CacheStrategy] = None,
        *,
        cache_prefix: str = "",
    ):
        self._slots: Dict[str, StrategySlot] = {
            "strategy": StrategySlot(
                name="strategy",
                strategy=strategy or NoCacheStrategy(),
                registry={
                    "no_cache": NoCacheStrategy,
                    "system_cache": SystemCacheStrategy,
                    "aggressive_cache": AggressiveCacheStrategy,
                },
                description="Prompt caching strategy",
            ),
        }
        self._cache_prefix = str(cache_prefix)

    @property
    def _strategy(self) -> CacheStrategy:
        return self._slots["strategy"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "cache"

    @property
    def order(self) -> int:
        return 5

    @property
    def category(self) -> str:
        return "pre_flight"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="cache",
            fields=[
                ConfigField(
                    name="cache_prefix",
                    type="string",
                    label="Cache Prefix",
                    description="Prefix prepended to cache keys for namespace isolation.",
                    default="",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {"cache_prefix": self._cache_prefix}

    def update_config(self, config: Dict[str, Any]) -> None:
        if "cache_prefix" in config:
            self._cache_prefix = str(config["cache_prefix"])

    def should_bypass(self, state: PipelineState) -> bool:
        return isinstance(self._strategy, NoCacheStrategy)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        self._strategy.apply_cache_markers(state)

        state.add_event(
            "cache.applied",
            {
                "strategy": type(self._strategy).__name__,
                "system_is_blocks": isinstance(state.system, list),
            },
        )

        return input
