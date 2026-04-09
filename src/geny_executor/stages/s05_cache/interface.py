"""Stage 5: Cache — interface definitions."""

from __future__ import annotations

from abc import abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState

EPHEMERAL_CACHE = {"type": "ephemeral"}


class CacheStrategy(Strategy):
    """Base interface for prompt caching decisions."""

    @abstractmethod
    def apply_cache_markers(self, state: PipelineState) -> None:
        """Insert cache_control markers into state.system and state.messages."""
