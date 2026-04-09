"""Cache strategies — Level 2 strategies for prompt caching."""

from __future__ import annotations

from geny_executor.core.state import PipelineState
from geny_executor.stages.s05_cache.interface import CacheStrategy, EPHEMERAL_CACHE


class NoCacheStrategy(CacheStrategy):
    """No caching — pass through unchanged."""

    @property
    def name(self) -> str:
        return "no_cache"

    @property
    def description(self) -> str:
        return "No prompt caching"

    def apply_cache_markers(self, state: PipelineState) -> None:
        pass


class SystemCacheStrategy(CacheStrategy):
    """Cache system prompt only.

    Converts system to content blocks with cache_control on the last block.
    """

    @property
    def name(self) -> str:
        return "system_cache"

    @property
    def description(self) -> str:
        return "Cache system prompt"

    def apply_cache_markers(self, state: PipelineState) -> None:
        system = state.system
        if not system:
            return

        if isinstance(system, str):
            # Convert to content blocks with cache marker
            state.system = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": EPHEMERAL_CACHE,
                }
            ]
        elif isinstance(system, list):
            # Add cache marker to last block
            if system:
                last = system[-1]
                if isinstance(last, dict) and "cache_control" not in last:
                    last["cache_control"] = EPHEMERAL_CACHE


class AggressiveCacheStrategy(CacheStrategy):
    """Cache system + tools + stable history prefix.

    Breakpoints:
      1. End of system prompt
      2. End of tools definition (via system blocks)
      3. Last stable history point (Nth message from end)
    """

    def __init__(self, stable_history_offset: int = 4):
        self._stable_offset = stable_history_offset

    @property
    def name(self) -> str:
        return "aggressive_cache"

    @property
    def description(self) -> str:
        return "Cache system + tools + stable history"

    def apply_cache_markers(self, state: PipelineState) -> None:
        # 1. Cache system prompt
        self._cache_system(state)

        # 2. Cache stable history prefix
        self._cache_history_prefix(state)

    def _cache_system(self, state: PipelineState) -> None:
        system = state.system
        if not system:
            return

        if isinstance(system, str):
            state.system = [{"type": "text", "text": system, "cache_control": EPHEMERAL_CACHE}]
        elif isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict):
                last["cache_control"] = EPHEMERAL_CACHE

    def _cache_history_prefix(self, state: PipelineState) -> None:
        msgs = state.messages
        if len(msgs) <= self._stable_offset:
            return

        # Mark the message at the stable boundary
        boundary_idx = len(msgs) - self._stable_offset - 1
        if boundary_idx < 0:
            return

        msg = msgs[boundary_idx]
        content = msg.get("content")

        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content, "cache_control": EPHEMERAL_CACHE}]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict) and "cache_control" not in last_block:
                last_block["cache_control"] = EPHEMERAL_CACHE
