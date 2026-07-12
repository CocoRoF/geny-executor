"""Cache strategies — Level 2 strategies for prompt caching."""

from __future__ import annotations

from typing import Any, Dict, List

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s05_cache.interface import CacheStrategy, EPHEMERAL_CACHE

# CLI-style aliases hosts routinely store in ``state.model`` — the
# canonical ``claude-*`` id is resolved INSIDE the Anthropic client,
# i.e. after Stage 5 has already run. See ``_supports_cache_control``.
_CLAUDE_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku"})


def _supports_cache_control(state: PipelineState) -> bool:
    """Check if the current backend accepts Anthropic cache_control markers.

    Gate on the resolved client's ``provider`` — NOT the raw model
    string. Geny stores CLI-style aliases (``"opus"``/``"sonnet"``) in
    ``state.model`` and the canonical ``claude-*`` id only materializes
    inside the Anthropic client, after this stage. The pre-2.50
    ``state.model.startswith("claude-")`` gate therefore silently
    disabled ALL prompt caching for alias-configured sessions — full
    prefill of tools + system + history on every turn (TTFT audit
    2026-07-12, finding A1).

    Only ``provider == "anthropic"`` gets markers: OpenAI/Google/vLLM
    cache automatically on prefix and would reject the extension keys;
    the claude_code CLI owns its own caching.
    """
    client = state.llm_client
    provider = str(getattr(client, "provider", "") or "") if client is not None else ""
    if provider:
        return provider == "anthropic"
    # No client attached (unit tests, dry construction): fall back to the
    # model-string heuristic, alias-aware this time.
    model = state.model or ""
    return model.startswith("claude-") or model in _CLAUDE_MODEL_ALIASES


def _strip_stale_markers(state: PipelineState) -> None:
    """Remove cache_control markers left over from previous turns.

    ``state.messages`` (and the version-cached ``state.tools``) persist
    across turns, so a moving history breakpoint would ACCUMULATE one
    marker per turn — the Anthropic API rejects requests with more than
    4 cache_control blocks. Stripping before every re-apply keeps the
    request at exactly the breakpoints the strategy places this turn.
    (``state.system`` is rebuilt by Stage 3 each turn, but strip it too:
    a host may hand Stage 5 a block-list system it reuses.)
    """
    system = state.system
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                block.pop("cache_control", None)
    for tool in state.tools or []:
        if isinstance(tool, dict):
            tool.pop("cache_control", None)
    for msg in state.messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


def _cache_system(state: PipelineState) -> None:
    """Place the system breakpoint at the end of the STABLE region.

    When Stage 3 recorded a stable/volatile split
    (``state.shared['system_parts']``, volatile_placement="system"),
    the system string is re-emitted as two content blocks with the
    marker on the stable one — the volatile tail (clock, retrieved
    memory) stays outside the cached prefix, so a ticked minute no
    longer re-prefills the system prompt. The leading separator on the
    volatile block keeps the rendered text byte-identical to the
    original joined string.

    Without split info: previous behavior (whole string / last block).
    """
    system = state.system
    if not system:
        return

    if isinstance(system, str):
        parts = state.shared.get("system_parts")
        stable = parts.get("stable_text") if isinstance(parts, dict) else None
        volatile = parts.get("volatile_text") if isinstance(parts, dict) else None
        if stable and volatile and system == f"{stable}\n\n{volatile}":
            state.system = [
                {"type": "text", "text": stable, "cache_control": EPHEMERAL_CACHE},
                {"type": "text", "text": f"\n\n{volatile}"},
            ]
        else:
            state.system = [{"type": "text", "text": system, "cache_control": EPHEMERAL_CACHE}]
    elif isinstance(system, list) and system:
        last = system[-1]
        if isinstance(last, dict) and "cache_control" not in last:
            last["cache_control"] = EPHEMERAL_CACHE


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

    Converts system to content blocks with cache_control on the stable
    region (see ``_cache_system``). Only applies to Anthropic backends —
    other providers are bypassed.
    """

    @property
    def name(self) -> str:
        return "system_cache"

    @property
    def description(self) -> str:
        return "Cache system prompt"

    def apply_cache_markers(self, state: PipelineState) -> None:
        if not _supports_cache_control(state):
            return
        _strip_stale_markers(state)
        _cache_system(state)


class AggressiveCacheStrategy(CacheStrategy):
    """Cache tools + system + stable history prefix.

    Breakpoints (Anthropic allows 4; this places up to 3):
      1. End of the tools array — the largest, most stable fixed block
         (~10K tokens for a full built-in set); caching it independently
         means a system edit no longer re-prefills every tool schema.
      2. End of the STABLE system region (before the volatile tail).
      3. A moving point N messages from the end of history, so the
         conversation prefix re-caches incrementally as it grows.

    Stale markers from previous turns are stripped before re-applying —
    without that the moving history breakpoint accumulates one marker
    per turn and trips the API's 4-block limit.
    """

    def __init__(self, stable_history_offset: int = 4):
        self._stable_offset = stable_history_offset

    @property
    def name(self) -> str:
        return "aggressive_cache"

    @property
    def description(self) -> str:
        return "Cache tools + system + stable history"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="aggressive_cache",
            fields=[
                ConfigField(
                    name="stable_history_offset",
                    type="integer",
                    label="Stable history offset",
                    description="Place a cache breakpoint this many messages before the end of state.messages.",
                    default=4,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("stable_history_offset")
        if isinstance(n, int) and n >= 0:
            self._stable_offset = n

    def get_config(self) -> Dict[str, Any]:
        return {"stable_history_offset": self._stable_offset}

    def apply_cache_markers(self, state: PipelineState) -> None:
        if not _supports_cache_control(state):
            return

        _strip_stale_markers(state)

        # 1. Cache the tool schema block
        self._cache_tools(state)

        # 2. Cache the stable system region
        _cache_system(state)

        # 3. Cache stable history prefix
        self._cache_history_prefix(state)

    def _cache_tools(self, state: PipelineState) -> None:
        tools: List[Any] = state.tools or []
        if tools and isinstance(tools[-1], dict):
            tools[-1]["cache_control"] = EPHEMERAL_CACHE

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
