"""2.2.0 Wave 1 — s05 cache_prefix wired into cache key construction.

Audit "validated-but-inert" table: ``cache_prefix`` was accepted by the
stage schema, serialized, restored — and read by nothing. The wiring
derives a content-addressed key for the cached prefix (Anthropic's
prompt cache takes no wire-level key, so the prefix namespaces the
host-side accounting key rather than mutating the prompt) and publishes
it on ``state.shared['cache_key']`` + the ``cache.applied`` event.
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s05_cache.artifact.default.stage import CacheStage
from geny_executor.stages.s05_cache.artifact.default.strategies import SystemCacheStrategy


def _state(system: str = "You are a helpful assistant.") -> PipelineState:
    state = PipelineState(session_id="s")
    state.model = "claude-sonnet-4-6"
    state.system = system
    return state


def _applied_event(state: PipelineState) -> dict:
    events = [e for e in state.events if e.get("type") == "cache.applied"]
    assert len(events) == 1
    return events[0]["data"]


class TestCachePrefixInKey:
    @pytest.mark.asyncio
    async def test_prefix_lands_in_shared_cache_key(self):
        stage = CacheStage(strategy=SystemCacheStrategy(), cache_prefix="tenant-a")
        state = _state()

        await stage.execute("in", state)

        key = state.shared["cache_key"]
        assert key.startswith("tenant-a:")
        assert _applied_event(state)["cache_key"] == key

    @pytest.mark.asyncio
    async def test_no_prefix_yields_bare_digest(self):
        stage = CacheStage(strategy=SystemCacheStrategy())
        state = _state()

        await stage.execute("in", state)

        assert ":" not in state.shared["cache_key"]

    @pytest.mark.asyncio
    async def test_same_content_different_prefix_different_key(self):
        """The whole point of the knob: namespace isolation. Two sessions
        with identical prompts but different prefixes must not share a key."""
        state_a, state_b = _state(), _state()
        await CacheStage(strategy=SystemCacheStrategy(), cache_prefix="a").execute("x", state_a)
        await CacheStage(strategy=SystemCacheStrategy(), cache_prefix="b").execute("x", state_b)

        assert state_a.shared["cache_key"] != state_b.shared["cache_key"]
        # ...while the content digest itself matches.
        assert state_a.shared["cache_key"].split(":", 1)[1] == (
            state_b.shared["cache_key"].split(":", 1)[1]
        )

    @pytest.mark.asyncio
    async def test_same_prefix_same_content_stable_key(self):
        state_a, state_b = _state(), _state()
        stage = CacheStage(strategy=SystemCacheStrategy(), cache_prefix="p")
        await stage.execute("x", state_a)
        await stage.execute("x", state_b)

        assert state_a.shared["cache_key"] == state_b.shared["cache_key"]

    @pytest.mark.asyncio
    async def test_update_config_changes_prefix(self):
        """The manifest path: cache_prefix arrives via update_config."""
        stage = CacheStage(strategy=SystemCacheStrategy())
        stage.update_config({"cache_prefix": "from-manifest"})
        state = _state()

        await stage.execute("in", state)

        assert state.shared["cache_key"].startswith("from-manifest:")

    @pytest.mark.asyncio
    async def test_prefix_does_not_touch_the_prompt(self):
        """A caching knob must never change what the model sees — the
        prefix lives only in the derived key."""
        state = _state(system="stable prompt")
        await CacheStage(strategy=SystemCacheStrategy(), cache_prefix="tenant-a").execute(
            "in", state
        )

        assert isinstance(state.system, list)
        assert all("tenant-a" not in str(block.get("text", "")) for block in state.system)
