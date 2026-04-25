"""Unit tests for the Stage 14 OrderedEmitterChain (S7.11)."""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_emit import (
    EmitResult,
    Emitter,
    OrderedEmitterChain,
)


# ── helpers ───────────────────────────────────────────────────────────────


class StubEmitter(Emitter):
    """Test emitter — records calls, lets the test choose the result."""

    def __init__(
        self,
        name: str,
        *,
        requires: Tuple[str, ...] = (),
        timeout_seconds: Optional[float] = None,
        result: Optional[EmitResult] = None,
        sleep_seconds: float = 0.0,
        raises: Optional[BaseException] = None,
    ) -> None:
        self._name = name
        self.requires = tuple(requires)
        self.timeout_seconds = timeout_seconds
        self._result = result or EmitResult(emitted=True, channels=[name])
        self._sleep = sleep_seconds
        self._raises = raises
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def emit(self, state: PipelineState) -> EmitResult:
        self.calls += 1
        if self._sleep > 0:
            await asyncio.sleep(self._sleep)
        if self._raises is not None:
            raise self._raises
        return self._result


def _events(state: PipelineState, type_: str) -> List[dict]:
    return [e for e in state.events if e["type"] == type_]


# ── topological ordering ────────────────────────────────────────────────


class TestTopologicalOrder:
    @pytest.mark.asyncio
    async def test_no_deps_runs_in_declared_order(self):
        a = StubEmitter("a")
        b = StubEmitter("b")
        c = StubEmitter("c")
        chain = OrderedEmitterChain([a, b, c])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert [r.emitter_name for r in results] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_simple_chain_dependency(self):
        # tts depends on text — even though declared first, text should run first.
        text = StubEmitter("text")
        tts = StubEmitter("tts", requires=("text",))
        chain = OrderedEmitterChain([tts, text])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert [r.emitter_name for r in results] == ["text", "tts"]

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        a = StubEmitter("a")
        b = StubEmitter("b", requires=("a",))
        c = StubEmitter("c", requires=("a",))
        d = StubEmitter("d", requires=("b", "c"))
        chain = OrderedEmitterChain([d, c, b, a])
        state = PipelineState()
        results = await chain.emit_all(state)
        order = [r.emitter_name for r in results]
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}

    @pytest.mark.asyncio
    async def test_unknown_dependency_runs_anyway_with_event(self):
        em = StubEmitter("solo", requires=("ghost",))
        chain = OrderedEmitterChain([em])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert [r.emitter_name for r in results] == ["solo"]
        assert results[0].emitted is True
        warnings = _events(state, "emit.unknown_dependency")
        assert len(warnings) == 1
        assert warnings[0]["data"]["dependency"] == "ghost"

    @pytest.mark.asyncio
    async def test_cycle_falls_back_to_declared_order_with_event(self):
        # a → b → a cycle
        a = StubEmitter("a", requires=("b",))
        b = StubEmitter("b", requires=("a",))
        chain = OrderedEmitterChain([a, b])
        state = PipelineState()
        results = await chain.emit_all(state)
        # declared order preserved
        assert [r.emitter_name for r in results] == ["a", "b"]
        cyc = _events(state, "emit.cycle_detected")
        assert len(cyc) == 1
        assert cyc[0]["data"]["total"] == 2


# ── failure isolation + dep-failure skip ───────────────────────────────


class TestDepFailureSkip:
    @pytest.mark.asyncio
    async def test_dependent_skipped_when_required_raises(self):
        text = StubEmitter("text", raises=RuntimeError("boom"))
        tts = StubEmitter("tts", requires=("text",))
        chain = OrderedEmitterChain([text, tts])
        state = PipelineState()
        results = await chain.emit_all(state)

        text_r, tts_r = results
        assert text_r.emitted is False
        assert "boom" in text_r.metadata["error"]
        assert tts_r.emitted is False
        assert tts_r.metadata["skipped"] == "dep_failed"
        assert tts_r.metadata["deps"] == ["text"]
        assert tts.calls == 0

        evts = _events(state, "emit.skipped_dep_failed")
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_dependent_runs_when_required_succeeds(self):
        text = StubEmitter("text")
        tts = StubEmitter("tts", requires=("text",))
        chain = OrderedEmitterChain([text, tts])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert all(r.emitted for r in results)
        assert tts.calls == 1

    @pytest.mark.asyncio
    async def test_independent_emitter_runs_when_sibling_fails(self):
        bad = StubEmitter("bad", raises=RuntimeError("x"))
        ok = StubEmitter("ok")  # no deps
        chain = OrderedEmitterChain([bad, ok])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert results[0].emitted is False
        assert results[1].emitted is True


# ── timeout & backpressure ──────────────────────────────────────────────


class TestTimeoutAndBackpressure:
    @pytest.mark.asyncio
    async def test_timeout_increments_counter_and_emits_event(self):
        slow = StubEmitter("slow", timeout_seconds=0.05, sleep_seconds=0.5)
        chain = OrderedEmitterChain([slow])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert results[0].emitted is False
        assert results[0].metadata["error"] == "timeout"
        assert chain.consecutive_timeouts["slow"] == 1
        evts = _events(state, "emit.timeout")
        assert len(evts) == 1
        assert evts[0]["data"]["consecutive_timeouts"] == 1

    @pytest.mark.asyncio
    async def test_consecutive_timeouts_skip_after_threshold(self):
        slow = StubEmitter("slow", timeout_seconds=0.01, sleep_seconds=0.2)
        chain = OrderedEmitterChain([slow], backpressure_threshold=2)
        state = PipelineState()

        # First two passes time out and increment.
        await chain.emit_all(state)
        await chain.emit_all(state)
        assert chain.consecutive_timeouts["slow"] == 2

        # Third pass: counter is at threshold → skipped without invoking emit.
        prior_calls = slow.calls
        results = await chain.emit_all(state)
        assert slow.calls == prior_calls  # didn't actually call emit
        assert results[0].metadata["skipped"] == "backpressure"
        skips = _events(state, "emit.skipped_backpressure")
        assert len(skips) == 1

    @pytest.mark.asyncio
    async def test_success_resets_timeout_counter(self):
        # First call times out, second succeeds — counter should reset.
        controllable = StubEmitter("c", timeout_seconds=0.05, sleep_seconds=0.5)
        chain = OrderedEmitterChain([controllable])
        state = PipelineState()
        await chain.emit_all(state)
        assert chain.consecutive_timeouts["c"] == 1

        # Make the next call fast.
        controllable._sleep = 0.0
        await chain.emit_all(state)
        assert "c" not in chain.consecutive_timeouts

    @pytest.mark.asyncio
    async def test_reset_backpressure_clears_counter(self):
        slow = StubEmitter("slow", timeout_seconds=0.01, sleep_seconds=0.2)
        chain = OrderedEmitterChain([slow], backpressure_threshold=10)
        state = PipelineState()
        await chain.emit_all(state)
        assert chain.consecutive_timeouts["slow"] == 1

        chain.reset_backpressure("slow")
        assert "slow" not in chain.consecutive_timeouts

    @pytest.mark.asyncio
    async def test_reset_backpressure_all(self):
        a = StubEmitter("a", timeout_seconds=0.01, sleep_seconds=0.2)
        b = StubEmitter("b", timeout_seconds=0.01, sleep_seconds=0.2)
        chain = OrderedEmitterChain([a, b], backpressure_threshold=10)
        state = PipelineState()
        await chain.emit_all(state)
        assert set(chain.consecutive_timeouts) == {"a", "b"}

        chain.reset_backpressure()
        assert chain.consecutive_timeouts == {}

    @pytest.mark.asyncio
    async def test_non_timeout_exceptions_do_not_count(self):
        bad = StubEmitter("bad", raises=ValueError("x"))
        chain = OrderedEmitterChain([bad], backpressure_threshold=2)
        state = PipelineState()
        await chain.emit_all(state)
        await chain.emit_all(state)
        await chain.emit_all(state)
        assert chain.consecutive_timeouts == {}


# ── construction / API ─────────────────────────────────────────────────


class TestApiSurface:
    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            OrderedEmitterChain(backpressure_threshold=0)

    def test_add_appends(self):
        chain = OrderedEmitterChain()
        chain.add(StubEmitter("a"))
        chain.add(StubEmitter("b"))
        assert [e.name for e in chain.emitters] == ["a", "b"]

    def test_emitters_returns_copy(self):
        em = StubEmitter("a")
        chain = OrderedEmitterChain([em])
        listed = chain.emitters
        listed.append(StubEmitter("b"))
        # Internal list unchanged.
        assert [e.name for e in chain.emitters] == ["a"]

    @pytest.mark.asyncio
    async def test_emitter_name_populated_on_success(self):
        em = StubEmitter("foo", result=EmitResult(emitted=True, channels=["foo"]))
        chain = OrderedEmitterChain([em])
        state = PipelineState()
        results = await chain.emit_all(state)
        assert results[0].emitter_name == "foo"


# ── EmitResult schema ──────────────────────────────────────────────────


class TestEmitResultBackcompat:
    def test_default_emitter_name_blank(self):
        r = EmitResult()
        assert r.emitter_name == ""
        assert r.emitted is True

    def test_explicit_emitter_name(self):
        r = EmitResult(emitter_name="foo")
        assert r.emitter_name == "foo"
