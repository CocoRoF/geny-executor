"""Tests for parallel sub-agent orchestration (Phase D2)."""

from __future__ import annotations

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.subagent_type import (
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)


# ---------------------------------------------------------------------------
# Test pipelines
# ---------------------------------------------------------------------------


class _SleepPipeline:
    """Awaits ``delay`` seconds before returning ``payload``."""

    def __init__(self, delay: float, payload: str) -> None:
        self._delay = delay
        self._payload = payload

    async def run(self, task, sub_state):
        await asyncio.sleep(self._delay)
        return type(
            "R", (), {"success": True, "text": f"{self._payload}:{task}", "error": None}
        )()


class _CountingPipeline:
    """Tracks how many instances are concurrently mid-run."""

    in_flight = 0
    peak = 0
    _lock = asyncio.Lock()

    @classmethod
    def reset(cls) -> None:
        cls.in_flight = 0
        cls.peak = 0

    async def run(self, task, sub_state):
        async with self._lock:
            type(self).in_flight += 1
            type(self).peak = max(type(self).peak, type(self).in_flight)
        await asyncio.sleep(0.05)
        async with self._lock:
            type(self).in_flight -= 1
        return type("R", (), {"success": True, "text": "ok", "error": None})()


# ---------------------------------------------------------------------------
# Parallel + serial mix
# ---------------------------------------------------------------------------


def _registry_with_two_parallel_and_one_serial() -> SubagentTypeRegistry:
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="research",
        factory=lambda ctx: _SleepPipeline(0.10, "research"),
        parallel=True,
        max_concurrent=4,
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="summarize",
        factory=lambda ctx: _SleepPipeline(0.10, "summarize"),
        parallel=True,
        max_concurrent=4,
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="critic",
        factory=lambda ctx: _SleepPipeline(0.05, "critic"),
        parallel=False,
    ))
    return reg


@pytest.mark.asyncio
async def test_parallel_pair_finishes_within_serial_budget() -> None:
    """Two parallel pipelines of 100ms each plus one serial 50ms should
    finish in under ~250ms total wall time (well under 100+100+50=250ms
    if they were all serial, but more importantly under ~200ms if
    parallel is truly parallel)."""
    reg = _registry_with_two_parallel_and_one_serial()
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "critic", "task": "c"},        # serial first
        {"agent_type": "research", "task": "r"},      # parallel group
        {"agent_type": "summarize", "task": "s"},     # parallel group
    ]
    t0 = time.monotonic()
    result = await orch.orchestrate(state)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    assert result.delegated is True
    assert len(result.sub_results) == 3
    # Serial first, then parallel (in registry/input order).
    assert result.sub_results[0]["agent_type"] == "critic"
    assert {r["agent_type"] for r in result.sub_results[1:]} == {"research", "summarize"}
    # Wall time: 50ms (serial) + ~100ms (parallel pair) < 200ms.
    assert elapsed_ms < 200, f"wall time {elapsed_ms}ms too high — parallel fan-out broken?"


@pytest.mark.asyncio
async def test_parallel_fan_out_respects_max_concurrent() -> None:
    """When max_concurrent caps the semaphore at 2 but we send 4
    parallel requests, only 2 run simultaneously."""
    _CountingPipeline.reset()

    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="worker",
        factory=lambda ctx: _CountingPipeline(),
        parallel=True,
        max_concurrent=2,
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "worker", "task": f"t{i}"} for i in range(4)
    ]
    await orch.orchestrate(state)
    assert _CountingPipeline.peak == 2, f"peak={_CountingPipeline.peak} (expected 2)"


@pytest.mark.asyncio
async def test_parallel_group_uses_min_max_concurrent() -> None:
    """When the parallel group mixes max_concurrent=4 and =2, the
    semaphore caps at the minimum (=2)."""
    _CountingPipeline.reset()
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="loose",
        factory=lambda ctx: _CountingPipeline(),
        parallel=True,
        max_concurrent=4,
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="tight",
        factory=lambda ctx: _CountingPipeline(),
        parallel=True,
        max_concurrent=2,
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "loose", "task": "a"},
        {"agent_type": "loose", "task": "b"},
        {"agent_type": "tight", "task": "c"},
        {"agent_type": "tight", "task": "d"},
    ]
    await orch.orchestrate(state)
    assert _CountingPipeline.peak <= 2


@pytest.mark.asyncio
async def test_serial_preserves_input_order() -> None:
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="a", factory=lambda ctx: _SleepPipeline(0, "a"),
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="b", factory=lambda ctx: _SleepPipeline(0, "b"),
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="c", factory=lambda ctx: _SleepPipeline(0, "c"),
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "b", "task": "B"},
        {"agent_type": "a", "task": "A"},
        {"agent_type": "c", "task": "C"},
    ]
    result = await orch.orchestrate(state)
    assert [r["agent_type"] for r in result.sub_results] == ["b", "a", "c"]


@pytest.mark.asyncio
async def test_parallel_failure_isolated_from_siblings() -> None:
    """A factory raise on one parallel sub-agent doesn't crash the
    rest — each result lands as success or structured failure."""
    def good_factory(ctx):
        return _SleepPipeline(0.01, "good")

    def bad_factory(ctx):
        raise RuntimeError("boom")

    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="good", factory=good_factory, parallel=True,
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="bad", factory=bad_factory, parallel=True,
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "good", "task": "g"},
        {"agent_type": "bad",  "task": "b"},
        {"agent_type": "good", "task": "g2"},
    ]
    result = await orch.orchestrate(state)
    by_type = {r["agent_type"]: r for r in result.sub_results}
    assert by_type["good"]["success"] is True or any(
        r["agent_type"] == "good" and r["success"] for r in result.sub_results
    )
    bad = next(r for r in result.sub_results if r["agent_type"] == "bad")
    assert bad["success"] is False
    assert "boom" in (bad.get("error") or "")


@pytest.mark.asyncio
async def test_empty_delegate_requests_no_op() -> None:
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="x", factory=lambda ctx: _SleepPipeline(0, "x"), parallel=True,
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = []
    result = await orch.orchestrate(state)
    assert result.delegated is False


@pytest.mark.asyncio
async def test_all_serial_path_unchanged_from_d1() -> None:
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="a", factory=lambda ctx: _SleepPipeline(0, "a"), parallel=False,
    ))
    reg.register(SubagentTypeDescriptor(
        agent_type="b", factory=lambda ctx: _SleepPipeline(0, "b"), parallel=False,
    ))
    orch = SubagentTypeOrchestrator(reg)
    state = PipelineState(session_id="s")
    state.delegate_requests = [
        {"agent_type": "a", "task": "A"},
        {"agent_type": "b", "task": "B"},
    ]
    result = await orch.orchestrate(state)
    assert [r["agent_type"] for r in result.sub_results] == ["a", "b"]
