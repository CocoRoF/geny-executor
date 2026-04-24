"""StreamingToolExecutor — Phase 2 Week 4 Checkpoint 1 tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from geny_executor.stages.s10_tool import StreamingToolExecutor
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.tools.base import (
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
)
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Fake tools
# ─────────────────────────────────────────────────────────────────


class _TimedTool(Tool):
    """Same shape as the partition executor fixture."""

    def __init__(self, name: str, *, concurrency_safe: bool, sleep_ms: int = 40):
        self._name = name
        self._safe = concurrency_safe
        self._sleep = sleep_ms / 1000.0
        self.started_at: List[float] = []
        self.finished_at: List[float] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Timed ({self._sleep}s, safe={self._safe})"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=self._safe, read_only=self._safe)

    async def execute(self, input, context):
        self.started_at.append(time.monotonic())
        await asyncio.sleep(self._sleep)
        self.finished_at.append(time.monotonic())
        return ToolResult(content=f"{self._name}:done")


def _make_registry(tools: List[Tool]) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _call(name: str, idx: int) -> Dict[str, Any]:
    return {"tool_use_id": f"tu_{idx}", "tool_name": name, "tool_input": {"i": idx}}


# ─────────────────────────────────────────────────────────────────
# Basic correctness
# ─────────────────────────────────────────────────────────────────


class TestStreamingExecutorBasics:
    def test_all_safe_preserves_receive_order(self) -> None:
        async def _run() -> List[Dict[str, Any]]:
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=40)
            b = _TimedTool("B", concurrency_safe=True, sleep_ms=40)
            c = _TimedTool("C", concurrency_safe=True, sleep_ms=40)
            reg = _make_registry([a, b, c])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            # Add calls incrementally (simulating streamed arrival)
            for i, name in enumerate(["A", "B", "C"]):
                await ex.add(_call(name, i), ctx)
                await asyncio.sleep(0)  # yield to let safe tasks start

            return await ex.drain(ctx)

        results = asyncio.run(_run())
        assert len(results) == 3
        assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_1", "tu_2"]

    def test_all_safe_runs_in_parallel(self) -> None:
        async def _run():
            tools = [
                _TimedTool(f"T{i}", concurrency_safe=True, sleep_ms=60)
                for i in range(3)
            ]
            reg = _make_registry(tools)
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            for i, t in enumerate(tools):
                await ex.add(_call(t.name, i), ctx)

            t0 = time.monotonic()
            await ex.drain(ctx)
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        # Three tools × 60ms serially would be 180ms. Parallel should
        # take ~60ms. Allow some slack for scheduling.
        assert elapsed < 0.15, f"expected parallel execution, got {elapsed:.3f}s"

    def test_unsafe_blocks_subsequent_calls(self) -> None:
        async def _run():
            safe_tool = _TimedTool("S", concurrency_safe=True, sleep_ms=30)
            unsafe_tool = _TimedTool("U", concurrency_safe=False, sleep_ms=40)
            reg = _make_registry([safe_tool, unsafe_tool])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            await ex.add(_call("U", 0), ctx)  # unsafe first
            await ex.add(_call("S", 1), ctx)  # safe queued
            await ex.add(_call("S", 2), ctx)  # safe queued

            results = await ex.drain(ctx)
            return results, unsafe_tool, safe_tool

        results, unsafe_tool, safe_tool = asyncio.run(_run())

        assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_1", "tu_2"]
        # Safe tools started only after the unsafe one finished.
        unsafe_done = unsafe_tool.finished_at[0]
        for start in safe_tool.started_at:
            assert start >= unsafe_done - 0.005

    def test_unsafe_after_safe_serializes(self) -> None:
        """Unsafe added after safe should wait for the in-flight safe batch."""

        async def _run():
            safe_tool = _TimedTool("S", concurrency_safe=True, sleep_ms=50)
            unsafe_tool = _TimedTool("U", concurrency_safe=False, sleep_ms=20)
            reg = _make_registry([safe_tool, unsafe_tool])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            await ex.add(_call("S", 0), ctx)
            await ex.add(_call("S", 1), ctx)
            # yield so the safe tasks actually start before we add unsafe
            await asyncio.sleep(0)
            await ex.add(_call("U", 2), ctx)

            results = await ex.drain(ctx)
            return results, unsafe_tool, safe_tool

        results, unsafe_tool, safe_tool = asyncio.run(_run())
        assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_1", "tu_2"]
        # Unsafe started only after the two safe tasks finished.
        last_safe_done = max(safe_tool.finished_at)
        assert unsafe_tool.started_at[0] >= last_safe_done - 0.005


# ─────────────────────────────────────────────────────────────────
# Bounded parallelism
# ─────────────────────────────────────────────────────────────────


class TestBoundedParallelism:
    def test_max_concurrency_limit(self) -> None:
        async def _run():
            tools = [
                _TimedTool(f"T{i}", concurrency_safe=True, sleep_ms=80)
                for i in range(4)
            ]
            reg = _make_registry(tools)
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(
                registry=reg, router=router, max_concurrency=2
            )
            ctx = ToolContext()

            for i, t in enumerate(tools):
                await ex.add(_call(t.name, i), ctx)

            t0 = time.monotonic()
            await ex.drain(ctx)
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        # With concurrency=2 and 4 tasks of 80ms, expected ≈ 160ms.
        assert 0.12 <= elapsed <= 0.30, (
            f"bounded parallelism expected (0.12–0.30s), got {elapsed:.3f}s"
        )


# ─────────────────────────────────────────────────────────────────
# Fail-closed on missing metadata
# ─────────────────────────────────────────────────────────────────


class _BrokenCapsTool(Tool):
    @property
    def name(self) -> str:
        return "Broken"

    @property
    def description(self) -> str:
        return "capabilities() raises"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input):
        raise RuntimeError("intentional")

    async def execute(self, input, context):
        return ToolResult(content="broken:done")


class TestFailClosed:
    def test_broken_capabilities_treated_as_unsafe(self) -> None:
        async def _run():
            broken = _BrokenCapsTool()
            reg = _make_registry([broken])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()
            await ex.add(_call("Broken", 0), ctx)
            return await ex.drain(ctx)

        results = asyncio.run(_run())
        assert len(results) == 1
        assert results[0]["tool_use_id"] == "tu_0"

    def test_unknown_tool_routes_as_error_but_serialized(self) -> None:
        async def _run():
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=10)
            reg = _make_registry([a])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()
            await ex.add(_call("A", 0), ctx)
            await ex.add(_call("Unknown", 1), ctx)
            return await ex.drain(ctx)

        results = asyncio.run(_run())
        assert len(results) == 2
        assert results[0]["tool_use_id"] == "tu_0"
        assert results[1]["tool_use_id"] == "tu_1"
        assert results[1].get("is_error") is True


# ─────────────────────────────────────────────────────────────────
# Event callbacks
# ─────────────────────────────────────────────────────────────────


class TestEventCallbacks:
    def test_events_fire_per_call(self) -> None:
        events: List[tuple[str, Dict[str, Any]]] = []

        async def _run():
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=10)
            b = _TimedTool("B", concurrency_safe=False, sleep_ms=10)
            reg = _make_registry([a, b])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            await ex.add(_call("A", 0), ctx)
            await ex.add(_call("B", 1), ctx)
            await ex.drain(ctx, on_event=lambda t, p: events.append((t, p)))

        asyncio.run(_run())

        starts = [e for e in events if e[0] == "tool.call_start"]
        completes = [e for e in events if e[0] == "tool.call_complete"]
        assert len(starts) == 2
        assert len(completes) == 2


# ─────────────────────────────────────────────────────────────────
# API discipline
# ─────────────────────────────────────────────────────────────────


class TestApiDiscipline:
    def test_add_after_drain_raises(self) -> None:
        async def _run():
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=5)
            reg = _make_registry([a])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()
            await ex.add(_call("A", 0), ctx)
            await ex.drain(ctx)
            # Second add must raise
            with pytest.raises(RuntimeError, match="after drain"):
                await ex.add(_call("A", 1), ctx)

        asyncio.run(_run())

    def test_missing_tool_use_id_raises(self) -> None:
        async def _run():
            ex = StreamingToolExecutor()
            with pytest.raises(ValueError, match="tool_use_id"):
                await ex.add({"tool_name": "X"}, ToolContext())

        asyncio.run(_run())

    def test_duplicate_tool_use_id_raises(self) -> None:
        async def _run():
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=5)
            reg = _make_registry([a])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()
            await ex.add(_call("A", 0), ctx)
            with pytest.raises(ValueError, match="duplicate"):
                await ex.add(_call("A", 0), ctx)  # same id

        asyncio.run(_run())

    def test_pending_count_tracks_inflight_and_queued(self) -> None:
        async def _run():
            safe_tool = _TimedTool("S", concurrency_safe=True, sleep_ms=80)
            unsafe_tool = _TimedTool("U", concurrency_safe=False, sleep_ms=20)
            reg = _make_registry([safe_tool, unsafe_tool])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            await ex.add(_call("U", 0), ctx)
            await ex.add(_call("S", 1), ctx)
            await ex.add(_call("S", 2), ctx)
            # Before draining, 3 calls are pending (1 in-flight + 2 queued)
            assert ex.pending_count >= 2
            await ex.drain(ctx)
            assert ex.pending_count == 0

        asyncio.run(_run())

    def test_late_bind_registry_and_router(self) -> None:
        """Executors with deferred wiring work the same as eager ones."""

        async def _run():
            a = _TimedTool("A", concurrency_safe=True, sleep_ms=5)
            reg = _make_registry([a])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor()  # nothing bound yet
            ex.bind_registry(reg)
            ex.bind_router(router)
            ctx = ToolContext()
            await ex.add(_call("A", 0), ctx)
            return await ex.drain(ctx)

        results = asyncio.run(_run())
        assert results[0]["tool_use_id"] == "tu_0"


# ─────────────────────────────────────────────────────────────────
# Interleaving across safe/unsafe/safe
# ─────────────────────────────────────────────────────────────────


class TestInterleavedMix:
    def test_safe_unsafe_safe_ordering(self) -> None:
        """Calls arrive S/U/S → results emerge in receive order, U stalls the
        later safe until the unsafe has finished."""

        async def _run():
            s1 = _TimedTool("A", concurrency_safe=True, sleep_ms=30)
            u = _TimedTool("U", concurrency_safe=False, sleep_ms=30)
            s2 = _TimedTool("B", concurrency_safe=True, sleep_ms=30)
            reg = _make_registry([s1, u, s2])
            router = RegistryRouter(reg)
            ex = StreamingToolExecutor(registry=reg, router=router)
            ctx = ToolContext()

            await ex.add(_call("A", 0), ctx)
            await ex.add(_call("U", 1), ctx)
            await ex.add(_call("B", 2), ctx)

            results = await ex.drain(ctx)
            return results, s1, u, s2

        results, s1, u, s2 = asyncio.run(_run())

        assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_1", "tu_2"]
        # u started only after s1 finished (in-flight at time of enqueue)
        assert u.started_at[0] >= s1.finished_at[0] - 0.005
        # s2 started only after u finished (queued behind barrier)
        assert s2.started_at[0] >= u.finished_at[0] - 0.005
