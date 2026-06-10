"""``Pipeline.aclose()`` / ``close()`` teardown tests (2.2.0, audit §2.4).

The library never owned teardown before 2.2.0 — ``disconnect_all`` had
exactly one caller (the build-failure unwind inside
``from_manifest_async``), so hosts stopping a session with declared
``mcp_servers`` orphaned stdio MCP child processes (live Geny prod
leak). ``aclose()`` is the aggregation point; these tests pin its
contract: full aggregation, idempotency, and safety on pipelines whose
construction never attached anything.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from geny_executor import Pipeline, PipelineConfig


class _FakeMCPManager:
    """Counts disconnect_all calls (the §2.4 leak's missing call)."""

    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect_all(self) -> None:
        self.disconnect_calls += 1


class _ExplodingMCPManager:
    async def disconnect_all(self) -> None:
        raise RuntimeError("server wedged")


class _FakeToolProvider:
    """Mimics the started-provider surface shutdown_providers walks."""

    def __init__(self, log: List[str], name: str = "fake") -> None:
        self._log = log
        self.name = name

    async def shutdown(self) -> None:
        self._log.append(f"shutdown:{self.name}")


def _pipeline_with_resources(
    mcp: Any = None, providers: List[Any] | None = None
) -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="aclose-test"))
    if mcp is not None:
        pipeline._mcp_manager = mcp
    if providers is not None:
        pipeline._tool_providers = providers
    return pipeline


@pytest.mark.asyncio
async def test_aclose_disconnects_mcp_and_shuts_providers():
    log: List[str] = []
    mcp = _FakeMCPManager()
    pipeline = _pipeline_with_resources(mcp, [_FakeToolProvider(log)])

    await pipeline.aclose()

    assert mcp.disconnect_calls == 1
    assert log == ["shutdown:fake"]
    assert pipeline.tool_providers == []


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """Second call must be a no-op — hosts double-close from both the
    session reaper and the atexit path; that must not double-disconnect."""
    mcp = _FakeMCPManager()
    pipeline = _pipeline_with_resources(mcp)

    await pipeline.aclose()
    await pipeline.aclose()

    assert mcp.disconnect_calls == 1


@pytest.mark.asyncio
async def test_aclose_on_bare_pipeline_is_safe():
    """A hand-built pipeline (or one whose build failed before any
    resource attached) has nothing to tear down — aclose must not raise."""
    pipeline = Pipeline(PipelineConfig(name="bare"))
    await pipeline.aclose()  # no MCP manager, no providers, no HITL
    await pipeline.aclose()  # and stays idempotent


@pytest.mark.asyncio
async def test_aclose_mcp_failure_still_shuts_providers():
    """Best-effort contract: a wedged MCP server must not leak the tool
    providers behind it."""
    log: List[str] = []
    pipeline = _pipeline_with_resources(_ExplodingMCPManager(), [_FakeToolProvider(log)])

    await pipeline.aclose()  # must not raise

    assert log == ["shutdown:fake"]


@pytest.mark.asyncio
async def test_aclose_cancels_pending_hitl():
    """Stage coroutines blocked on a human verdict must unwind at close
    instead of awaiting a future nobody will ever resolve."""
    from geny_executor.stages.s15_hitl.types import HITLDecision

    pipeline = Pipeline(PipelineConfig(name="hitl-close"))
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    pipeline._pending_hitl["tok-1"] = future

    await pipeline.aclose()

    assert future.done()
    assert future.result() == HITLDecision.CANCEL
    assert pipeline.list_pending_hitl() == []


@pytest.mark.asyncio
async def test_sync_close_inside_loop_schedules_aclose():
    """close() inside a running loop schedules aclose as a task and
    keeps a reference so the loop cannot GC it mid-flight."""
    mcp = _FakeMCPManager()
    pipeline = _pipeline_with_resources(mcp)

    pipeline.close()
    assert pipeline._close_task is not None
    await pipeline._close_task

    assert mcp.disconnect_calls == 1
    # And idempotent through the sync wrapper too.
    pipeline.close()
    assert mcp.disconnect_calls == 1


def test_sync_close_outside_loop_blocks_until_done():
    mcp = _FakeMCPManager()
    pipeline = _pipeline_with_resources(mcp)

    pipeline.close()

    assert mcp.disconnect_calls == 1


# ── closed-pipeline guard (review N2) ─────────────────────────────────


def _runnable_pipeline() -> Pipeline:
    from geny_executor.stages.s01_input import InputStage
    from geny_executor.stages.s06_api import APIStage, MockProvider
    from geny_executor.stages.s09_parse import ParseStage
    from geny_executor.stages.s21_yield import YieldStage

    pipeline = Pipeline(PipelineConfig(name="closed-guard"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


@pytest.mark.asyncio
async def test_run_on_closed_pipeline_raises():
    """A closed pipeline used to run silently with MCP disconnected —
    every tool call degraded with no hint why. Fail fast instead."""
    pipeline = _runnable_pipeline()
    result = await pipeline.run("before close")
    assert result.success is True

    await pipeline.aclose()

    with pytest.raises(RuntimeError, match="pipeline is closed — build a new one"):
        await pipeline.run("after close")


@pytest.mark.asyncio
async def test_run_stream_on_closed_pipeline_raises():
    pipeline = _runnable_pipeline()
    await pipeline.aclose()

    agen = pipeline.run_stream("after close")
    with pytest.raises(RuntimeError, match="pipeline is closed — build a new one"):
        await agen.__anext__()
