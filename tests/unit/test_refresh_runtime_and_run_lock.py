"""refresh_runtime + run-in-progress lock (2.2.0, audit §3.3/#5, §3.5).

``attach_runtime`` is construction-time-only, so hosts that needed
between-turn runtime updates (credential rotation, tool_context swap)
reached into private setters (Geny's ~220-line queue_runtime_refresh).
``refresh_runtime`` is the legal API: same wiring, gated only on
run-in-progress. The same engine-wired flag finally makes
``MutationLocked`` fireable in prod — the manual ``lock_stage`` API was
dead (nothing ever called it).
"""

from __future__ import annotations

import asyncio

import pytest

from geny_executor import (
    MutationLocked,
    Pipeline,
    PipelineConfig,
    PipelineMutator,
    PipelineState,
)
from geny_executor.core.stage import Stage
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s10_tool import ToolStage
from geny_executor.stages.s21_yield import YieldStage
from geny_executor.tools.base import ToolContext


class _GateStage(Stage):
    """Blocks mid-run until released — lets tests observe an in-flight run."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def name(self) -> str:
        return "gate"

    @property
    def order(self) -> int:
        return 2

    async def execute(self, input, state):  # noqa: ANN001
        self.entered.set()
        await self.release.wait()
        return input


def _make_pipeline(with_gate: bool = False, with_tool: bool = False):
    pipeline = Pipeline(PipelineConfig(name="run-lock"))
    pipeline.register_stage(InputStage())
    gate = None
    if with_gate:
        gate = _GateStage()
        pipeline.register_stage(gate)
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    if with_tool:
        pipeline.register_stage(ToolStage())
    pipeline.register_stage(YieldStage())
    return pipeline, gate


# ── refresh_runtime between turns ────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_runtime_between_turns_swaps_tool_context():
    pipeline, _ = _make_pipeline(with_tool=True)
    state = PipelineState(session_id="refresh")
    await pipeline.run("turn one", state)

    # attach_runtime keeps its historical hard-error contract…
    with pytest.raises(RuntimeError, match="attach_runtime"):
        pipeline.attach_runtime(tool_context=ToolContext())

    # …refresh_runtime is the sanctioned between-turn path.
    new_ctx = ToolContext(working_dir="/tmp/new-turn")
    pipeline.refresh_runtime(tool_context=new_ctx)

    tool_stage = next(s for s in pipeline.stages if s.name == "tool")
    assert tool_stage._context is new_ctx

    # And the pipeline still runs.
    result = await pipeline.run("turn two", state)
    assert result.success is True


def test_refresh_runtime_rejects_unknown_kwargs():
    pipeline, _ = _make_pipeline()
    with pytest.raises(TypeError):
        pipeline.refresh_runtime(not_a_real_kwarg=1)


# ── run-in-progress lock ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_during_run_raises():
    pipeline, gate = _make_pipeline(with_gate=True)
    task = asyncio.create_task(pipeline.run("blocked turn"))
    await gate.entered.wait()

    assert pipeline.run_in_progress is True
    with pytest.raises(RuntimeError, match="run is in\\s+progress"):
        pipeline.refresh_runtime(tool_context=ToolContext())

    gate.release.set()
    result = await task
    assert result.success is True
    assert pipeline.run_in_progress is False
    # Legal again once the run drained.
    pipeline.refresh_runtime(session_runtime=object())


@pytest.mark.asyncio
async def test_mutator_raises_during_run():
    pipeline, gate = _make_pipeline(with_gate=True)
    mutator = PipelineMutator(pipeline)
    task = asyncio.create_task(pipeline.run("blocked turn"))
    await gate.entered.wait()

    with pytest.raises(MutationLocked):
        mutator.update_model_config({"temperature": 0.7})
    with pytest.raises(MutationLocked):
        mutator.update_stage_config(6, {"provider": "openai"})
    with pytest.raises(MutationLocked):
        mutator.swap_strategy(6, "retry", "none")

    gate.release.set()
    await task

    # Unlocked after the run — the same mutation now succeeds.
    result = mutator.update_model_config({"temperature": 0.7})
    assert result.success is True


@pytest.mark.asyncio
async def test_run_stream_background_task_holds_the_lock():
    """The lock covers the streaming background task's lifetime, not
    just the visible iteration of the generator."""
    pipeline, gate = _make_pipeline(with_gate=True)

    agen = pipeline.run_stream("streamed turn")
    # Drive the generator until the run is provably in flight.
    first = await agen.__anext__()
    assert first.type == "pipeline.start"
    consume = asyncio.create_task(agen.__anext__())
    await gate.entered.wait()

    assert pipeline.run_in_progress is True
    with pytest.raises(RuntimeError):
        pipeline.refresh_runtime(tool_context=ToolContext())

    gate.release.set()
    await consume
    # Drain the rest.
    async for _ in agen:
        pass
    assert pipeline.run_in_progress is False


# ── lock_stage stays a working manual flag ───────────────────────────


def test_manual_lock_stage_still_blocks_and_unblocks():
    """Legacy per-stage lock keeps working for host-side freezes; it is
    documented as manual-only (the engine wires run_in_progress instead)."""
    pipeline, _ = _make_pipeline()
    mutator = PipelineMutator(pipeline)

    mutator.lock_stage(6)
    with pytest.raises(MutationLocked):
        mutator.update_stage_config(6, {"provider": "openai"})

    mutator.unlock_stage(6)
    assert mutator.update_stage_config(6, {"provider": "openai"}).success is True
