"""Pipeline-fired lifecycle hooks (2.2.0, audit §3.5 dead-handler fix).

``HookEvent`` advertised pipeline start/end, stage enter/exit and
loop-iteration-end for two releases while no engine path fired them —
hosts bound dead handlers. The pipeline now mirrors its bus events to
an attached hook runner. These tests pin: the five kinds fire with the
right payload shape, nothing fires without a runner, and a broken
handler cannot break the run.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.hooks.config import HookConfig
from geny_executor.hooks.events import FIRED_EVENTS, HookEvent, HookEventPayload
from geny_executor.hooks.runner import HookRunner
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _recording_runner() -> Tuple[HookRunner, List[HookEventPayload]]:
    """In-process-only runner (no GENY_ALLOW_HOOKS — subprocess layer
    stays locked, which is exactly the in-process firing path the
    pipeline uses)."""
    runner = HookRunner(HookConfig(enabled=True, entries={}, audit_log_path=None), env={})
    seen: List[HookEventPayload] = []

    for event in (
        HookEvent.PIPELINE_START,
        HookEvent.PIPELINE_END,
        HookEvent.STAGE_ENTER,
        HookEvent.STAGE_EXIT,
        HookEvent.LOOP_ITERATION_END,
    ):
        async def handler(payload, _event=event):  # noqa: ANN001
            seen.append(payload)
            return None

        runner.register_in_process(event, handler)
    return runner, seen


def _make_pipeline(runner: HookRunner | None = None) -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="hook-lifecycle"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    if runner is not None:
        pipeline.attach_runtime(hook_runner=runner)
    return pipeline


@pytest.mark.asyncio
async def test_run_fires_all_five_lifecycle_kinds():
    runner, seen = _recording_runner()
    pipeline = _make_pipeline(runner)

    result = await pipeline.run("hello", PipelineState(session_id="hooks"))
    assert result.success is True

    kinds = [p.event for p in seen]
    assert kinds[0] == HookEvent.PIPELINE_START
    assert kinds[-1] == HookEvent.PIPELINE_END
    assert HookEvent.STAGE_ENTER in kinds
    assert HookEvent.STAGE_EXIT in kinds
    assert HookEvent.LOOP_ITERATION_END in kinds


@pytest.mark.asyncio
async def test_stage_events_carry_stage_identity():
    runner, seen = _recording_runner()
    pipeline = _make_pipeline(runner)
    await pipeline.run("hello", PipelineState(session_id="hooks"))

    enters = [p for p in seen if p.event == HookEvent.STAGE_ENTER]
    names = [p.stage_name for p in enters]
    orders = [p.stage_order for p in enters]
    assert "input" in names and "api" in names
    assert 1 in orders and 6 in orders
    # Session/pipeline correlation rides every payload.
    assert all(p.session_id == "hooks" for p in seen)
    assert all(p.pipeline_id for p in seen)


@pytest.mark.asyncio
async def test_pipeline_end_reports_success_flag():
    runner, seen = _recording_runner()
    pipeline = _make_pipeline(runner)
    await pipeline.run("hello", PipelineState(session_id="hooks"))

    end = [p for p in seen if p.event == HookEvent.PIPELINE_END][0]
    assert end.details["success"] is True
    assert "iterations" in end.details


@pytest.mark.asyncio
async def test_loop_iteration_end_carries_decision():
    runner, seen = _recording_runner()
    pipeline = _make_pipeline(runner)
    await pipeline.run("hello", PipelineState(session_id="hooks"))

    loop_ends = [p for p in seen if p.event == HookEvent.LOOP_ITERATION_END]
    assert loop_ends
    assert loop_ends[-1].details["loop_decision"] == "complete"


@pytest.mark.asyncio
async def test_run_stream_fires_lifecycle_hooks_too():
    runner, seen = _recording_runner()
    pipeline = _make_pipeline(runner)

    async for _ in pipeline.run_stream("hello", PipelineState(session_id="hooks")):
        pass

    kinds = {p.event for p in seen}
    assert HookEvent.PIPELINE_START in kinds
    assert HookEvent.PIPELINE_END in kinds
    start = [p for p in seen if p.event == HookEvent.PIPELINE_START][0]
    assert start.details == {"streaming": True}


@pytest.mark.asyncio
async def test_no_runner_attached_runs_clean():
    """The no-hooks fast path: nothing fires, nothing breaks."""
    pipeline = _make_pipeline(runner=None)
    assert pipeline._hook_runner is None
    result = await pipeline.run("hello")
    assert result.success is True


@pytest.mark.asyncio
async def test_broken_handler_cannot_break_the_run():
    """Observability must never kill the pipeline — runner-level
    fail-isolation plus the pipeline's own catch."""
    runner = HookRunner(HookConfig(enabled=True, entries={}, audit_log_path=None), env={})

    async def explode(payload):  # noqa: ANN001
        raise RuntimeError("handler bug")

    runner.register_in_process(HookEvent.PIPELINE_START, explode)
    runner.register_in_process(HookEvent.STAGE_ENTER, explode)
    pipeline = _make_pipeline(runner)

    result = await pipeline.run("hello")
    assert result.success is True


def test_fired_events_contract_includes_lifecycle_five():
    """The taxonomy honesty set must track the new fire-sites."""
    assert {
        HookEvent.PIPELINE_START,
        HookEvent.PIPELINE_END,
        HookEvent.STAGE_ENTER,
        HookEvent.STAGE_EXIT,
        HookEvent.LOOP_ITERATION_END,
    } <= FIRED_EVENTS
