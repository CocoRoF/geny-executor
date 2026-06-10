"""Cancellation & consumer-abandonment suite (2.2.0 wave 3).

Audit 2026-06-09 §3.7: "cancellation 테스트 0 (run_stream 소비자 이탈,
CLI subprocess 고아화) — 두 host 모두 SSE 서버라 client disconnect 가
일상". Both hosts stream pipeline events over SSE, so a browser tab
closing mid-answer is the NORMAL ending of a run, not an edge case —
yet nothing pinned what the engine does when the consumer walks away.

This module pins the pipeline half of that contract (the CLI
subprocess half lives in ``tests/llm_client/unit/test_cli_cancellation.py``):

  * run_stream abandonment — the background run task keeps executing
    to completion (documented run-to-completion semantics), the
    ``_runs_in_flight`` counter drains back to 0, the journal/taps
    record the terminal event nobody watched, and nothing trickles
    out after completion.
  * events() tap closed mid-stream — queue detaches, no leaked tasks,
    the journal keeps recording for the next subscriber.
  * run() task cancelled mid-turn — the run-in-progress lock releases
    and the pipeline accepts the next turn.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from typing import Any, List

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.core.stage import Stage
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s21_yield import YieldStage


class _SlowStreamProvider(MockProvider):
    """MockProvider whose stream trickles deltas with real awaits.

    Keeps a run observably in flight long enough for a consumer to
    abandon the stream between deltas — MockProvider's stock stream
    yields everything in one scheduler slice, which can't reproduce a
    mid-stream disconnect.
    """

    def __init__(self, *, words: int = 10, delay_s: float = 0.02) -> None:
        super().__init__(default_text=" ".join(f"w{i}" for i in range(words)))
        self._delay_s = delay_s

    async def create_message_stream(self, request):  # noqa: ANN001
        response = await self.create_message(request)
        text = response.content[0].text or ""
        for word in text.split(" "):
            await asyncio.sleep(self._delay_s)
            yield {"type": "text_delta", "text": word + " "}
        yield {"type": "message_complete", "response": response}


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


def _pipeline(provider: Any = None) -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="cancel-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider or _SlowStreamProvider()))
    pipeline.register_stage(YieldStage())
    return pipeline


async def _wait_runs_drained(pipeline: Pipeline, timeout: float = 2.0) -> None:
    """Poll until the run-in-flight counter returns to 0."""
    deadline = time.monotonic() + timeout
    while pipeline._runs_in_flight:
        assert time.monotonic() < deadline, (
            f"_runs_in_flight stuck at {pipeline._runs_in_flight} after "
            f"{timeout}s — the abandoned background run task never finished"
        )
        await asyncio.sleep(0.01)


async def _abandon_after_first_delta(pipeline: Pipeline, state: PipelineState) -> str:
    """Consume a run_stream until the first text.delta, then walk away.

    Returns the abandoned run's run_id (from its pipeline.start event).
    ``aclosing`` makes the disconnect deterministic — an SSE server's
    generator gets closed by the framework exactly like this when the
    client drops.
    """
    run_id = ""
    stream = pipeline.run_stream("abandoned turn", state)
    async with aclosing(stream):
        async for event in stream:
            if event.type == "pipeline.start":
                run_id = event.run_id
            if event.type == "text.delta":
                break
    assert run_id, "stream never produced pipeline.start"
    return run_id


# ── run_stream consumer abandonment ──────────────────────────────────


@pytest.mark.asyncio
async def test_abandoned_stream_background_run_completes_and_lock_drains():
    """Breaking out of run_stream mid-delta must not wedge the engine:
    the background task runs phases to completion (documented
    run-to-completion semantics) and ``_runs_in_flight`` returns to 0,
    so refresh_runtime / the mutator unlock without host intervention."""
    pipeline = _pipeline()
    state = PipelineState(session_id="abandon-1")

    run_id = await _abandon_after_first_delta(pipeline, state)
    assert pipeline.run_in_progress is True  # background task still executing

    await _wait_runs_drained(pipeline)

    # The journal recorded the completion nobody watched — a late
    # subscriber can still reconstruct how the abandoned run ended.
    journal_types = [
        (e.type, e.run_id) for e in pipeline._event_journal
    ]
    assert ("pipeline.complete", run_id) in journal_types
    # And the lock actually released: between-turn maintenance is legal.
    pipeline.refresh_runtime(session_runtime=object())


@pytest.mark.asyncio
async def test_abandoned_stream_tap_keeps_recording_then_goes_silent():
    """An events() subscriber (the audit's host-UI shape) sees the
    abandoned run through to pipeline.complete, then NOTHING more —
    an abandoned generator must not keep producing events after the
    run finished. aclose() then completes cleanly and unwinds the tap."""
    pipeline = _pipeline()
    seen: List[Any] = []

    async def tap_consumer() -> None:
        async for event in pipeline.events():
            seen.append(event)

    tap_task = asyncio.create_task(tap_consumer())
    await asyncio.sleep(0)  # let the tap register before the run

    run_id = await _abandon_after_first_delta(pipeline, PipelineState(session_id="abandon-2"))
    await _wait_runs_drained(pipeline)

    # Tap saw the terminal event despite the run_stream consumer leaving.
    await asyncio.sleep(0.05)  # let queued events flush into the tap
    assert any(e.type == "pipeline.complete" and e.run_id == run_id for e in seen)

    # Silence after completion: no stray deltas trickle out of the
    # abandoned generator or the finished background task.
    count_after_complete = len(seen)
    await asyncio.sleep(0.2)
    assert len(seen) == count_after_complete

    # Teardown completes cleanly and wakes the blocked subscriber.
    await asyncio.wait_for(pipeline.aclose(), 2)
    await asyncio.wait_for(tap_task, 2)
    assert pipeline._event_taps == []


@pytest.mark.asyncio
async def test_abandoned_stream_unsubscribes_its_bus_collector():
    """The run-scoped bus collector must detach when the generator is
    closed — otherwise every disconnected SSE client leaves a callback
    feeding an unbounded queue forever (per-disconnect memory leak)."""
    pipeline = _pipeline()
    handlers_before = len(pipeline.event_bus._handlers.get("*", []))

    await _abandon_after_first_delta(pipeline, PipelineState(session_id="abandon-3"))
    await _wait_runs_drained(pipeline)

    assert len(pipeline.event_bus._handlers.get("*", [])) == handlers_before


# ── events() subscriber closes mid-stream ────────────────────────────


@pytest.mark.asyncio
async def test_tap_close_mid_run_leaks_no_tasks_and_journal_continues():
    """Closing an events() tap mid-run detaches its queue without
    leaking any asyncio task, and the journal keeps recording for the
    NEXT subscriber (the tap is an observer, never load-bearing)."""
    pipeline = _pipeline(MockProvider(default_text="quick"))
    tasks_before = asyncio.all_tasks()

    consumed: List[Any] = []

    async def consume_one() -> None:
        async with aclosing(pipeline.events()) as tap:
            async for event in tap:
                consumed.append(event)
                break  # subscriber disconnects mid-stream

    tap_task = asyncio.create_task(consume_one())
    await asyncio.sleep(0)
    assert len(pipeline._event_taps) == 1

    result = await pipeline.run("turn one", PipelineState(session_id="tap-close"))
    assert result.success
    await asyncio.wait_for(tap_task, 2)
    assert consumed, "subscriber must have seen at least one event"
    assert pipeline._event_taps == []

    # Journal keeps recording after the tap died.
    journal_after_first = len(pipeline._event_journal)
    await pipeline.run("turn two", PipelineState(session_id="tap-close-2"))
    assert len(pipeline._event_journal) > journal_after_first

    # No task leaked: everything spawned since the baseline has finished.
    leaked = {t for t in asyncio.all_tasks() - tasks_before if not t.done()}
    assert leaked == set(), f"leaked tasks: {leaked}"


# ── mid-run cancellation of run() ────────────────────────────────────


@pytest.mark.asyncio
async def test_cancelled_run_releases_lock_and_next_run_succeeds():
    """Host cancels the turn (request timeout, user hit stop): the
    run-in-progress lock must release and the pipeline must accept the
    next turn — a wedged counter would brick the session's mutator and
    refresh_runtime forever."""
    pipeline = Pipeline(PipelineConfig(name="cancel-run"))
    pipeline.register_stage(InputStage())
    gate = _GateStage()
    pipeline.register_stage(gate)
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(YieldStage())

    task = asyncio.create_task(pipeline.run("doomed turn"))
    await gate.entered.wait()
    assert pipeline.run_in_progress is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert pipeline.run_in_progress is False

    # The pipeline is immediately usable for the next turn.
    gate.release.set()
    result = await pipeline.run("next turn", PipelineState(session_id="after-cancel"))
    assert result.success is True
    assert pipeline.run_in_progress is False


@pytest.mark.asyncio
async def test_cancelled_run_leaves_no_pending_lock_for_refresh():
    """refresh_runtime is the documented between-turn API — it must be
    legal right after a cancelled turn, not blocked by a ghost run."""
    pipeline = Pipeline(PipelineConfig(name="cancel-refresh"))
    pipeline.register_stage(InputStage())
    gate = _GateStage()
    pipeline.register_stage(gate)
    pipeline.register_stage(APIStage(provider=MockProvider()))
    pipeline.register_stage(YieldStage())

    task = asyncio.create_task(pipeline.run("doomed"))
    await gate.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pipeline.refresh_runtime(session_runtime=object())  # must not raise
