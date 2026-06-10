"""PipelineEvent correlation fields — session_id / run_id / seq (2.2.0).

Audit 2026-06-09 §3.2: hosts running several sessions in one process
had no way to attribute an event to its conversation; Geny wrapped
every collector in closure-captured session ids. The engine now stamps
correlation on every event it publishes, on both channels.
"""

from __future__ import annotations

import asyncio

import pytest

from geny_executor import Pipeline, PipelineEvent, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _pipeline() -> Pipeline:
    p = Pipeline()
    p.register_stage(InputStage())
    p.register_stage(APIStage(provider=MockProvider()))
    p.register_stage(ParseStage())
    p.register_stage(YieldStage())
    return p


def test_pipeline_event_field_appends_are_backcompat():
    """The 2.2.0 fields are appends with defaults — pre-existing
    positional construction must keep working."""
    e = PipelineEvent("x.y", "stage", 3)
    assert e.session_id == ""
    assert e.run_id == ""
    assert e.seq == 0


@pytest.mark.asyncio
async def test_run_stamps_session_and_run_id_on_bus_events():
    pipeline = _pipeline()
    seen = []
    pipeline.on("*", lambda e: seen.append(e))

    state = PipelineState(session_id="sess-42")
    await pipeline.run("hello", state)

    assert seen, "expected events"
    run_ids = {e.run_id for e in seen}
    assert len(run_ids) == 1
    assert "" not in run_ids
    assert all(e.session_id == "sess-42" for e in seen)


@pytest.mark.asyncio
async def test_state_channel_events_carry_correlation_too():
    """text.delta & friends originate in state.add_event — the bridge
    must stamp them identically to bus-native events."""
    pipeline = _pipeline()
    state = PipelineState(session_id="sess-deltas")

    events = []
    async for event in pipeline.run_stream("hi", state):
        events.append(event)

    deltas = [e for e in events if e.type == "text.delta"]
    assert deltas, f"no text.delta in {[e.type for e in events]}"
    bus_native = [e for e in events if e.type == "stage.enter"]
    assert bus_native
    assert {e.run_id for e in deltas} == {e.run_id for e in bus_native}
    assert all(e.session_id == "sess-deltas" for e in deltas)


@pytest.mark.asyncio
async def test_each_run_gets_a_fresh_run_id():
    pipeline = _pipeline()
    seen = []
    pipeline.on("pipeline.start", lambda e: seen.append(e))

    state = PipelineState(session_id="s")
    await pipeline.run("one", state)
    await pipeline.run("two", state)

    assert len(seen) == 2
    assert seen[0].run_id != seen[1].run_id
    # ... but the session id is stable across the turns.
    assert seen[0].session_id == seen[1].session_id == "s"


@pytest.mark.asyncio
async def test_seq_is_monotonic_across_runs_and_channels():
    pipeline = _pipeline()
    seen = []
    pipeline.on("*", lambda e: seen.append(e))

    state = PipelineState()
    await pipeline.run("one", state)
    await pipeline.run("two", state)

    seqs = [e.seq for e in seen]
    assert all(s > 0 for s in seqs)
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "seq must never repeat on one pipeline"


@pytest.mark.asyncio
async def test_run_stream_events_share_one_run_id_and_ordered_seq():
    pipeline = _pipeline()
    events = []
    async for event in pipeline.run_stream("hi", PipelineState(session_id="x")):
        events.append(event)

    assert events[0].type == "pipeline.start"
    assert events[-1].type == "pipeline.complete"
    assert len({e.run_id for e in events}) == 1
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


@pytest.mark.asyncio
async def test_overlapping_streams_do_not_cross_pollinate():
    """Two concurrent run_stream calls on ONE pipeline (distinct
    states): each consumer must only see its own run's events — the
    collector filters on run_id (2.2.0; pre-unification the bus half of
    the merge already leaked across runs, the state half didn't —
    making the leak total would have been a regression)."""
    pipeline = _pipeline()

    async def consume(text, sid):
        out = []
        async for event in pipeline.run_stream(text, PipelineState(session_id=sid)):
            out.append(event)
        return out

    a, b = await asyncio.gather(consume("alpha", "sess-a"), consume("beta", "sess-b"))

    a_runs = {e.run_id for e in a}
    b_runs = {e.run_id for e in b}
    assert len(a_runs) == 1 and len(b_runs) == 1
    assert a_runs != b_runs
    assert {e.session_id for e in a} == {"sess-a"}
    assert {e.session_id for e in b} == {"sess-b"}
