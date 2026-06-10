"""2.2.0 event-channel unification (audit 2026-06-09 §3.2).

Pre-2.2.0 the engine ran two disjoint event worlds: the EventBus
(stage transitions, ``pipeline.on()``) and ``state.add_event`` (text
deltas, api telemetry) — bus subscribers could never see the state
channel, and ``run_stream`` merged the two with a pair of collectors.
These tests pin the unified model: state events forward into the bus,
``run_stream`` subscribes exactly once, and nothing is duplicated or
reordered.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from geny_executor import Pipeline, PipelineState
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


@pytest.mark.asyncio
async def test_bus_subscribers_see_state_events_in_run():
    """The headline fix: pipeline.on('*') sees text.delta / api.* now."""
    pipeline = _pipeline()
    seen = []
    pipeline.on("*", lambda e: seen.append(e.type))

    await pipeline.run("hello")

    assert "text.delta" in seen
    assert "api.request" in seen
    assert "api.response" in seen
    # The bus-native family still arrives alongside.
    assert "stage.enter" in seen
    assert "pipeline.complete" in seen


@pytest.mark.asyncio
async def test_exact_and_prefix_subscriptions_match_state_events():
    pipeline = _pipeline()
    exact, prefixed = [], []
    pipeline.on("text.delta", lambda e: exact.append(e))
    pipeline.on("api.*", lambda e: prefixed.append(e))

    await pipeline.run("hello")

    assert exact, "exact-match subscriber must see bridged state events"
    assert any(e.type == "api.request" for e in prefixed)


@pytest.mark.asyncio
async def test_run_stream_no_duplicates():
    """One subscription, one delivery: the old two-collector merge
    could not duplicate (channels were disjoint) — the unified bus path
    must not start."""
    pipeline = _pipeline()
    events = []
    async for event in pipeline.run_stream("hi"):
        events.append(event)

    counts = Counter(e.type for e in events)
    assert counts["pipeline.start"] == 1
    assert counts["pipeline.complete"] == 1
    # Each seq appears exactly once — the strongest no-dup statement.
    seqs = [e.seq for e in events]
    assert len(set(seqs)) == len(seqs)


@pytest.mark.asyncio
async def test_run_stream_ordering_deltas_inside_api_stage_window():
    """Synchronous delivery preserved: deltas must land between the api
    stage's enter and exit (i.e. live, not buffered until stage end —
    the original reason run_stream existed)."""
    pipeline = _pipeline()
    types = []
    async for event in pipeline.run_stream("hi"):
        if event.type in ("stage.enter", "stage.exit") and event.stage == "api":
            types.append(f"{event.type}:api")
        elif event.type == "text.delta":
            types.append("text.delta")

    assert "text.delta" in types
    enter_idx = types.index("stage.enter:api")
    exit_idx = types.index("stage.exit:api")
    delta_indices = [i for i, t in enumerate(types) if t == "text.delta"]
    assert all(enter_idx < i < exit_idx for i in delta_indices), types


@pytest.mark.asyncio
async def test_run_stream_lifecycle_events_reach_bus_subscribers_too():
    """pipeline.start/complete were queue-only in streaming mode —
    on() subscribers and the journal never saw a streamed run's
    lifecycle. Now they flow through the bus like everything else."""
    pipeline = _pipeline()
    seen = []
    pipeline.on("pipeline.*", lambda e: seen.append(e.type))

    async for _ in pipeline.run_stream("hi"):
        pass

    assert "pipeline.start" in seen
    assert "pipeline.complete" in seen


@pytest.mark.asyncio
async def test_streaming_error_announced_once_with_correlation():
    """A failing run announces pipeline.error exactly once on the
    stream, carrying the run's correlation ids."""
    from geny_executor.core.stage import Stage

    class FailingStage(Stage):
        name = "input"
        order = 1

        async def execute(self, input, state):
            raise RuntimeError("boom")

    pipeline = Pipeline()
    pipeline.register_stage(FailingStage())

    events = []
    async for event in pipeline.run_stream("hi", PipelineState(session_id="err-s")):
        events.append(event)

    errors = [e for e in events if e.type == "pipeline.error"]
    assert len(errors) == 1
    assert errors[0].session_id == "err-s"
    assert errors[0].run_id
    assert errors[0].data["code"]


@pytest.mark.asyncio
async def test_legacy_state_event_listener_still_works():
    """Hosts that installed state._event_listener directly keep their
    feed — the bridge supplements, it does not replace."""
    pipeline = _pipeline()
    state = PipelineState()
    legacy = []
    state._event_listener = lambda d: legacy.append(d["type"])

    await pipeline.run("hello", state)

    assert "text.delta" in legacy


@pytest.mark.asyncio
async def test_bridge_repointed_when_state_moves_between_pipelines():
    """A long-lived state migrated to a fresh pipeline must feed the
    NEW pipeline's subscribers, not the old one's."""
    p1, p2 = _pipeline(), _pipeline()
    state = PipelineState()
    first, second = [], []
    p1.on("text.delta", lambda e: first.append(e))
    p2.on("text.delta", lambda e: second.append(e))

    await p1.run("one", state)
    n_first = len(first)
    assert n_first > 0
    await p2.run("two", state)

    assert len(first) == n_first, "old pipeline must not receive the new run"
    assert second, "new pipeline must receive the migrated state's events"


@pytest.mark.asyncio
async def test_add_event_without_pipeline_is_safe():
    """Bare states (unit-tested stages, host fixtures) have no bridge —
    add_event must behave exactly as before."""
    state = PipelineState()
    state.add_event("text.delta", {"text": "x"})
    assert state.events[-1]["type"] == "text.delta"
