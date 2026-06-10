"""``Pipeline.events()`` — the multi-subscriber cursor tap (2.2.0).

Audit 2026-06-09 §3.2 / Tier 1-1: ``run_stream`` is single-consumer and
run-scoped; ``on()`` is callback-only with no catch-up. Geny compensated
with a 50ms polling loop over ``state.events``. The tap is the
library-owned replacement: bounded ring journal, ``seq`` cursors,
N concurrent subscribers, clean detach on generator close and
``aclose()``.
"""

from __future__ import annotations

import asyncio

import pytest

from geny_executor import Pipeline, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s21_yield import YieldStage


def _pipeline(**kwargs) -> Pipeline:
    p = Pipeline(**kwargs)
    p.register_stage(InputStage())
    p.register_stage(APIStage(provider=MockProvider()))
    p.register_stage(YieldStage())
    return p


async def _consume_until_complete(tap):
    out = []
    async for event in tap:
        out.append(event)
        if event.type == "pipeline.complete":
            break
    return out


@pytest.mark.asyncio
async def test_two_concurrent_subscribers_see_identical_sequences():
    pipeline = _pipeline()
    t1 = asyncio.create_task(_consume_until_complete(pipeline.events()))
    t2 = asyncio.create_task(_consume_until_complete(pipeline.events()))
    await asyncio.sleep(0)  # let both taps register before the run

    await pipeline.run("hello")

    a = await asyncio.wait_for(t1, 2)
    b = await asyncio.wait_for(t2, 2)
    assert [e.seq for e in a] == [e.seq for e in b]
    assert [e.type for e in a] == [e.type for e in b]
    assert a, "subscribers must have seen the run"
    assert "text.delta" in [e.type for e in a], "tap carries the state channel too"


@pytest.mark.asyncio
async def test_late_joiner_replays_from_cursor_without_dup_or_gap():
    pipeline = _pipeline()
    await pipeline.run("turn one")

    # Live subscriber from now on; late joiner replays turn one's
    # journal, then crosses the replay→live boundary into turn two.
    live = asyncio.create_task(_consume_until_complete(pipeline.events()))
    late = asyncio.create_task(_collect_n_completes(pipeline.events(replay_from=0), 2))
    await asyncio.sleep(0)

    state = PipelineState()
    await pipeline.run("turn two", state)

    live_events = await asyncio.wait_for(live, 2)
    late_events = await asyncio.wait_for(late, 2)

    # Late joiner: BOTH turns, every seq exactly once, no gap at the
    # replay→live seam (the seam bug this cursor design prevents).
    late_seqs = [e.seq for e in late_events]
    assert late_seqs == list(range(1, late_seqs[-1] + 1))
    # Live subscriber: only turn two — picks up exactly where the
    # journal's turn-one tail ended.
    live_seqs = [e.seq for e in live_events]
    assert live_seqs[0] > 1
    assert live_seqs == late_seqs[live_seqs[0] - 1 :]


@pytest.mark.asyncio
async def test_replay_cursor_resumes_after_a_seen_seq():
    pipeline = _pipeline()
    await pipeline.run("one")
    journal = list(pipeline._event_journal)
    cursor = journal[3].seq  # pretend the host processed up to here

    tap = pipeline.events(replay_from=cursor)
    replayed = []
    # Drain only the replay portion (no live events are coming).
    agen = tap.__aiter__()
    try:
        while True:
            replayed.append(await asyncio.wait_for(agen.__anext__(), 0.2))
    except asyncio.TimeoutError:
        await tap.aclose()

    assert [e.seq for e in replayed] == [e.seq for e in journal if e.seq > cursor]


@pytest.mark.asyncio
async def test_default_is_live_only():
    pipeline = _pipeline()
    await pipeline.run("one")

    tap = pipeline.events()  # replay_from=-1 → nothing from the journal
    agen = tap.__aiter__()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(agen.__anext__(), 0.1)
    await tap.aclose()


@pytest.mark.asyncio
async def test_generator_close_mid_stream_detaches_queue():
    """Closing a tap mid-run leaks nothing: queue list shrinks back and
    the run itself is unaffected. Uses ``contextlib.aclosing`` — the
    documented consumption pattern for breaking out early (a bare
    ``break`` leaves the generator suspended until GC finalizes it,
    which still detaches, just not deterministically)."""
    from contextlib import aclosing

    pipeline = _pipeline()
    consumed = []

    async def consume_one():
        async with aclosing(pipeline.events()) as tap:
            async for event in tap:
                consumed.append(event)
                break  # abandon mid-stream — aclosing detaches on exit

    task = asyncio.create_task(consume_one())
    await asyncio.sleep(0)
    assert len(pipeline._event_taps) == 1

    result = await pipeline.run("hello")
    assert result.success
    await asyncio.wait_for(task, 2)

    assert pipeline._event_taps == []
    assert consumed and consumed[0].seq > 0


@pytest.mark.asyncio
async def test_aclose_terminates_live_taps():
    """aclose() must wake blocked subscribers so host tasks unwind —
    the tap-side half of the §2.4 teardown contract."""
    pipeline = _pipeline()

    async def consume_all():
        return [e async for e in pipeline.events()]

    task = asyncio.create_task(consume_all())
    await asyncio.sleep(0)
    assert len(pipeline._event_taps) == 1

    await pipeline.aclose()
    events = await asyncio.wait_for(task, 2)
    assert events == []
    assert pipeline._event_taps == []


@pytest.mark.asyncio
async def test_events_on_closed_pipeline_returns_immediately():
    pipeline = _pipeline()
    await pipeline.aclose()
    assert [e async for e in pipeline.events(replay_from=0)] == []


@pytest.mark.asyncio
async def test_journal_is_bounded_ring():
    """The journal drops oldest-first at the configured cap; replay
    starts at the oldest retained event instead of growing without
    bound (the audit's events-list complaint, not repeated here)."""
    pipeline = _pipeline(event_journal_size=5)
    await pipeline.run("hello")

    journal = list(pipeline._event_journal)
    assert len(journal) == 5
    # Newest five seqs retained.
    assert journal[-1].seq == pipeline._event_seq
    assert [e.seq for e in journal] == list(
        range(pipeline._event_seq - 4, pipeline._event_seq + 1)
    )

    # Late joiner replaying from 0 only gets what the ring still holds.
    tap = pipeline.events(replay_from=0)
    agen = tap.__aiter__()
    replayed = []
    try:
        while True:
            replayed.append(await asyncio.wait_for(agen.__anext__(), 0.1))
    except asyncio.TimeoutError:
        await tap.aclose()
    assert [e.seq for e in replayed] == [e.seq for e in journal]


def test_invalid_journal_size_rejected():
    with pytest.raises(ValueError):
        Pipeline(event_journal_size=0)


@pytest.mark.asyncio
async def test_tap_sees_runs_across_sessions_with_correlation():
    """The tap is pipeline-scoped on purpose (a session UI filters by
    session_id/run_id) — verify both turns arrive, distinguishable."""
    pipeline = _pipeline()
    task = asyncio.create_task(_collect_n_completes(pipeline.events(), 2))
    await asyncio.sleep(0)

    await pipeline.run("a", PipelineState(session_id="s1"))
    await pipeline.run("b", PipelineState(session_id="s2"))

    events = await asyncio.wait_for(task, 2)
    sessions = {e.session_id for e in events if e.type == "pipeline.start"}
    assert sessions == {"s1", "s2"}
    assert len({e.run_id for e in events if e.type == "pipeline.start"}) == 2


async def _collect_n_completes(tap, n: int):
    out = []
    completes = 0
    async for event in tap:
        out.append(event)
        if event.type == "pipeline.complete":
            completes += 1
            if completes == n:
                break
    return out
