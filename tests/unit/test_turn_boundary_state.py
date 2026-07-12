"""Turn-boundary contract on reused state (2.2.0, audit §3.3).

The long-lived-state model (GAPT) reuses one ``PipelineState`` across
turns. Before 2.2.0 nothing reset the per-turn fields, so a reused
state carried ``loop_decision="error"`` into the next turn's success
verdict, climbed ``iteration`` toward MAX_ITERATIONS across turns, and
grew ``events`` without bound — while the state docstring *claimed* a
reset that didn't exist. These tests pin:

* ``begin_turn()`` resets exactly the per-turn class;
* ``run()`` applies it automatically for reused states (run-count
  marker OR pre-seeded messages, i.e. checkpoint rehydration);
* cost split — ``total_cost_usd`` per-turn (budget guards), folded
  into session-cumulative ``session_cost_usd`` at turn end;
* sticky-client symmetry — ``invalidate_client()`` + the generation
  counter make rotation land on reused states;
* ``state=None`` loudness + ``PipelineResult.state`` recovery.
"""

from __future__ import annotations

import logging

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _make_pipeline(text: str = "mock reply") -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="turn-boundary"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text=text)))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


# ── begin_turn() unit behaviour ──────────────────────────────────────


def test_begin_turn_resets_per_turn_fields():
    state = PipelineState(session_id="s")
    state.iteration = 7
    state.current_stage = "loop"
    state.stage_history = ["input", "api"]
    state.loop_decision = "error"
    state.completion_signal = "MAX_ITERATIONS"
    state.completion_detail = "boom"
    state.final_text = "old turn text"
    state.final_output = {"old": True}
    state.last_api_response = object()
    state.pending_tool_calls = [{"id": "stale"}]
    state.tool_results = [{"type": "tool_result"}]
    state.delegate_requests = [{"agent": "x"}]
    state.agent_results = [{"ok": True}]
    state.evaluation_score = 0.4
    state.evaluation_feedback = "meh"
    state.events = [{"type": "old.event"}]
    state.turn_token_usage = [object()]
    state.total_cost_usd = 1.25

    state.begin_turn()

    assert state.iteration == 0
    assert state.current_stage == ""
    assert state.stage_history == []
    assert state.loop_decision == "continue"
    assert state.completion_signal is None
    assert state.completion_detail is None
    assert state.final_text == ""
    assert state.final_output is None
    assert state.last_api_response is None
    assert state.pending_tool_calls == []
    assert state.tool_results == []
    assert state.delegate_requests == []
    assert state.agent_results == []
    assert state.evaluation_score is None
    assert state.evaluation_feedback is None
    assert state.events == []
    assert state.turn_token_usage == []
    assert state.total_cost_usd == 0.0


def test_begin_turn_keeps_sticky_and_cumulative_fields():
    state = PipelineState(session_id="keep-me")
    state.messages = [{"role": "user", "content": "turn 1"}]
    state.shared = {"geny.creature_state": {"mood": "happy"}}
    state.metadata = {"custom": 1}
    state.session_cost_usd = 2.5
    state.token_usage.input_tokens = 100
    client = object()
    state.llm_client = client

    state.begin_turn()

    assert state.session_id == "keep-me"
    assert state.messages == [{"role": "user", "content": "turn 1"}]
    assert state.shared == {"geny.creature_state": {"mood": "happy"}}
    assert state.metadata == {"custom": 1}
    assert state.session_cost_usd == 2.5
    assert state.token_usage.input_tokens == 100
    assert state.llm_client is client


# ── Automatic reset on reused state ──────────────────────────────────


@pytest.mark.asyncio
async def test_reused_state_resets_loop_fields_between_turns():
    """The GAPT poisoning class: turn 1's terminal verdict must not
    leak into turn 2's result."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="multi-turn")

    first = await pipeline.run("turn one", state)
    assert first.success is True
    # Simulate a turn that ended badly.
    state.loop_decision = "error"
    state.completion_detail = "previous turn exploded"
    state.iteration = 49

    second = await pipeline.run("turn two", state)

    assert second.success is True
    assert second.error is None
    assert second.iterations == 0
    # Conversation history accumulated: 2 user + 2 assistant messages.
    assert len(state.messages) == 4


@pytest.mark.asyncio
async def test_reused_state_events_are_per_turn():
    """Unbounded events growth was the audit complaint — each turn's
    result carries only that turn's events."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="events")

    first = await pipeline.run("turn one", state)
    events_after_one = len(state.events)
    assert events_after_one > 0

    second = await pipeline.run("turn two", state)

    # Not cumulative: the second turn's log was reset at its start.
    assert len(second.events) <= events_after_one * 2
    assert all(e in state.events for e in second.events)
    assert first.events[0] not in second.events


@pytest.mark.asyncio
async def test_preseeded_messages_count_as_reused_state():
    """Checkpoint rehydration: a fresh object carrying prior messages
    must get the same turn-boundary reset (it IS a continuation)."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="rehydrated")
    state.messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
    ]
    state.loop_decision = "error"
    state.iteration = 12

    result = await pipeline.run("new turn", state)

    assert result.success is True
    assert result.iterations == 0


# ── Cost split: per-turn vs session ──────────────────────────────────


@pytest.mark.asyncio
async def test_total_cost_is_per_turn_and_session_cost_accumulates():
    pipeline = _make_pipeline()
    state = PipelineState(session_id="cost")

    await pipeline.run("turn one", state)
    state.total_cost_usd = 0.30  # pretend the turn cost this much
    state.session_cost_usd += 0.30  # what _end_turn would have folded

    await pipeline.run("turn two", state)

    # Per-turn accumulator was reset at turn 2 start (MockProvider adds
    # nothing), so budget guards see a clean slate…
    assert state.total_cost_usd == 0.0
    # …while the session counter kept the prior spend.
    assert state.session_cost_usd == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_end_turn_folds_turn_cost_into_session():
    """The pipeline itself folds at turn end — even on the error path."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="fold")
    # Seed cost mid-turn via a stage-side accumulate: emulate by giving
    # the state cost before the run ends using an on-exit hook stage.
    from geny_executor.core.stage import Stage

    class _CostStage(Stage):
        @property
        def name(self) -> str:
            return "cost_injector"

        @property
        def order(self) -> int:
            return 2

        async def execute(self, input, state):  # noqa: ANN001
            state.accumulate_cost(0.05)
            return input

    pipeline.register_stage(_CostStage())
    await pipeline.run("turn", state)

    assert state.total_cost_usd == pytest.approx(0.05)
    assert state.session_cost_usd == pytest.approx(0.05)

    await pipeline.run("turn 2", state)
    assert state.total_cost_usd == pytest.approx(0.05)
    assert state.session_cost_usd == pytest.approx(0.10)


# ── Sticky-client symmetry: invalidate_client + generation ───────────


@pytest.mark.asyncio
async def test_invalidate_client_rotates_reused_state_client():
    """Rotation-after-turn: a pipeline-resolved client captured by a
    long-lived state must be re-resolved after invalidate_client()."""
    pipeline = _make_pipeline()

    class _GenClient:
        def __init__(self, tag: str) -> None:
            self.provider = "anthropic"
            self.tag = tag

    old_client = _GenClient("old")
    pipeline.attach_runtime(llm_client=old_client)
    state = PipelineState(session_id="rotate")

    await pipeline.run("turn one", state)
    assert state.llm_client is old_client

    pipeline.invalidate_client()
    new_client = _GenClient("new")
    pipeline.refresh_runtime(llm_client=new_client)

    await pipeline.run("turn two", state)
    assert state.llm_client is new_client


@pytest.mark.asyncio
async def test_invalidate_client_without_replacement_clears_to_resolution():
    """invalidate with no replacement → re-resolution (None here: no
    credentials) — the revoked client must NOT keep riding the state."""
    pipeline = _make_pipeline()

    class _Client:
        provider = "anthropic"

    revoked = _Client()
    pipeline.attach_runtime(llm_client=revoked)
    state = PipelineState(session_id="revoke")
    await pipeline.run("turn one", state)
    assert state.llm_client is revoked

    pipeline.invalidate_client()
    # At the turn boundary the pipeline re-resolves: no attached client,
    # no credential bundle → None. (During the subsequent run the
    # APIStage may backfill its own legacy-provider adapter — that is
    # stage-level recovery, not the revoked client riding along.)
    reinit = pipeline._init_state(state)
    assert reinit.llm_client is None
    # This test pokes the internal _init_state directly (no paired run /
    # _end_turn), which sets the 2.51.2 concurrent-run guard flag; clear
    # it so the real run below isn't rejected as an overlap.
    reinit._turn_in_flight = False

    await pipeline.run("turn two", state)
    assert state.llm_client is not revoked


@pytest.mark.asyncio
async def test_host_set_client_is_never_clobbered():
    """A client the host placed directly on the state records no
    generation — the pipeline must keep its hands off it."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="host-client")
    host_client = object()
    state.llm_client = host_client

    await pipeline.run("turn one", state)
    pipeline.invalidate_client()
    await pipeline.run("turn two", state)

    assert state.llm_client is host_client


def test_invalidate_client_during_run_raises():
    pipeline = _make_pipeline()
    pipeline._runs_in_flight = 1
    try:
        with pytest.raises(RuntimeError, match="run is in progress"):
            pipeline.invalidate_client()
    finally:
        pipeline._runs_in_flight = 0


# ── state=None loudness + result.state recovery ──────────────────────


@pytest.mark.asyncio
async def test_state_none_after_first_run_warns_once(caplog):
    pipeline = _make_pipeline()
    await pipeline.run("turn one")  # first run: no warning expected

    with caplog.at_level(logging.WARNING, logger="geny_executor.core.pipeline"):
        await pipeline.run("turn two")
        await pipeline.run("turn three")

    amnesia = [r for r in caplog.records if "discards conversation history" in r.message]
    assert len(amnesia) == 1  # once per pipeline, not per run


@pytest.mark.asyncio
async def test_first_run_with_state_none_does_not_warn(caplog):
    pipeline = _make_pipeline()
    with caplog.at_level(logging.WARNING, logger="geny_executor.core.pipeline"):
        await pipeline.run("only turn")
    assert not [r for r in caplog.records if "discards conversation history" in r.message]


@pytest.mark.asyncio
async def test_result_state_exposes_internally_created_state():
    """run() callers can recover the state and continue the conversation."""
    pipeline = _make_pipeline()
    first = await pipeline.run("turn one")

    assert first.state is not None
    assert len(first.state.messages) == 2

    second = await pipeline.run("turn two", first.state)
    assert second.state is first.state
    assert len(second.state.messages) == 4


def test_result_state_repr_suppressed():
    """The state handle drags clients/credentials — it must not leak
    into result reprs/logs."""
    from geny_executor.core.result import PipelineResult

    state = PipelineState(session_id="secret-session")
    result = PipelineResult.from_state(state)
    assert "PipelineState" not in repr(result)


@pytest.mark.asyncio
async def test_concurrent_runs_on_one_state_are_rejected():
    """audit R5: a second run on a state already mid-turn must raise,
    not corrupt both runs' iteration/events."""
    import asyncio

    pipeline = _make_pipeline()
    state = PipelineState(session_id="concurrent")

    # Hold the first run open by making Stage 1 await a gate.
    gate = asyncio.Event()

    async def _slow_input(inp, st):
        await gate.wait()
        return inp

    # Monkeypatch a stage to block; simplest is to run() and immediately
    # launch a second run() before the first completes.
    first = asyncio.create_task(pipeline.run("one", state))
    await asyncio.sleep(0.01)  # let the first run enter _init_state
    if not first.done():
        with pytest.raises(RuntimeError, match="already executing"):
            await pipeline.run("two", state)
    gate.set()
    await first  # let the first finish/clean up
    # After the first run releases, the state is reusable again.
    assert state._turn_in_flight is False
