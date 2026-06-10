"""Per-run ``ModelOverrides`` (2.2.0, audit §3.1).

GAPT had no sanctioned "this run only, use a different model" API and
mutated ``pipeline._config.model.*`` directly with a hand-built
baseline/revert dance. ``run(..., overrides=ModelOverrides(...))`` is
the public funnel: applied to state AFTER ``apply_to_state`` (so it
wins for that run), reverted by the NEXT run's stomp by construction,
and announced via ``config.override_applied`` events.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from geny_executor import (
    ModelConfig,
    ModelOverrides,
    Pipeline,
    PipelineConfig,
    PipelineState,
)
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _make_pipeline() -> Pipeline:
    pipeline = Pipeline(
        PipelineConfig(
            name="overrides",
            model=ModelConfig(model="claude-sonnet-4-6", max_tokens=8192, temperature=0.0),
        )
    )
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


# ── Dataclass surface ────────────────────────────────────────────────


def test_model_overrides_is_frozen():
    overrides = ModelOverrides(model="claude-opus-4-7")
    with pytest.raises(dataclasses.FrozenInstanceError):
        overrides.model = "other"  # type: ignore[misc]


def test_non_none_fields_only_lists_set_values():
    overrides = ModelOverrides(model="claude-opus-4-7", thinking_enabled=True)
    assert overrides.non_none_fields() == {
        "model": "claude-opus-4-7",
        "thinking_enabled": True,
    }
    assert ModelOverrides().non_none_fields() == {}


def test_exported_from_package_root():
    import geny_executor

    assert "ModelOverrides" in geny_executor.__all__
    assert geny_executor.ModelOverrides is ModelOverrides


# ── One-run lifetime ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_override_wins_for_one_run_then_reverts():
    pipeline = _make_pipeline()
    state = PipelineState(session_id="s")

    first = await pipeline.run(
        "turn one",
        state,
        overrides=ModelOverrides(model="claude-opus-4-7", max_tokens=1024, temperature=0.9),
    )
    assert first.model == "claude-opus-4-7"
    assert state.max_tokens == 1024
    assert state.temperature == pytest.approx(0.9)

    # Next run with NO overrides: apply_to_state stomps back to config.
    second = await pipeline.run("turn two", state)
    assert second.model == "claude-sonnet-4-6"
    assert state.max_tokens == 8192
    assert state.temperature == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_override_applies_after_config_stomp_each_run():
    """Overrides must be re-supplied per run — they are a value for one
    run, not a sticky session setting."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="s")

    await pipeline.run("one", state, overrides=ModelOverrides(thinking_enabled=True))
    assert state.thinking_enabled is True

    await pipeline.run("two", state)
    assert state.thinking_enabled is False


# ── Events ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_override_applied_events_emitted_per_field():
    pipeline = _make_pipeline()
    result = await pipeline.run(
        "turn",
        overrides=ModelOverrides(model="claude-opus-4-7", thinking_budget_tokens=2048),
    )

    events = [e for e in result.events if e["type"] == "config.override_applied"]
    payloads = {e["data"]["field"]: e["data"] for e in events}
    assert payloads == {
        "model": {"field": "model", "value": "claude-opus-4-7", "source": "per_run"},
        "thinking_budget_tokens": {
            "field": "thinking_budget_tokens",
            "value": 2048,
            "source": "per_run",
        },
    }


@pytest.mark.asyncio
async def test_no_override_emits_no_events():
    pipeline = _make_pipeline()
    result = await pipeline.run("turn", overrides=ModelOverrides())
    assert not [e for e in result.events if e["type"] == "config.override_applied"]


@pytest.mark.asyncio
async def test_override_events_visible_in_run_stream():
    """Streaming hosts see the events too — that's why application is
    deferred to phase start (after the listener attaches)."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="stream")
    seen = []
    async for event in pipeline.run_stream(
        "turn", state, overrides=ModelOverrides(max_tokens=4096)
    ):
        seen.append(event)

    override_events = [e for e in seen if e.type == "config.override_applied"]
    assert len(override_events) == 1
    assert override_events[0].data == {
        "field": "max_tokens",
        "value": 4096,
        "source": "per_run",
    }


@pytest.mark.asyncio
async def test_override_events_attributed_to_their_own_run_under_concurrency():
    """2.2.0 review B2: the override events were stashed on a
    pipeline-global FIFO flushed by whichever run started next, so two
    overlapping run_streams could deliver one run's overrides into the
    OTHER run's stream (wrong run_id). They now queue per-state."""

    class _SlowProvider(MockProvider):
        async def create_message(self, request):  # noqa: ANN001
            await asyncio.sleep(0.01)  # force the runs to interleave
            return await super().create_message(request)

    pipeline = Pipeline(
        PipelineConfig(name="overrides-concurrent", model=ModelConfig(max_tokens=8192))
    )
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=_SlowProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state_with = PipelineState(session_id="with-overrides")
    state_without = PipelineState(session_id="plain")

    async def _consume(text, state, **kwargs):
        return [e async for e in pipeline.run_stream(text, state, **kwargs)]

    # Start the overridden run first so its events are stashed before
    # the plain run begins — the exact window the global FIFO leaked in.
    events_with, events_without = await asyncio.gather(
        _consume("one", state_with, overrides=ModelOverrides(max_tokens=1024)),
        _consume("two", state_without),
    )

    applied = [e for e in events_with if e.type == "config.override_applied"]
    leaked = [e for e in events_without if e.type == "config.override_applied"]
    assert [e.data["field"] for e in applied] == ["max_tokens"]
    assert leaked == []

    # run_id attribution: the override event carries the overridden
    # run's own correlation id.
    (run_id_with,) = {e.run_id for e in events_with if e.run_id}
    assert applied[0].run_id == run_id_with


@pytest.mark.asyncio
async def test_interleaved_run_does_not_flush_another_runs_overrides():
    """Deterministic B2 shape: a run_stream is started (its overrides
    are stashed at _init_state) but its background task has not flushed
    yet when a second run()'s _run_phases begins — the second run must
    NOT deliver the first run's override events as its own."""
    pipeline = _make_pipeline()
    state_a = PipelineState(session_id="stream-a")
    state_b = PipelineState(session_id="plain-b")

    gen_a = pipeline.run_stream(
        "one", state_a, overrides=ModelOverrides(model="claude-opus-4-7")
    )
    # First __anext__ runs A's _init_state (overrides stashed) and
    # yields pipeline.start before A's background task flushes anything.
    first = await gen_a.__anext__()
    assert first.type == "pipeline.start"

    result_b = await pipeline.run("two", state_b)
    leaked = [e for e in result_b.events if e["type"] == "config.override_applied"]
    assert leaked == []

    remaining = [e async for e in gen_a]
    applied = [e for e in remaining if e.type == "config.override_applied"]
    assert [e.data["field"] for e in applied] == ["model"]
    assert {e.session_id for e in applied} == {"stream-a"}
