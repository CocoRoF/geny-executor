"""Per-run ``ModelOverrides`` (2.2.0, audit §3.1).

GAPT had no sanctioned "this run only, use a different model" API and
mutated ``pipeline._config.model.*`` directly with a hand-built
baseline/revert dance. ``run(..., overrides=ModelOverrides(...))`` is
the public funnel: applied to state AFTER ``apply_to_state`` (so it
wins for that run), reverted by the NEXT run's stomp by construction,
and announced via ``config.override_applied`` events.
"""

from __future__ import annotations

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
