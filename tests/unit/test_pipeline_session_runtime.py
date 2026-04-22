"""Tests for state.session_runtime slot and Pipeline.attach_runtime wiring (v0.30.0).

Pins the contract for the plugin-oriented ``session_runtime`` kwarg:

- Free-shape (``Any``) — executor enforces no Protocol.
- Propagates into ``state.session_runtime`` at run start.
- Defaults to ``None`` when unattached, so existing hosts are unaffected.
- Post-run re-attach is refused (same discipline as the other kwargs).
- Stages observe the attached object via ``state.session_runtime``.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage


class _HostRuntime:
    """Illustrative host-side container — executor never sees this type.

    Plugin authors are free to use dataclasses, attrs classes, plain
    objects, or even a SimpleNamespace. The executor just propagates the
    reference; attribute access is the host's concern.
    """

    def __init__(self, *, creature_state=None, emitters=None):
        self.creature_state = creature_state
        self.emitters = emitters or []


def _minimal_pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="session-runtime-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


# ── default / opt-out semantics ─────────────────────────────────────


def test_fresh_state_has_null_session_runtime():
    state = PipelineState()
    assert state.session_runtime is None


def test_attach_runtime_no_session_runtime_kwarg_leaves_state_none():
    """Existing hosts that never pass session_runtime see no change."""
    pipeline = _minimal_pipeline()
    pipeline.attach_runtime()  # no kwargs
    state = pipeline._init_state(None)
    assert state.session_runtime is None


# ── attach → propagate ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_runtime_accepts_arbitrary_session_runtime():
    pipeline = _minimal_pipeline()
    runtime = _HostRuntime(creature_state={"species": "slime"})
    pipeline.attach_runtime(session_runtime=runtime)
    result = await pipeline.run("hi")
    assert result is not None


@pytest.mark.asyncio
async def test_session_runtime_lands_on_state_inside_stage():
    captured: dict = {}

    class _Probe(InputStage):
        async def execute(self, input, state):
            captured["runtime"] = state.session_runtime
            return await super().execute(input, state)

    pipeline = Pipeline(PipelineConfig(name="probe"))
    pipeline.register_stage(_Probe())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    runtime = _HostRuntime(creature_state={"mood": 0.7})
    pipeline.attach_runtime(session_runtime=runtime)
    await pipeline.run("hi")

    assert captured["runtime"] is runtime
    # Duck-type access matches the docstring's guideline
    assert getattr(captured["runtime"], "creature_state", None) == {"mood": 0.7}


# ── type openness ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_runtime_accepts_any_type():
    """Executor must not inspect or constrain the runtime's shape."""
    pipeline = _minimal_pipeline()

    # A dict, a plain object, a lambda — the executor treats them all alike.
    for runtime in [
        {"creature_state": "bare dict"},
        object(),
        lambda: None,
    ]:
        pipeline._attached_session_runtime = None
        pipeline._has_started = False
        pipeline.attach_runtime(session_runtime=runtime)
        state = pipeline._init_state(None)
        assert state.session_runtime is runtime


# ── explicit state wins over the attached default ──────────────────


def test_pre_populated_state_session_runtime_is_preserved():
    """When the caller supplies a state whose session_runtime is already
    set, _init_state must not overwrite it with the attached default —
    matching the existing llm_client semantics."""
    pipeline = _minimal_pipeline()
    attached = _HostRuntime(creature_state="attached")
    caller_supplied = _HostRuntime(creature_state="caller")
    pipeline.attach_runtime(session_runtime=attached)

    pre_state = PipelineState()
    pre_state.session_runtime = caller_supplied
    out_state = pipeline._init_state(pre_state)
    assert out_state.session_runtime is caller_supplied


# ── re-entry / update discipline ────────────────────────────────────


def test_attach_runtime_session_runtime_idempotent_before_run():
    pipeline = _minimal_pipeline()
    first = _HostRuntime(creature_state=1)
    second = _HostRuntime(creature_state=2)
    pipeline.attach_runtime(session_runtime=first)
    pipeline.attach_runtime(session_runtime=second)
    assert pipeline._attached_session_runtime is second


@pytest.mark.asyncio
async def test_attach_runtime_session_runtime_refused_after_run():
    pipeline = _minimal_pipeline()
    pipeline._init_state(None)
    with pytest.raises(RuntimeError, match="attach_runtime"):
        pipeline.attach_runtime(session_runtime=_HostRuntime())


# ── independence from other kwargs ─────────────────────────────────


def test_session_runtime_does_not_affect_llm_client_resolution():
    """Attaching a session_runtime must not perturb the existing
    llm_client / stage-slot paths."""
    pipeline = _minimal_pipeline()
    pipeline.attach_runtime(session_runtime=_HostRuntime())
    state = pipeline._init_state(None)
    # No api stage, no llm_client attached → still None
    assert state.llm_client is None
    assert state.session_runtime is not None
