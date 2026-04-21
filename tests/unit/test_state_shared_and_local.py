"""Tests for PipelineState.shared and Stage.local_state helper."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.state import PipelineState
from geny_executor.core.stage import Stage


class _StubStage(Stage):
    """Minimal Stage subclass for testing helpers — does not execute."""

    def __init__(self, name: str, order: int) -> None:
        self._name = name
        self._order = order

    @property
    def name(self) -> str:
        return self._name

    @property
    def order(self) -> int:
        return self._order

    async def execute(self, input, state):
        return input


def test_shared_defaults_empty():
    state = PipelineState()
    assert state.shared == {}


def test_shared_is_isolated_per_state():
    s1 = PipelineState()
    s2 = PipelineState()
    s1.shared["x"] = 1
    assert s2.shared == {}


def test_shared_roundtrip():
    state = PipelineState()
    state.shared["context_summary"] = "hello"
    assert state.shared["context_summary"] == "hello"


def test_local_state_creates_on_first_access():
    state = PipelineState()
    stage = _StubStage(name="context", order=2)
    ls = stage.local_state(state)
    assert ls == {}
    ls["compacted_at_iteration"] = 3
    assert state.metadata["context"]["compacted_at_iteration"] == 3


def test_local_state_idempotent_on_repeat():
    state = PipelineState()
    stage = _StubStage(name="memory", order=15)
    first = stage.local_state(state)
    first["count"] = 7
    second = stage.local_state(state)
    assert second is first
    assert second["count"] == 7


def test_local_state_disjoint_between_stages():
    state = PipelineState()
    a = _StubStage(name="context", order=2)
    b = _StubStage(name="memory", order=15)
    a.local_state(state)["k"] = "A"
    b.local_state(state)["k"] = "B"
    assert a.local_state(state)["k"] == "A"
    assert b.local_state(state)["k"] == "B"


def test_shared_and_metadata_are_separate_buckets():
    state = PipelineState()
    state.shared["s"] = 1
    state.metadata["m"] = 2
    assert "s" not in state.metadata
    assert "m" not in state.shared


def test_local_state_does_not_touch_shared():
    state = PipelineState()
    stage = _StubStage(name="context", order=2)
    stage.local_state(state)["k"] = "v"
    assert state.shared == {}
