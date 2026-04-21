"""Tests for Stage.resolve_model_config — full ModelConfig resolution."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.core.stage import Stage


class _StubStage(Stage):
    def __init__(self, name: str = "stub", order: int = 99) -> None:
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


def test_resolve_no_override_returns_state_defaults():
    state = PipelineState(model="claude-sonnet-4-6", max_tokens=4096, temperature=0.3)
    stage = _StubStage()
    cfg = stage.resolve_model_config(state)
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.max_tokens == 4096
    assert cfg.temperature == 0.3


def test_resolve_override_wins_completely():
    state = PipelineState(model="claude-sonnet-4-6", max_tokens=64000, temperature=0.7)
    stage = _StubStage()
    stage.model_override = ModelConfig(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        temperature=0.0,
    )
    cfg = stage.resolve_model_config(state)
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 2048
    assert cfg.temperature == 0.0
    assert cfg is stage.model_override


def test_resolve_thinking_fields_from_state_when_no_override():
    state = PipelineState(
        thinking_enabled=True,
        thinking_budget_tokens=5000,
        thinking_type="adaptive",
    )
    stage = _StubStage()
    cfg = stage.resolve_model_config(state)
    assert cfg.thinking_enabled is True
    assert cfg.thinking_budget_tokens == 5000
    assert cfg.thinking_type == "adaptive"


def test_resolve_thinking_fields_from_override():
    state = PipelineState(thinking_enabled=True, thinking_budget_tokens=10000)
    stage = _StubStage()
    stage.model_override = ModelConfig(
        model="claude-haiku-4-5-20251001",
        thinking_enabled=False,
    )
    cfg = stage.resolve_model_config(state)
    assert cfg.thinking_enabled is False


def test_legacy_resolve_model_returns_string():
    state = PipelineState(model="claude-sonnet-4-6")
    stage = _StubStage()
    assert stage.resolve_model(state) == "claude-sonnet-4-6"
    stage.model_override = ModelConfig(model="claude-haiku-4-5-20251001")
    assert stage.resolve_model(state) == "claude-haiku-4-5-20251001"


def test_resolve_stop_sequences_copied_not_shared():
    state = PipelineState(stop_sequences=["END"])
    stage = _StubStage()
    cfg = stage.resolve_model_config(state)
    assert cfg.stop_sequences == ["END"]
    cfg.stop_sequences.append("STOP")
    assert state.stop_sequences == ["END"]
