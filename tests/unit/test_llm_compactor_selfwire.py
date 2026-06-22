"""LLMSummaryCompactor self-wires from state.model (2.19.0)."""
import pytest
from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.artifact.default.compactors import (
    LLMSummaryCompactor,
)


class _Resp:
    text = "compressed recap preserving the facts"


class _Client:
    def __init__(self):
        self.calls = []
    async def create_message(self, *, model_config, messages, purpose=""):
        self.calls.append(model_config.model)
        return _Resp()


@pytest.mark.asyncio
async def test_selfwires_resolve_cfg_from_state_model():
    client = _Client()
    state = PipelineState(session_id="s1")
    state.model = "claude-haiku-4-5-20251001"
    state.llm_client = client
    state.messages = [{"role": "user", "content": f"m{i}"} for i in range(40)]
    c = LLMSummaryCompactor(keep_recent=5)  # no resolve_cfg → self-wire
    await c.compact(state)
    assert client.calls == ["claude-haiku-4-5-20251001"]   # used state.model
    # old messages collapsed (summary + the kept-recent tail) — far fewer than 40
    assert len(state.messages) < 40
    assert "compressed recap" in state.messages[0]["content"]


@pytest.mark.asyncio
async def test_falls_back_when_no_client():
    state = PipelineState(session_id="s2")
    state.model = "claude-haiku-4-5-20251001"
    state.llm_client = None
    state.messages = [{"role": "user", "content": f"m{i}"} for i in range(40)]
    c = LLMSummaryCompactor(keep_recent=5)
    await c.compact(state)  # no client → static fallback, must not raise
    assert len(state.messages) <= 40
