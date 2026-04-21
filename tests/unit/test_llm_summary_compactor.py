"""Tests for LLMSummaryCompactor — gated on override + client."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.llm_client import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.stages.s02_context.artifact.default.compactors import (
    LLMSummaryCompactor,
    SummaryCompactor,
)


class _FakeClient(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities(supports_thinking=True)

    def __init__(self, text: str = "SUMMARY", **kwargs):
        super().__init__(**kwargs)
        self._text = text
        self.calls: list = []

    async def _send(self, request, *, purpose=""):
        raise RuntimeError("unused — create_message is overridden")

    async def create_message(self, **kwargs):
        self.calls.append(kwargs)
        return APIResponse(
            content=[ContentBlock(type="text", text=self._text)],
            stop_reason="end_turn",
            model=kwargs["model_config"].model,
        )


class _BoomClient(_FakeClient):
    async def create_message(self, **kwargs):
        raise RuntimeError("boom")


def _state_with_messages(n: int) -> PipelineState:
    s = PipelineState()
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        s.messages.append({"role": role, "content": f"msg {i}"})
    return s


@pytest.mark.asyncio
async def test_no_override_falls_back_to_placeholder():
    state = _state_with_messages(25)
    state.llm_client = _FakeClient("LLM SUMMARY")
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="claude-sonnet-4-6"),
        has_override=lambda: False,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(state)
    assert len(state.messages) == 10 + 2
    assert "[Summary of 15 previous messages" in state.messages[0]["content"]


@pytest.mark.asyncio
async def test_override_and_client_triggers_llm_call():
    state = _state_with_messages(25)
    client = _FakeClient("REAL SUMMARY")
    state.llm_client = client
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="claude-haiku-4-5-20251001", max_tokens=512),
        has_override=lambda: True,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(state)
    assert state.messages[0]["content"] == "REAL SUMMARY"
    assert len(client.calls) == 1
    assert client.calls[0]["purpose"] == "s02.compact"
    events = [e for e in state.events if e["type"] == "memory.compaction.summarized"]
    assert len(events) == 1
    assert events[0]["data"]["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_client_failure_falls_back_to_placeholder():
    state = _state_with_messages(25)
    state.llm_client = _BoomClient("never")
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="claude-haiku-4-5-20251001"),
        has_override=lambda: True,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(state)
    assert "[Summary of 15 previous messages" in state.messages[0]["content"]
    assert any(e["type"] == "memory.compaction.llm_failed" for e in state.events)


@pytest.mark.asyncio
async def test_below_keep_recent_is_noop():
    state = _state_with_messages(5)
    original = list(state.messages)
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="x"),
        has_override=lambda: True,
        client_getter=lambda s: _FakeClient(),
    )
    await comp.compact(state)
    assert state.messages == original


@pytest.mark.asyncio
async def test_no_client_falls_back_to_placeholder():
    state = _state_with_messages(25)
    state.llm_client = None
    comp = LLMSummaryCompactor(
        keep_recent=10,
        resolve_cfg=lambda s: ModelConfig(model="x"),
        has_override=lambda: True,
        client_getter=lambda s: s.llm_client,
    )
    await comp.compact(state)
    assert "[Summary of 15 previous messages" in state.messages[0]["content"]


def test_summary_compactor_still_usable_by_name():
    """Legacy SummaryCompactor class is not removed."""
    comp = SummaryCompactor(keep_recent=5)
    assert comp.name == "summary"
