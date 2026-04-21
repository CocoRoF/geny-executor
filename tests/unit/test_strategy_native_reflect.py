"""Tests for GenyMemoryStrategy native reflection path."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.llm_client import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.memory.strategy import GenyMemoryStrategy, ReflectionResolver


class _ScriptedClient(BaseClient):
    """Always returns JSON matching the GenyMemoryStrategy native prompt."""

    provider = "scripted"
    capabilities = ClientCapabilities(supports_thinking=True)

    def __init__(self, payload: dict, **kwargs):
        super().__init__(**kwargs)
        self._payload = payload
        self.calls: list = []

    async def _send(self, request, *, purpose=""):
        raise RuntimeError("unused")

    async def create_message(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self._payload)
        return APIResponse(
            content=[ContentBlock(type="text", text=text)],
            stop_reason="end_turn",
            model=kwargs["model_config"].model,
        )


class _FakeManager:
    def __init__(self):
        self.notes: list = []
        self.dated: list = []
        self.recorded: list = []

    def record_message(self, role, content):
        self.recorded.append((role, content))

    def remember_dated(self, summary):
        self.dated.append(summary)

    def write_note(self, **kwargs):
        self.notes.append(kwargs)
        return f"note_{len(self.notes)}.md"


def _state_with_conversation() -> PipelineState:
    state = PipelineState()
    state.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    state.final_text = "world"
    return state


@pytest.mark.asyncio
async def test_native_path_runs_when_callback_missing_and_override_set():
    payload = {
        "learned": [
            {
                "title": "T",
                "content": "C",
                "category": "insights",
                "tags": ["a"],
                "importance": "high",
            }
        ],
        "should_save": True,
    }
    mgr = _FakeManager()
    state = _state_with_conversation()
    client = _ScriptedClient(payload)
    state.llm_client = client

    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=None,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="claude-haiku-4-5-20251001", max_tokens=1024),
            has_override=lambda: True,
        ),
    )
    await strat._reflect(state)
    assert len(mgr.notes) == 1
    assert mgr.notes[0]["title"] == "T"
    assert len(client.calls) == 1
    events = [e for e in state.events if e["type"] == "memory.reflection.native"]
    assert len(events) == 1
    assert events[0]["data"]["model"] == "claude-haiku-4-5-20251001"
    assert events[0]["data"]["saved"] == 1


@pytest.mark.asyncio
async def test_native_path_skipped_when_no_override():
    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = _ScriptedClient({"learned": [], "should_save": False})
    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=None,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="x"),
            has_override=lambda: False,
        ),
    )
    await strat._reflect(state)
    assert mgr.notes == []
    assert state.metadata.get("needs_reflection") is True
    assert any(e["type"] == "memory.reflection_queued" for e in state.events)


@pytest.mark.asyncio
async def test_native_path_skipped_when_no_client():
    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = None
    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=None,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="x"),
            has_override=lambda: True,
        ),
    )
    await strat._reflect(state)
    assert mgr.notes == []
    assert state.metadata.get("needs_reflection") is True
    queued = [e for e in state.events if e["type"] == "memory.reflection_queued"]
    assert len(queued) == 1
    assert queued[0]["data"].get("reason") == "no_llm_client"


@pytest.mark.asyncio
async def test_callback_still_wins_when_both_available():
    calls: list = []

    async def cb(inp, out):
        calls.append((inp, out))
        return [
            {
                "title": "cb",
                "content": "via callback",
                "category": "insights",
                "tags": [],
                "importance": "medium",
            }
        ]

    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = _ScriptedClient({"learned": [], "should_save": False})
    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=cb,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="x"),
            has_override=lambda: True,
        ),
    )
    await strat._reflect(state)
    assert len(calls) == 1
    assert mgr.notes and mgr.notes[0]["title"] == "cb"


@pytest.mark.asyncio
async def test_should_save_false_records_no_insights():
    payload = {"learned": [], "should_save": False}
    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = _ScriptedClient(payload)

    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=None,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="claude-haiku-4-5-20251001"),
            has_override=lambda: True,
        ),
    )
    await strat._reflect(state)
    assert mgr.notes == []
    events = [e for e in state.events if e["type"] == "memory.reflection.native"]
    assert len(events) == 1
    assert events[0]["data"]["saved"] == 0


@pytest.mark.asyncio
async def test_json_parse_error_emits_llm_failed_event():
    class _BadClient(_ScriptedClient):
        async def create_message(self, **kwargs):
            return APIResponse(
                content=[ContentBlock(type="text", text="not json at all")],
                stop_reason="end_turn",
                model=kwargs["model_config"].model,
            )

    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = _BadClient({})

    strat = GenyMemoryStrategy(
        mgr,
        llm_reflect=None,
        resolver=ReflectionResolver(
            resolve_cfg=lambda s: ModelConfig(model="x"),
            has_override=lambda: True,
        ),
    )
    await strat._reflect(state)
    assert mgr.notes == []
    assert any(e["type"] == "memory.reflection.llm_failed" for e in state.events)


@pytest.mark.asyncio
async def test_no_resolver_uses_pre_cycle_behavior():
    """Back-compat: existing callers who don't pass a resolver still queue."""
    mgr = _FakeManager()
    state = _state_with_conversation()
    state.llm_client = _ScriptedClient({})

    strat = GenyMemoryStrategy(mgr, llm_reflect=None)
    await strat._reflect(state)
    assert mgr.notes == []
    assert state.metadata.get("needs_reflection") is True
