"""Tests for BaseClient — capability filtering and request building."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


class _EchoClient(BaseClient):
    """Fake client that records the APIRequest built by _build_request."""

    def __init__(self, capabilities: ClientCapabilities, **kwargs) -> None:
        super().__init__(**kwargs)
        self.capabilities = capabilities
        self.last_request: APIRequest | None = None

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        self.last_request = request
        return APIResponse(
            content=[ContentBlock(type="text", text="ok")],
            stop_reason="end_turn",
            model=request.model,
        )


def _collect_events() -> tuple[list[dict], callable]:
    collected: list[dict] = []
    return collected, collected.append


@pytest.mark.asyncio
async def test_drops_thinking_when_not_supported():
    events, sink = _collect_events()
    client = _EchoClient(ClientCapabilities(supports_thinking=False), event_sink=sink)
    cfg = ModelConfig(model="test-model", thinking_enabled=True, thinking_budget_tokens=5000)
    response = await client.create_message(model_config=cfg, messages=[])
    assert client.last_request is not None
    assert client.last_request.thinking is None
    assert any(e.get("field") == "thinking_enabled" for e in events)
    assert response.text == "ok"


@pytest.mark.asyncio
async def test_keeps_thinking_when_supported():
    events, sink = _collect_events()
    client = _EchoClient(ClientCapabilities(supports_thinking=True), event_sink=sink)
    cfg = ModelConfig(
        model="test-model",
        thinking_enabled=True,
        thinking_type="enabled",
        thinking_budget_tokens=5000,
    )
    await client.create_message(model_config=cfg, messages=[])
    assert client.last_request.thinking == {"type": "enabled", "budget_tokens": 5000}
    assert events == []


@pytest.mark.asyncio
async def test_drops_top_k_when_not_supported():
    events, sink = _collect_events()
    client = _EchoClient(ClientCapabilities(supports_top_k=False), event_sink=sink)
    cfg = ModelConfig(model="test-model", top_k=40)
    await client.create_message(model_config=cfg, messages=[])
    assert client.last_request.top_k is None
    assert any(e.get("field") == "top_k" for e in events)


@pytest.mark.asyncio
async def test_keeps_top_k_when_supported():
    events, sink = _collect_events()
    client = _EchoClient(ClientCapabilities(supports_top_k=True), event_sink=sink)
    cfg = ModelConfig(model="test-model", top_k=40)
    await client.create_message(model_config=cfg, messages=[])
    assert client.last_request.top_k == 40
    assert events == []


@pytest.mark.asyncio
async def test_drops_tool_choice_when_not_supported():
    events, sink = _collect_events()
    client = _EchoClient(ClientCapabilities(supports_tool_choice=False), event_sink=sink)
    cfg = ModelConfig(model="test-model")
    await client.create_message(model_config=cfg, messages=[], tool_choice={"type": "auto"})
    assert any(e.get("field") == "tool_choice" for e in events)


@pytest.mark.asyncio
async def test_event_sink_none_is_safe():
    client = _EchoClient(ClientCapabilities(supports_thinking=False), event_sink=None)
    cfg = ModelConfig(model="test-model", thinking_enabled=True)
    await client.create_message(model_config=cfg, messages=[])


@pytest.mark.asyncio
async def test_builds_request_with_all_fields():
    events, sink = _collect_events()
    client = _EchoClient(
        ClientCapabilities(
            supports_thinking=True,
            supports_tools=True,
            supports_tool_choice=True,
            supports_top_k=True,
        ),
        event_sink=sink,
    )
    cfg = ModelConfig(
        model="m",
        max_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        stop_sequences=["END"],
        thinking_enabled=True,
        thinking_budget_tokens=8000,
        thinking_type="enabled",
    )
    await client.create_message(
        model_config=cfg,
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        tools=[{"name": "t"}],
        tool_choice={"type": "auto"},
    )
    req = client.last_request
    assert req.model == "m"
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.max_tokens == 2048
    assert req.temperature == 0.3
    assert req.top_p == 0.9
    assert req.top_k == 50
    assert req.stop_sequences == ["END"]
    assert req.system == "sys"
    assert req.tools == [{"name": "t"}]
    assert req.tool_choice == {"type": "auto"}
    assert req.thinking == {"type": "enabled", "budget_tokens": 8000}
    assert events == []
