"""An OpenAI reasoning model has to be spoken to on the right wire.

OpenAI serves the same model over two APIs, and on Chat Completions a
reasoning model refuses function tools outright:

    Function tools with reasoning_effort are not supported for
    gpt-5.6-terra in /v1/chat/completions. To use function tools, use
    /v1/responses or set reasoning_effort to 'none'.

This was found in production, on an account that was configured correctly
and could not take a single turn — an agent always carries tools. The client
sends those turns to the Responses API, where tools and reasoning both fit.

What these tests hold down is the routing, in both directions: a reasoning
model with tools must leave on the Responses wire carrying its reasoning,
and everything else — plain models, and the compatible servers that have no
``/responses`` at all — must stay on Chat Completions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.openai import OpenAIClient, OpenAIResponsesClient
from geny_executor.llm_client.openai_compatible import OllamaClient
from geny_executor.llm_client.vllm import VLLMClient


def _sse(*events: Dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


_TOOL_REPLY = _sse(
    {"type": "response.output_item.added",
     "item": {"id": "i1", "type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "item_id": "i1", "delta": '{"file_path": "/a"}'},
    {"type": "response.output_item.done",
     "item": {"id": "i1", "type": "function_call", "call_id": "call_1", "name": "Read",
              "arguments": '{"file_path": "/a"}'}},
    {"type": "response.completed", "response": {"model": "gpt-5.6-terra", "usage": {}}},
)

_TOOLS = [{"name": "Read", "description": "read a file",
           "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}]


def _request(client: OpenAIClient, model: str = "gpt-5.6-terra", tools: Any = _TOOLS):
    return client._build_request(
        model_config=ModelConfig(model=model, thinking_enabled=True, thinking_budget_tokens=30000),
        messages=[{"role": "user", "content": "hi"}],
        system="be brief",
        tools=tools,
        tool_choice=None,
        stream=True,
    )


class TestWhichWire:
    def test_reasoning_model_with_tools_goes_to_responses(self) -> None:
        client = OpenAIClient(api_key="sk-test")
        assert client._prefers_responses(_request(client)) is True

    def test_ordinary_model_stays_on_chat_completions(self) -> None:
        client = OpenAIClient(api_key="sk-test")
        assert client._prefers_responses(_request(client, model="gpt-4o")) is False

    def test_no_tools_stays_on_chat_completions(self) -> None:
        """Without tools the combination is legal, and Chat Completions is
        the wire the rest of this client is built around."""
        client = OpenAIClient(api_key="sk-test")
        assert client._prefers_responses(_request(client, tools=None)) is False

    @pytest.mark.parametrize("client", [
        VLLMClient(api_key="x", base_url="http://localhost:8000/v1"),
        OllamaClient(base_url="http://localhost:11434/v1"),
    ], ids=["vllm", "ollama"])
    def test_compatible_servers_never_get_sent_to_responses(self, client: OpenAIClient) -> None:
        """They serve ``/chat/completions`` and nothing else. A model named
        ``gpt-5-something`` on a local server must not be routed off it."""
        assert client._prefers_responses(_request(client, model="gpt-5-local")) is False

    def test_a_base_url_override_is_left_alone(self) -> None:
        """Pointed at a proxy or a gateway, this client speaks whatever that
        endpoint was configured for — we do not know it has ``/responses``."""
        client = OpenAIClient(api_key="sk-test", base_url="https://gateway.internal/v1")
        assert client._prefers_responses(_request(client)) is False

    def test_openai_base_url_still_routes(self) -> None:
        client = OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")
        assert client._prefers_responses(_request(client)) is True


class TestRoutedTurn:
    """The whole point: tools AND reasoning survive the trip."""

    @pytest.mark.asyncio
    async def test_stream_carries_tools_and_reasoning_and_returns_tool_use(self) -> None:
        sent: List[Dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            assert request.url.path.endswith("/responses")
            assert request.headers["Authorization"] == "Bearer sk-test"
            return httpx.Response(200, content=_TOOL_REPLY)

        client = OpenAIClient(api_key="sk-test")
        client._responses = OpenAIResponsesClient(
            api_key="sk-test", transport=httpx.MockTransport(handler))

        chunks = []
        async for chunk in client.create_message_stream(
            model_config=ModelConfig(model="gpt-5.6-terra", thinking_enabled=True,
                                     thinking_budget_tokens=30000),
            messages=[{"role": "user", "content": "hi"}],
            system="be brief",
            tools=_TOOLS,
        ):
            chunks.append(chunk)

        body = sent[0]
        assert [t["name"] for t in body["tools"]] == ["Read"]
        assert body["reasoning"]["effort"] == "high"   # 30k budget → high

        response = chunks[-1]["response"]
        calls = [b for b in response.content if b.type == "tool_use"]
        assert [(c.tool_name, c.tool_input) for c in calls] == [("Read", {"file_path": "/a"})]
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_non_streaming_send_takes_the_same_route(self) -> None:
        client = OpenAIClient(api_key="sk-test")
        client._responses = OpenAIResponsesClient(
            api_key="sk-test",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=_TOOL_REPLY)))
        response = await client._send(_request(client))
        assert any(b.type == "tool_use" for b in response.content)


class TestEffort:
    """A Stage 8 thinking budget means the same thing on either wire."""

    @pytest.mark.parametrize("budget,expected", [(2000, "low"), (10000, "medium"), (30000, "high")])
    def test_budget_maps_through_the_canonical_translator(self, budget: int, expected: str) -> None:
        wire = OpenAIResponsesClient(api_key="sk-test")
        client = OpenAIClient(api_key="sk-test")
        request = client._build_request(
            model_config=ModelConfig(model="gpt-5.6-terra", thinking_enabled=True,
                                     thinking_budget_tokens=budget),
            messages=[{"role": "user", "content": "hi"}], system="", tools=_TOOLS,
            tool_choice=None, stream=True)
        assert wire._effort_for(request) == expected

    def test_thinking_off_sends_no_reasoning_block(self) -> None:
        wire = OpenAIResponsesClient(api_key="sk-test")
        client = OpenAIClient(api_key="sk-test")
        request = client._build_request(
            model_config=ModelConfig(model="gpt-5.6-terra"),
            messages=[{"role": "user", "content": "hi"}], system="", tools=_TOOLS,
            tool_choice=None, stream=True)
        assert "reasoning" not in wire._body(request)


class TestCredentialChanges:
    def test_configure_drops_the_delegate(self) -> None:
        """The delegate holds a copy of the key. Rotating credentials on the
        owner must not leave the old ones answering on the other wire."""
        client = OpenAIClient(api_key="sk-old")
        first = client._responses_client()
        client.configure(api_key="sk-new")
        second = client._responses_client()
        assert second is not first
        assert second._auth_headers()["Authorization"] == "Bearer sk-new"
