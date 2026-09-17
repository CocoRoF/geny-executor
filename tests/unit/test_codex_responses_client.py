"""A ChatGPT plan spoken to over the Responses API, without a `codex` binary.

Signing in to Codex yields OAuth tokens for
``https://chatgpt.com/backend-api/codex``, which speaks the OpenAI Responses
API. This client drives it directly so the pipeline keeps the tool loop:
native ``function_call`` items become canonical ``tool_use`` blocks and
Stage 10 executes them like any other backend's.

Two things here are easy to get wrong and expensive to get wrong:

 · The refresh token is single-use. If the client rotates it and does not
   report the new pair back to the host, the account is logged out on the
   next turn — with no error until then.
 · Anthropic's ``cache_control`` markers are an Anthropic extension. Sent
   here they are a 400 on every call.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import ErrorCategory, APIError
from geny_executor.llm_client.codex import CodexResponsesClient


def _sse(*events: Dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


_TEXT_REPLY = _sse(
    {"type": "response.output_text.delta", "delta": "Hello"},
    {"type": "response.output_text.delta", "delta": " there"},
    {"type": "response.completed", "response": {
        "model": "gpt-5", "usage": {"input_tokens": 12, "output_tokens": 3,
                                    "input_tokens_details": {"cached_tokens": 4}}}},
)

_TOOL_REPLY = _sse(
    {"type": "response.output_item.added",
     "item": {"id": "i1", "type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "item_id": "i1", "delta": '{"file_path"'},
    {"type": "response.function_call_arguments.delta", "item_id": "i1", "delta": ': "/a"}'},
    {"type": "response.output_item.done",
     "item": {"id": "i1", "type": "function_call", "call_id": "call_1", "name": "Read",
              "arguments": '{"file_path": "/a"}'}},
    {"type": "response.completed", "response": {"model": "gpt-5", "usage": {}}},
)


def _client(handler, **kwargs: Any) -> CodexResponsesClient:
    return CodexResponsesClient(
        tokens={"access_token": "at", "refresh_token": "rt"},
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def _drain(client: CodexResponsesClient, **kwargs: Any):
    chunks = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="gpt-5"),
        messages=kwargs.pop("messages", [{"role": "user", "content": "hi"}]),
        **kwargs,
    ):
        chunks.append(chunk)
    return chunks


class TestWire:
    @pytest.mark.asyncio
    async def test_text_streams_and_usage_separates_cached_tokens(self) -> None:
        chunks = await _drain(_client(lambda r: httpx.Response(200, content=_TEXT_REPLY)))
        assert "".join(c["text"] for c in chunks if c["type"] == "text_delta") == "Hello there"
        usage = chunks[-1]["response"].usage
        assert (usage.input_tokens, usage.cache_read_input_tokens) == (8, 4)

    @pytest.mark.asyncio
    async def test_a_function_call_becomes_a_canonical_tool_use(self) -> None:
        chunks = await _drain(_client(lambda r: httpx.Response(200, content=_TOOL_REPLY)))
        response = chunks[-1]["response"]
        block = response.content[0]
        assert (block.type, block.tool_name, block.tool_input) == ("tool_use", "Read", {"file_path": "/a"})
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_the_request_carries_the_account_and_originator(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = dict(request.headers)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, content=_TEXT_REPLY)

        await _drain(_client(handler), system="You are Geny.",
                     tools=[{"name": "Read", "description": "d", "input_schema": {"type": "object"}}])
        assert seen["headers"]["originator"] == "geny_executor"
        assert seen["body"]["instructions"] == "You are Geny."
        assert seen["body"]["tools"][0]["name"] == "Read"
        assert seen["body"]["store"] is False

    @pytest.mark.asyncio
    async def test_anthropic_cache_markers_are_stripped(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["raw"] = request.content.decode()
            return httpx.Response(200, content=_TEXT_REPLY)

        await _drain(_client(handler), messages=[{"role": "user", "content": [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}])
        assert "cache_control" not in seen["raw"]

    @pytest.mark.asyncio
    async def test_tool_results_replay_as_function_call_output(self) -> None:
        seen: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["input"] = json.loads(request.content)["input"]
            return httpx.Response(200, content=_TEXT_REPLY)

        await _drain(_client(handler), messages=[
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "/a"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "body"}]},
        ])
        kinds = [item["type"] for item in seen["input"]]
        assert "function_call" in kinds and "function_call_output" in kinds


class TestTokens:
    @pytest.mark.asyncio
    async def test_a_401_refreshes_once_and_reports_the_rotated_pair(self) -> None:
        """A refresh token is single-use: losing the new one logs the
        account out on the next turn, silently."""
        notes: List[Dict[str, Any]] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "auth.openai.com":
                return httpx.Response(200, json={"access_token": "at2", "refresh_token": "rt2"})
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(401, json={"error": {"message": "expired"}})
            assert request.headers["authorization"] == "Bearer at2"
            return httpx.Response(200, content=_TEXT_REPLY)

        await _drain(_client(handler, notify=notes.append))
        rotated = [n for n in notes if n["kind"] == "tokens"]
        assert rotated and rotated[-1]["tokens"]["refresh_token"] == "rt2"

    @pytest.mark.asyncio
    async def test_an_account_with_no_token_says_so_as_auth(self) -> None:
        client = CodexResponsesClient(tokens={}, transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=_TEXT_REPLY)))
        with pytest.raises(APIError) as caught:
            await _drain(client)
        assert caught.value.category == ErrorCategory.AUTH


class TestErrors:
    @pytest.mark.asyncio
    async def test_a_usage_limit_is_classified_as_rate_limited(self) -> None:
        """So the router fails over to another account instead of raising
        at the user."""
        client = _client(lambda r: httpx.Response(
            429, json={"error": {"message": "You have hit your usage limit."}}))
        with pytest.raises(APIError) as caught:
            await _drain(client)
        assert caught.value.category == ErrorCategory.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_the_account_label_is_in_the_message(self) -> None:
        client = _client(lambda r: httpx.Response(500, json={"error": {"message": "boom"}}),
                         account_label="work plan")
        with pytest.raises(APIError) as caught:
            await _drain(client)
        assert "work plan" in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_failed_event_mid_stream_raises(self) -> None:
        body = _sse({"type": "response.failed",
                     "response": {"error": {"code": "server_error", "message": "upstream"}}})
        with pytest.raises(APIError) as caught:
            await _drain(_client(lambda r: httpx.Response(200, content=body)))
        assert caught.value.category == ErrorCategory.SERVER_ERROR
