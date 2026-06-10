"""Stage 6 full chunk forwarding + structured api.error (2.2.0).

Audit 2026-06-09 §3.2 / Tier 1-1, the monkey-patch killer:
``_call_streaming`` used to forward only ``text_delta`` and the
terminal ``message_complete`` — thinking deltas, tool_use starts and
input-json fragments the clients already yielded died inside the
stage, so both reference hosts monkey-patched it. Every canonical
chunk type now maps to a catalogued state event; these tests pin the
mapping with API-shaped and CLI-shaped fake clients.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from geny_executor import Pipeline, PipelineState
from geny_executor.core.errors import APIError, ErrorCategory, ExecutorErrorCode
from geny_executor.llm_client import APIRequest, APIResponse, BaseClient, ClientCapabilities
from geny_executor.llm_client.types import ContentBlock
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage
from geny_executor.stages.s21_yield import YieldStage


def _response(text: str = "done") -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=3, output_tokens=2),
        model="fake-model",
        message_id="msg_fake",
    )


class _ChunkClient(BaseClient):
    """Fake client whose stream yields a scripted canonical chunk
    sequence — the shape every real client's ``create_message_stream``
    produces (anthropic / openai / google / vllm and the CLI
    accumulator all share this vocabulary)."""

    provider = "fake"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
    )

    def __init__(self, chunks: List[Dict[str, Any]], *, subprocess_backed: bool = False):
        super().__init__(api_key="k")
        self._chunks = chunks
        if subprocess_backed:
            # Per-instance capability override mirroring claude_code's
            # class-level flag.
            self.capabilities = ClientCapabilities(
                supports_thinking=True,
                supports_tools=True,
                supports_streaming=True,
                is_subprocess=True,
            )

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return _response()

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        for chunk in self._chunks:
            yield chunk


FULL_CHUNK_SCRIPT: List[Dict[str, Any]] = [
    {"type": "thinking_delta", "text": "hmm "},
    {"type": "thinking_delta", "text": "ok"},
    {"type": "content_block_stop"},
    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}},
    {"type": "input_json_delta", "delta": '{"file_path": "/tmp'},
    {"type": "input_json_delta", "delta": '/x"}'},
    {"type": "content_block_stop"},
    {"type": "tool_result", "tool_use_id": "tu_1", "content": "data", "is_error": False},
    {"type": "text_delta", "text": "answer"},
    {"type": "message_complete", "response": _response("answer")},
]


def _stage_state(client: BaseClient) -> PipelineState:
    state = PipelineState(stream=True)
    state.llm_client = client
    state.add_message("user", "hi")
    return state


def _events_of(state: PipelineState, event_type: str) -> List[Dict[str, Any]]:
    return [e for e in state.events if e["type"] == event_type]


class TestChunkForwarding:
    @pytest.mark.asyncio
    async def test_api_client_full_chunk_set_maps_to_state_events(self):
        client = _ChunkClient(list(FULL_CHUNK_SCRIPT))
        state = _stage_state(client)

        response = await APIStage().execute(None, state)

        assert response.text == "answer"
        assert [e["data"]["text"] for e in _events_of(state, "thinking.delta")] == ["hmm ", "ok"]
        tool_uses = _events_of(state, "api.tool_use")
        assert len(tool_uses) == 1
        assert tool_uses[0]["data"] == {
            "id": "tu_1",
            "name": "Read",
            "input": {},
            "source": "api",
        }
        assert [e["data"]["delta"] for e in _events_of(state, "api.input_json_delta")] == [
            '{"file_path": "/tmp',
            '/x"}',
        ]
        assert len(_events_of(state, "api.content_block_stop")) == 2
        results = _events_of(state, "api.tool_result")
        assert len(results) == 1
        assert results[0]["data"]["tool_use_id"] == "tu_1"
        assert results[0]["data"]["is_error"] is False
        assert results[0]["data"]["source"] == "api"
        # API-backed client: tool_use is a Stage-10 dispatch request,
        # NOT a CLI-side execution — no companion event.
        assert _events_of(state, "api.cli_tool_call") == []

    @pytest.mark.asyncio
    async def test_subprocess_client_tool_use_marked_cli_with_companion_event(self):
        client = _ChunkClient(list(FULL_CHUNK_SCRIPT), subprocess_backed=True)
        state = _stage_state(client)

        await APIStage().execute(None, state)

        tool_uses = _events_of(state, "api.tool_use")
        assert tool_uses[0]["data"]["source"] == "cli"
        cli_calls = _events_of(state, "api.cli_tool_call")
        assert len(cli_calls) == 1
        assert cli_calls[0]["data"] == tool_uses[0]["data"]
        assert _events_of(state, "api.tool_result")[0]["data"]["source"] == "cli"

    @pytest.mark.asyncio
    async def test_empty_text_and_thinking_deltas_not_forwarded(self):
        """Keep-alive empty chunks stay invisible (matches the
        pre-2.2.0 text.delta guard)."""
        client = _ChunkClient(
            [
                {"type": "text_delta", "text": ""},
                {"type": "thinking_delta", "text": ""},
                {"type": "message_complete", "response": _response()},
            ]
        )
        state = _stage_state(client)
        await APIStage().execute(None, state)
        assert _events_of(state, "text.delta") == []
        assert _events_of(state, "thinking.delta") == []

    @pytest.mark.asyncio
    async def test_anthropic_shaped_stream_produces_thinking_delta_via_run_stream(self):
        """End-to-end: an API-provider-shaped client (the canonical
        chunk vocabulary the AnthropicClient contract documents)
        streams thinking through Pipeline.run_stream."""
        client = _ChunkClient(
            [
                {"type": "thinking_delta", "text": "pondering"},
                {"type": "text_delta", "text": "hello"},
                {"type": "message_complete", "response": _response("hello")},
            ]
        )
        pipeline = Pipeline()
        pipeline.register_stage(InputStage())
        pipeline.register_stage(APIStage())
        pipeline.register_stage(YieldStage())
        pipeline.attach_runtime(llm_client=client)

        types = []
        async for event in pipeline.run_stream("hi"):
            types.append(event.type)
            if event.type == "thinking.delta":
                assert event.data["text"] == "pondering"

        assert "thinking.delta" in types
        assert "text.delta" in types
        assert types[-1] == "pipeline.complete"


class TestApiErrorEnvelope:
    @pytest.mark.asyncio
    async def test_api_error_event_emitted_before_exception_propagates(self):
        class _FailingClient(_ChunkClient):
            provider = "claude_code_cli"

            async def create_message_stream(self, **kwargs):  # type: ignore[override]
                raise APIError(
                    "Claude Code CLI auth failed",
                    category=ErrorCategory.CLI_AUTH_FAILED,
                )
                yield  # pragma: no cover — make it an async generator

        client = _FailingClient([])
        client._cli_version_value = "2.1.149"
        state = _stage_state(client)

        with pytest.raises(Exception):  # StageError-wrapped at pipeline level; raw here
            await APIStage().execute(None, state)

        errors = _events_of(state, "api.error")
        assert len(errors) == 1
        data = errors[0]["data"]
        assert data["code"] == ExecutorErrorCode.EXEC_CLI_AUTH_FAILED.value
        assert data["category"] == ErrorCategory.CLI_AUTH_FAILED.value
        assert data["provider"] == "claude_code_cli"
        assert data["cli_version"] == "2.1.149"
        assert "auth failed" in data["message"]

    @pytest.mark.asyncio
    async def test_api_error_event_without_cli_version_omits_field(self):
        class _FailingClient(_ChunkClient):
            provider = "anthropic"

            async def create_message_stream(self, **kwargs):  # type: ignore[override]
                raise APIError("rate limited", category=ErrorCategory.RATE_LIMIT)
                yield  # pragma: no cover

        # NoRetry-equivalent: RATE_LIMIT is retryable by the default
        # strategy, so use the non-retrying stage config via retry arg.
        from geny_executor.stages.s06_api.artifact.default.retry import NoRetry

        state = _stage_state(_FailingClient([]))
        with pytest.raises(Exception):
            await APIStage(retry=NoRetry()).execute(None, state)

        errors = _events_of(state, "api.error")
        assert len(errors) == 1
        assert "cli_version" not in errors[0]["data"]
        assert errors[0]["data"]["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_api_error_visible_in_run_stream_before_pipeline_error(self):
        """Hosts' UIs read the structured envelope from the stream —
        it must arrive before the terminal pipeline.error."""

        class _FailingClient(_ChunkClient):
            provider = "claude_code_cli"

            async def create_message_stream(self, **kwargs):  # type: ignore[override]
                raise APIError(
                    "not authenticated",
                    category=ErrorCategory.CLI_AUTH_FAILED,
                )
                yield  # pragma: no cover

        pipeline = Pipeline()
        pipeline.register_stage(InputStage())
        pipeline.register_stage(APIStage())
        pipeline.register_stage(YieldStage())
        pipeline.attach_runtime(llm_client=_FailingClient([]))

        types = []
        api_error_data = None
        async for event in pipeline.run_stream("hi"):
            types.append(event.type)
            if event.type == "api.error":
                api_error_data = event.data

        assert api_error_data is not None
        assert api_error_data["code"] == ExecutorErrorCode.EXEC_CLI_AUTH_FAILED.value
        assert types.index("api.error") < types.index("pipeline.error")
