"""OpenAI Chat Completions API provider.

Translates between geny-executor canonical format and OpenAI's API:
  - Canonical messages (Anthropic-style) ↔ OpenAI messages
  - Canonical tools ↔ OpenAI function tools
  - Canonical APIResponse ← OpenAI ChatCompletion
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s06_api.interface import APIProvider
from geny_executor.stages.s06_api.types import APIRequest, APIResponse, ContentBlock
from geny_executor.stages.s06_api._translate import (
    canonical_messages_to_openai,
    canonical_tools_to_openai,
    canonical_tool_choice_to_openai,
    canonical_thinking_to_openai,
    normalize_stop_reason,
)


class OpenAIProvider(APIProvider):
    """OpenAI Chat Completions API provider.

    Requires: pip install geny-executor[openai]
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def description(self) -> str:
        return "OpenAI Chat Completions API"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires the 'openai' package. "
                    "Install with: pip install geny-executor[openai]"
                )
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ── Non-streaming ──

    async def create_message(self, request: APIRequest) -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        try:
            raw = await client.chat.completions.create(**kwargs)
            return self._parse_response(raw)
        except Exception as e:
            raise self._classify_error(e) from e

    # ── Streaming ──

    async def create_message_stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        """Streaming call — yields text_delta events then message_complete."""
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        model = request.model
        finish_reason = ""
        usage_data: Optional[Any] = None

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    # Usage-only chunk (stream_options)
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, "model") and chunk.model:
                    model = chunk.model

                # Text delta
                if delta and delta.content:
                    accumulated_content += delta.content
                    yield {"type": "text_delta", "text": delta.content}

                # Tool call deltas
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = accumulated_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

        except Exception as e:
            raise self._classify_error(e) from e

        # Build final APIResponse
        blocks: List[ContentBlock] = []
        if accumulated_content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=accumulated_content,
                    raw={"type": "text", "text": accumulated_content},
                )
            )
        for tc in accumulated_tool_calls.values():
            try:
                tool_input = json.loads(tc["arguments"])
            except (json.JSONDecodeError, TypeError):
                tool_input = {}
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["id"],
                    tool_name=tc["name"],
                    tool_input=tool_input,
                    raw={
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tool_input,
                    },
                )
            )

        usage = TokenUsage()
        if usage_data:
            usage = TokenUsage(
                input_tokens=getattr(usage_data, "prompt_tokens", 0),
                output_tokens=getattr(usage_data, "completion_tokens", 0),
            )

        response = APIResponse(
            content=blocks,
            stop_reason=normalize_stop_reason(finish_reason, "openai"),
            usage=usage,
            model=model,
        )
        yield {"type": "message_complete", "response": response}

    # ── Request building ──

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        """Canonical APIRequest → OpenAI Chat Completions kwargs."""
        messages = canonical_messages_to_openai(request.messages, request.system)

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        # max_tokens
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        # Sampling
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        # top_k: not supported by OpenAI — silently ignored

        # Stop sequences
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        # Tools
        if request.tools:
            kwargs["tools"] = canonical_tools_to_openai(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = canonical_tool_choice_to_openai(request.tool_choice)

        # Reasoning (o-series models)
        if request.thinking:
            effort = canonical_thinking_to_openai(request.thinking)
            if effort:
                kwargs["reasoning_effort"] = effort

        return kwargs

    # ── Response parsing ──

    def _parse_response(self, raw: Any) -> APIResponse:
        """OpenAI ChatCompletion → Canonical APIResponse."""
        choice = raw.choices[0]
        blocks: List[ContentBlock] = []

        # Text content
        if choice.message.content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=choice.message.content,
                    raw={"type": "text", "text": choice.message.content},
                )
            )

        # Tool calls
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tc.id,
                        tool_name=tc.function.name,
                        tool_input=tool_input,
                        raw={
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tool_input,
                        },
                    )
                )

        # Token usage
        usage = TokenUsage(
            input_tokens=getattr(raw.usage, "prompt_tokens", 0),
            output_tokens=getattr(raw.usage, "completion_tokens", 0),
        )

        # Stop reason
        stop_reason = normalize_stop_reason(choice.finish_reason or "", "openai")

        return APIResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=raw,
        )

    # ── Error classification ──

    def _classify_error(self, e: Exception) -> APIError:
        try:
            import openai
        except ImportError:
            return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)

        if isinstance(e, openai.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, openai.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, openai.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, openai.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, openai.BadRequestError):
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, status_code=400, cause=e)
        if isinstance(e, openai.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
