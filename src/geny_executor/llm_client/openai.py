"""OpenAI Chat Completions API client.

Ported from the former :class:`OpenAIProvider` in
``stages/s06_api/artifact/openai/providers.py``. Translators are
imported from :mod:`geny_executor.llm_client.translators`, which
re-exports from the s06_api module during the PR-3→PR-4 bridge.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators import (
    canonical_messages_to_openai,
    canonical_thinking_to_openai,
    canonical_tool_choice_to_openai,
    canonical_tools_to_openai,
    normalize_stop_reason,
)
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


class OpenAIClient(BaseClient):
    """OpenAI Chat Completions API client.

    Requires: ``pip install geny-executor[openai]``
    """

    provider = "openai"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=False,
        supports_system_prompt=True,
        drops=("thinking_enabled", "top_k"),
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._client: Optional[Any] = None

    def configure(self, **kwargs: Any) -> None:
        super().configure(**kwargs)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI client requires the 'openai' package. "
                    "Install with: pip install geny-executor[openai]"
                ) from e
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        try:
            raw = await client.chat.completions.create(**kwargs)
            return self._parse_response(raw)
        except Exception as e:
            raise self._classify_error(e) from e

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
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
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
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, "model") and chunk.model:
                    model = chunk.model

                if delta and delta.content:
                    accumulated_content += delta.content
                    yield {"type": "text_delta", "text": delta.content}

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

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        """Canonical APIRequest → OpenAI Chat Completions kwargs."""
        messages = canonical_messages_to_openai(request.messages, request.system)

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = canonical_tools_to_openai(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = canonical_tool_choice_to_openai(request.tool_choice)

        if request.thinking:
            effort = canonical_thinking_to_openai(request.thinking)
            if effort:
                kwargs["reasoning_effort"] = effort

        return kwargs

    def _parse_response(self, raw: Any) -> APIResponse:
        choice = raw.choices[0]
        blocks: List[ContentBlock] = []

        if choice.message.content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=choice.message.content,
                    raw={"type": "text", "text": choice.message.content},
                )
            )

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

        usage = TokenUsage(
            input_tokens=getattr(raw.usage, "prompt_tokens", 0),
            output_tokens=getattr(raw.usage, "completion_tokens", 0),
        )
        stop_reason = normalize_stop_reason(choice.finish_reason or "", "openai")

        return APIResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=raw,
        )

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
