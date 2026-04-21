"""Anthropic Messages API client.

Near-verbatim port of the former :class:`AnthropicProvider` in
``stages/s06_api/artifact/default/providers.py``, restructured to
inherit from :class:`BaseClient` and expose a :class:`ClientCapabilities`
profile.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


class AnthropicClient(BaseClient):
    """Real Anthropic API client using the official SDK."""

    provider = "anthropic"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
        supports_system_prompt=True,
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
            import anthropic

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        try:
            raw_response = await client.messages.create(**kwargs)
            return self._parse_response(raw_response)
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
        """Streaming call via the SDK's high-level ``messages.stream()`` helper.

        NOTE: do not pass ``stream=True`` in kwargs — that helper handles it.
        """
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

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield {"type": "text_delta", "text": text}

                final = await stream.get_final_message()
                yield {
                    "type": "message_complete",
                    "response": self._parse_response(final),
                }
        except Exception as e:
            raise self._classify_error(e) from e

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
        }

        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            kwargs["top_k"] = request.top_k
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        if request.thinking:
            kwargs["thinking"] = request.thinking
        if request.metadata:
            kwargs["metadata"] = request.metadata

        return kwargs

    def _parse_response(self, raw: Any) -> APIResponse:
        content_blocks: List[ContentBlock] = []

        for block in raw.content:
            if block.type == "text":
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=block.text,
                        raw={"type": "text", "text": block.text},
                    )
                )
            elif block.type == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                        raw={
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        },
                    )
                )
            elif block.type == "thinking":
                content_blocks.append(
                    ContentBlock(
                        type="thinking",
                        thinking_text=block.thinking,
                        raw={"type": "thinking", "thinking": block.thinking},
                    )
                )

        if raw.usage:
            usage = TokenUsage(
                input_tokens=getattr(raw.usage, "input_tokens", 0),
                output_tokens=getattr(raw.usage, "output_tokens", 0),
                cache_creation_input_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0),
                cache_read_input_tokens=getattr(raw.usage, "cache_read_input_tokens", 0),
            )
        else:
            usage = TokenUsage()

        return APIResponse(
            content=content_blocks,
            stop_reason=raw.stop_reason or "",
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=raw,
        )

    def _classify_error(self, e: Exception) -> APIError:
        import anthropic

        if isinstance(e, anthropic.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, anthropic.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, anthropic.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, anthropic.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, anthropic.BadRequestError):
            msg = str(e).lower()
            if "token" in msg or "context" in msg:
                return APIError(
                    str(e), category=ErrorCategory.TOKEN_LIMIT, status_code=400, cause=e
                )
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, status_code=400, cause=e)
        if isinstance(e, anthropic.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
