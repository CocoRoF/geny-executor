"""API providers — concrete implementations for API calls."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s06_api.interface import APIProvider
from geny_executor.stages.s06_api.types import APIRequest, APIResponse, ContentBlock


class AnthropicProvider(APIProvider):
    """Real Anthropic API provider using the official SDK."""

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers
        self._client: Optional[Any] = None

    def configure(self, config: Dict[str, Any]) -> None:
        if "api_key" in config:
            self._api_key = config["api_key"] or ""
            self._client = None
        if "base_url" in config:
            self._base_url = config["base_url"]
            self._client = None
        if "default_headers" in config:
            self._default_headers = config["default_headers"]
            self._client = None

    def get_config(self) -> Dict[str, Any]:
        return {
            "api_key": "***" if self._api_key else "",
            "base_url": self._base_url,
            "default_headers": self._default_headers or {},
        }

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def description(self) -> str:
        return "Anthropic Messages API via official SDK"

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

    async def create_message(self, request: APIRequest) -> APIResponse:
        """Call Anthropic Messages API."""
        client = self._get_client()

        kwargs = self._build_kwargs(request)
        try:
            raw_response = await client.messages.create(**kwargs)
            return self._parse_response(raw_response)
        except Exception as e:
            raise self._classify_error(e) from e

    async def create_message_stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        """Streaming call to Anthropic Messages API.

        Uses client.messages.stream() (high-level SDK helper) which handles
        stream=True internally.  Do NOT pass stream=True in kwargs — the
        method does not accept it and raises TypeError.
        """
        client = self._get_client()
        kwargs = self._build_kwargs(request)

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield {"type": "text_delta", "text": text}

                # Final message with full structured response
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
                        type="text", text=block.text, raw={"type": "text", "text": block.text}
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

    def _convert_stream_event(self, event: Any) -> Dict[str, Any]:
        event_type = getattr(event, "type", str(type(event).__name__))
        data: Dict[str, Any] = {"type": event_type}

        if hasattr(event, "delta"):
            delta = event.delta
            if hasattr(delta, "text"):
                data["text"] = delta.text
            if hasattr(delta, "type"):
                data["delta_type"] = delta.type

        if hasattr(event, "index"):
            data["index"] = event.index

        if hasattr(event, "content_block"):
            block = event.content_block
            data["content_block"] = {
                "type": getattr(block, "type", "unknown"),
            }
            if hasattr(block, "id"):
                data["content_block"]["id"] = block.id
            if hasattr(block, "name"):
                data["content_block"]["name"] = block.name

        return data

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


class MockProvider(APIProvider):
    """Mock provider for testing — returns predefined responses."""

    def __init__(
        self, responses: Optional[List[APIResponse]] = None, default_text: str = "Mock response"
    ):
        self._responses = list(responses or [])
        self._default_text = default_text
        self._call_count = 0
        self._call_history: List[APIRequest] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def description(self) -> str:
        return "Mock provider for testing"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> List[APIRequest]:
        return self._call_history

    def add_response(self, response: APIResponse) -> None:
        """Add a response to the queue."""
        self._responses.append(response)

    async def create_message(self, request: APIRequest) -> APIResponse:
        self._call_history.append(request)
        self._call_count += 1

        if self._responses:
            return self._responses.pop(0)

        return APIResponse(
            content=[ContentBlock(type="text", text=self._default_text)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            model=request.model,
            message_id=f"mock_{self._call_count}",
        )

    async def create_message_stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        """Mock streaming — yields text word-by-word, then final message."""
        response = await self.create_message(request)
        # Extract text from first text block (skip tool_use blocks)
        text = ""
        for block in response.content:
            if block.type == "text" and block.text:
                text = block.text
                break
        if text:
            for word in text.split(" "):
                yield {"type": "text_delta", "text": word + " "}
        yield {"type": "message_complete", "response": response}


class RecordingProvider(APIProvider):
    """Records real API calls for replay testing."""

    def __init__(self, inner: APIProvider):
        self._inner = inner
        self._recordings: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def description(self) -> str:
        return "Records API calls for replay"

    @property
    def recordings(self) -> List[Dict[str, Any]]:
        return self._recordings

    async def create_message(self, request: APIRequest) -> APIResponse:
        response = await self._inner.create_message(request)
        self._recordings.append(
            {
                "request": {
                    "model": request.model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens,
                    "system": request.system,
                },
                "response": {
                    "text": response.text,
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
                "timestamp": time.time(),
            }
        )
        return response

    def save_recordings(self, path: str) -> None:
        """Save recordings to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._recordings, f, indent=2, ensure_ascii=False)
