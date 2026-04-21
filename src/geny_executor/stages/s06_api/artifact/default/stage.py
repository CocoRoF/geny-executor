"""Stage 6: API — calls Anthropic Messages API."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.interface import APIProvider, RetryStrategy
from geny_executor.stages.s06_api.artifact.default.providers import (
    AnthropicProvider,
    MockProvider,
)
from geny_executor.stages.s06_api.artifact.default.retry import (
    ExponentialBackoffRetry,
    NoRetry,
    RateLimitAwareRetry,
)
from geny_executor.stages.s06_api.types import APIRequest, APIResponse


class APIStage(Stage[Any, APIResponse]):
    """Stage 6: API.

    Dual abstraction:
      - Level 2 provider: actual API call implementation
      - Level 2 retry: error recovery strategy

    Streaming is controlled by PipelineState.stream (set from PipelineConfig).
    Falls back to the constructor parameter when state does not specify.

    Both streaming and non-streaming paths share the same retry strategy:
    same model, up to max_retries attempts, then fail.
    """

    def __init__(
        self,
        provider: Optional[APIProvider] = None,
        retry: Optional[RetryStrategy] = None,
        *,
        api_key: str = "",
        base_url: Optional[str] = None,
        stream: bool = True,
        timeout_ms: Optional[int] = None,
    ):
        if provider is not None:
            initial_provider: APIProvider = provider
        elif api_key:
            initial_provider = AnthropicProvider(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("Either 'provider' or 'api_key' must be provided")

        self._slots: Dict[str, StrategySlot] = {
            "provider": StrategySlot(
                name="provider",
                strategy=initial_provider,
                registry={
                    "anthropic": AnthropicProvider,
                    "mock": MockProvider,
                },
                description="API provider (actual LLM endpoint)",
            ),
            "retry": StrategySlot(
                name="retry",
                strategy=retry or ExponentialBackoffRetry(),
                registry={
                    "exponential_backoff": ExponentialBackoffRetry,
                    "no_retry": NoRetry,
                    "rate_limit_aware": RateLimitAwareRetry,
                },
                description="Retry strategy on API errors",
            ),
        }
        self._stream_default = stream
        self._base_url = base_url
        self._timeout_ms = timeout_ms

    @property
    def _provider(self) -> APIProvider:
        return self._slots["provider"].strategy  # type: ignore[return-value]

    @property
    def _retry(self) -> RetryStrategy:
        return self._slots["retry"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "api"

    @property
    def order(self) -> int:
        return 6

    @property
    def category(self) -> str:
        return "execution"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="api",
            fields=[
                ConfigField(
                    name="base_url",
                    type="string",
                    label="Base URL",
                    description="Override API endpoint (useful for proxies or mocks).",
                    default="",
                ),
                ConfigField(
                    name="stream",
                    type="boolean",
                    label="Stream",
                    description="Use Server-Sent Events streaming when supported.",
                    default=True,
                    ui_widget="toggle",
                ),
                ConfigField(
                    name="timeout_ms",
                    type="integer",
                    label="Timeout (ms)",
                    description="Per-request timeout in milliseconds. Blank for provider default.",
                    default=0,
                    min_value=0,
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_url": self._base_url or "",
            "stream": self._stream_default,
            "timeout_ms": self._timeout_ms or 0,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "base_url" in config:
            self._base_url = str(config["base_url"]) or None
        if "stream" in config:
            self._stream_default = bool(config["stream"])
        if "timeout_ms" in config:
            value = int(config["timeout_ms"])
            self._timeout_ms = value if value > 0 else None

    def _resolve_stream(self, state: PipelineState) -> bool:
        """Resolve streaming mode: state (from PipelineConfig) takes precedence."""
        state_stream = getattr(state, "stream", None)
        if state_stream is not None:
            return state_stream
        return self._stream_default

    async def execute(self, input: Any, state: PipelineState) -> APIResponse:
        request = self._build_request(state)
        use_stream = self._resolve_stream(state)

        state.add_event(
            "api.request",
            {
                "model": request.model,
                "message_count": len(request.messages),
                "has_tools": bool(request.tools),
                "has_thinking": bool(request.thinking),
                "stream": use_stream,
            },
        )

        if use_stream:
            response = await self._call_streaming_with_retry(request, state)
        else:
            response = await self._call_with_retry(request, state)

        # Store raw response for downstream stages
        state.last_api_response = response

        # Add assistant message to conversation (always List[Dict])
        assistant_content = self._build_assistant_content(response)
        state.add_message("assistant", assistant_content)

        state.add_event(
            "api.response",
            {
                "stop_reason": response.stop_reason,
                "text_length": len(response.text),
                "tool_calls": len(response.tool_calls),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

        return response

    def _build_request(self, state: PipelineState) -> APIRequest:
        """Build APIRequest from pipeline state.

        Follows Anthropic API constraints:
          - Use EITHER temperature OR top_p, not both.
          - thinking.budget_tokens must be < max_tokens when type="enabled".

        Honors per-stage overrides via :meth:`Stage.resolve_model_config` so
        all sampling + thinking fields are resolved together (override wins
        verbatim when set; otherwise falls back to ``state``).
        """
        cfg = self.resolve_model_config(state)
        request = APIRequest(
            model=cfg.model,
            messages=list(state.messages),
            max_tokens=cfg.max_tokens,
            system=state.system,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            stop_sequences=cfg.stop_sequences,
        )

        if state.tools:
            request.tools = state.tools
        if state.tool_choice:
            request.tool_choice = state.tool_choice

        if cfg.thinking_enabled:
            thinking_type = cfg.thinking_type
            thinking: dict = {"type": thinking_type}
            if thinking_type == "enabled":
                thinking["budget_tokens"] = cfg.thinking_budget_tokens
            if cfg.thinking_display:
                thinking["display"] = cfg.thinking_display
            request.thinking = thinking

        return request

    # ── API call methods (both paths use the same retry strategy) ──

    async def _call_streaming_with_retry(
        self, request: APIRequest, state: PipelineState
    ) -> APIResponse:
        """Execute streaming API call with retry on recoverable errors.

        Same model, up to max_retries attempts. No model switching.
        On retry, previously streamed text.delta events are already emitted
        but the final response is discarded — only the successful attempt's
        response is returned.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                return await self._call_streaming(request, state)
            except APIError as e:
                last_error = e
                if not self._retry.should_retry(e.category, attempt):
                    raise
                delay = self._retry.get_delay(attempt)
                state.add_event(
                    "api.retry",
                    {
                        "attempt": attempt + 1,
                        "category": e.category.value,
                        "delay": delay,
                        "stream": True,
                    },
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                category = ErrorCategory.UNKNOWN
                if not self._retry.should_retry(category, attempt):
                    raise APIError(str(e), category=category, cause=e) from e
                delay = self._retry.get_delay(attempt)
                state.add_event(
                    "api.retry",
                    {
                        "attempt": attempt + 1,
                        "category": category.value,
                        "delay": delay,
                        "stream": True,
                    },
                )
                await asyncio.sleep(delay)

        raise last_error or APIError("Max retries exceeded", category=ErrorCategory.UNKNOWN)

    async def _call_streaming(self, request: APIRequest, state: PipelineState) -> APIResponse:
        """Single streaming attempt — emits text.delta events."""
        response: Optional[APIResponse] = None

        async for chunk in self._provider.create_message_stream(request):
            chunk_type = chunk.get("type")
            if chunk_type == "message_complete":
                response = chunk["response"]
            elif chunk_type == "text_delta" and chunk.get("text"):
                state.add_event("text.delta", {"text": chunk["text"]})

        if response is None:
            raise APIError(
                "Stream ended without message_complete",
                category=ErrorCategory.UNKNOWN,
            )
        return response

    async def _call_with_retry(self, request: APIRequest, state: PipelineState) -> APIResponse:
        """Execute non-streaming API call with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._provider.create_message(request)
                return response
            except APIError as e:
                last_error = e
                if not self._retry.should_retry(e.category, attempt):
                    raise
                delay = self._retry.get_delay(attempt)
                state.add_event(
                    "api.retry",
                    {
                        "attempt": attempt + 1,
                        "category": e.category.value,
                        "delay": delay,
                    },
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                category = ErrorCategory.UNKNOWN
                if not self._retry.should_retry(category, attempt):
                    raise APIError(str(e), category=category, cause=e) from e
                delay = self._retry.get_delay(attempt)
                state.add_event(
                    "api.retry",
                    {
                        "attempt": attempt + 1,
                        "category": category.value,
                        "delay": delay,
                    },
                )
                await asyncio.sleep(delay)

        raise last_error or APIError("Max retries exceeded", category=ErrorCategory.UNKNOWN)

    # ── Response formatting ──

    def _build_assistant_content(self, response: APIResponse) -> List[Dict[str, Any]]:
        """Build assistant content for message history.

        Always returns List[Dict] (Anthropic content blocks format)
        for consistent downstream processing.
        """
        blocks: List[Dict[str, Any]] = []
        for block in response.content:
            if block.raw:
                blocks.append(block.raw)
            elif block.type == "text":
                blocks.append({"type": "text", "text": block.text or ""})
            elif block.type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.tool_use_id,
                        "name": block.tool_name,
                        "input": block.tool_input,
                    }
                )
        return blocks
