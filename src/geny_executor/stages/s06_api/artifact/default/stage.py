"""Stage 6: API — LLM call routed through ``state.llm_client``.

The stage no longer owns a vendor-provider strategy. It exposes a single
``provider`` config field (``anthropic`` / ``openai`` / ``google`` /
``vllm``) and delegates to the unified :class:`BaseClient` that lives on
``state.llm_client``. When no shared client is attached, the stage lazily
builds a local one via :class:`ClientRegistry` from its own
``provider`` / ``api_key`` / ``base_url`` fields.

For backward compatibility, ``provider=`` also accepts an
``APIProvider`` instance (the pre-PR-4 construction). In that case the
provider is wrapped once and stored; the PR-3 auto-bridge in
``Pipeline._resolve_llm_client`` produces an equivalent ``state.llm_client``
value and the execute path flows through the same unified surface.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from geny_executor.core.errors import APIError, ErrorCategory, ExecutorErrorCode
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.core.config import ModelConfig
from geny_executor.llm_client import BaseClient, ClientCapabilities, ClientRegistry
from geny_executor.stages.s06_api.interface import APIProvider, ModelRouter, RetryStrategy
from geny_executor.stages.s06_api.artifact.default.retry import (
    ExponentialBackoffRetry,
    NoRetry,
    RateLimitAwareRetry,
)
from geny_executor.stages.s06_api.artifact.default.router import (
    AdaptiveModelRouter,
    PassthroughRouter,
)
from geny_executor.stages.s06_api.types import APIRequest, APIResponse


class _LegacyProviderAdapter(BaseClient):
    """Test-only adapter that wraps a legacy :class:`APIProvider` as a
    :class:`BaseClient`. Production code never constructs this; it exists
    so that direct-construction tests (``APIStage(provider=MockProvider())``)
    keep working without a CredentialBundle. The capability flags are
    permissive so legacy provider fixtures aren't surprised by capability
    drops."""

    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
        supports_system_prompt=True,
    )

    def __init__(self, provider: APIProvider) -> None:
        super().__init__(api_key="", base_url=None)
        self._wrapped = provider
        self.provider = getattr(provider, "name", "") or "bridge"

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return await self._wrapped.create_message(request)

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ):
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        async for event in self._wrapped.create_message_stream(request):
            yield event


class APIStage(Stage[Any, APIResponse]):
    """Stage 6: API.

    Routes LLM calls through ``state.llm_client``. The host (production
    path) builds that client from the :class:`CredentialBundle` +
    ``config["provider"]`` inside ``Pipeline.from_manifest``.

    For direct-construction test fixtures, the constructor also accepts a
    legacy :class:`APIProvider` instance via ``provider=``. In that case
    the stage wraps it once in :class:`_LegacyProviderAdapter` and serves
    it via ``state.llm_client`` when the pipeline did not pre-populate one.
    """

    def __init__(
        self,
        provider: Union[str, APIProvider, None] = None,
        retry: Optional[RetryStrategy] = None,
        *,
        router: Optional[ModelRouter] = None,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        stream: bool = True,
        timeout_ms: Optional[int] = None,
    ):
        self._stream_default = stream
        self._timeout_ms = timeout_ms
        # Test-fixture conveniences. The production manifest path leaves
        # these empty; ``state.llm_client`` built from
        # ``Pipeline._resolve_llm_client`` carries the real credentials.
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = dict(default_headers) if default_headers else {}
        self._legacy_client: Optional[BaseClient] = None

        if isinstance(provider, str):
            self._provider_name = provider or "anthropic"
        elif provider is None:
            # No explicit provider — default to anthropic. If api_key was
            # passed, the stage can locally build a real client at
            # _resolve_client time.
            self._provider_name = "anthropic"
        else:
            # Legacy / test-only path: an APIProvider instance.
            self._provider_name = getattr(provider, "name", "") or "anthropic"
            self._legacy_client = _LegacyProviderAdapter(provider)

        self._slots: Dict[str, StrategySlot] = {
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
            "router": StrategySlot(
                name="router",
                strategy=router or PassthroughRouter(),
                registry={
                    "passthrough": PassthroughRouter,
                    "adaptive": AdaptiveModelRouter,
                },
                description="Adaptive model selection per call (passthrough = no override)",
            ),
        }

    @property
    def _retry(self) -> RetryStrategy:
        return self._slots["retry"].strategy  # type: ignore[return-value]

    @property
    def _router(self) -> ModelRouter:
        return self._slots["router"].strategy  # type: ignore[return-value]

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
        # Production providers come from ClientRegistry. "mock" is kept as
        # a schema-level option because round-trip serialization of test
        # fixtures (APIStage(provider=MockProvider())) records it as the
        # ``provider`` config value; strict schema validation would
        # otherwise reject the manifest.
        available = sorted(set(ClientRegistry.available()) | {"mock"})
        return ConfigSchema(
            name="api",
            fields=[
                ConfigField(
                    name="provider",
                    type="select",
                    label="Provider",
                    description="LLM provider to use for this stage.",
                    default="anthropic",
                    options=[{"value": p, "label": p} for p in available],
                ),
                ConfigField(
                    name="base_url",
                    type="string",
                    label="Base URL",
                    description="Override API endpoint (vLLM / proxy / mock server).",
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
            "provider": self._provider_name,
            "base_url": self._base_url or "",
            "stream": self._stream_default,
            "timeout_ms": self._timeout_ms or 0,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "provider" in config:
            new_name = str(config["provider"]) or "anthropic"
            if new_name != self._provider_name:
                self._provider_name = new_name
                # Invalidate any legacy/local client tied to the old provider.
                self._legacy_client = None
        if "base_url" in config:
            self._base_url = str(config["base_url"]) or None
        if "stream" in config:
            self._stream_default = bool(config["stream"])
        if "timeout_ms" in config:
            value = int(config["timeout_ms"])
            self._timeout_ms = value if value > 0 else None

    def _route_model(self, state: PipelineState) -> ModelConfig:
        """Resolve the baseline ModelConfig and pass it through the router slot.

        The default :class:`PassthroughRouter` returns ``None`` so this
        is identical to ``resolve_model_config(state)``. A custom router
        may return a swapped :class:`ModelConfig`; in that case we emit
        ``api.model_routed`` so observers can attribute cost/latency to
        the swap. The state is *not* mutated — the override only applies
        for this call.
        """
        cfg = self.resolve_model_config(state)
        try:
            override = self._router.route(cfg, state)
        except Exception as exc:
            state.add_event(
                "api.router.error",
                {"router": getattr(self._router, "name", ""), "error": str(exc)},
            )
            return cfg
        if override is None or override.model == cfg.model:
            return cfg
        state.add_event(
            "api.model_routed",
            {
                "router": getattr(self._router, "name", ""),
                "from": cfg.model,
                "to": override.model,
            },
        )
        return override

    def _resolve_stream(self, state: PipelineState) -> bool:
        state_stream = getattr(state, "stream", None)
        if state_stream is not None:
            return state_stream
        return self._stream_default

    def _resolve_client(self, state: PipelineState) -> BaseClient:
        """Return the effective :class:`BaseClient`.

        Preference:
          1. ``state.llm_client`` populated by ``Pipeline._init_state`` (the
             production manifest path) or ``Pipeline.attach_runtime``.
          2. The legacy adapter built in ``__init__`` when a test-fixture
             ``APIProvider`` was passed directly.
          3. A locally-built client from this stage's ``provider`` /
             ``api_key`` / ``base_url`` (test-fixture path: callers that
             constructed the stage with ``APIStage(api_key=...)`` rely on
             this).
          4. Raise — the stage cannot make an LLM call without a client.
        """
        if state.llm_client is not None:
            return state.llm_client
        if self._legacy_client is not None:
            state.llm_client = self._legacy_client
            return self._legacy_client
        if self._api_key or self._provider_name in ClientRegistry.available():
            try:
                client_cls = ClientRegistry.get(self._provider_name)
            except (KeyError, ValueError):
                client_cls = None
            if client_cls is not None:
                kwargs: Dict[str, Any] = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                if self._default_headers:
                    kwargs["default_headers"] = dict(self._default_headers)
                client = client_cls(**kwargs)
                state.llm_client = client
                return client
        raise APIError(
            "state.llm_client is None. Build the pipeline via "
            "Pipeline.from_manifest(credentials=...) or attach a client "
            "explicitly with Pipeline.attach_runtime(llm_client=...).",
            category=ErrorCategory.BAD_REQUEST,
            code=ExecutorErrorCode.EXEC_API_NO_CLIENT,
        )

    async def execute(self, input: Any, state: PipelineState) -> APIResponse:
        cfg = self._route_model(state)
        client = self._resolve_client(state)
        use_stream = self._resolve_stream(state)

        state.add_event(
            "api.request",
            {
                "model": cfg.model,
                "provider": getattr(client, "provider", ""),
                "message_count": len(state.messages),
                "has_tools": bool(state.tools),
                "has_thinking": cfg.thinking_enabled,
                "stream": use_stream,
            },
        )

        if use_stream:
            response = await self._call_streaming_with_retry(client, cfg, state)
        else:
            response = await self._call_with_retry(client, cfg, state)

        state.last_api_response = response

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
        """Assemble a canonical :class:`APIRequest` from state.

        Kept for introspection and legacy test fixtures; execute() no
        longer routes through this method (it calls ``client.create_message``
        which builds the request internally with capability filtering).
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
            thinking: Dict[str, Any] = {"type": cfg.thinking_type}
            if cfg.thinking_type == "enabled":
                thinking["budget_tokens"] = cfg.thinking_budget_tokens
            if cfg.thinking_display:
                thinking["display"] = cfg.thinking_display
            request.thinking = thinking
        return request

    # ── Retry wrappers ──

    def _call_kwargs(self, cfg: Any, state: PipelineState) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_config": cfg,
            "messages": list(state.messages),
            "purpose": "api",
        }
        if state.system:
            kwargs["system"] = state.system
        if state.tools:
            kwargs["tools"] = state.tools
        if state.tool_choice:
            kwargs["tool_choice"] = state.tool_choice
        return kwargs

    async def _call_with_retry(
        self, client: BaseClient, cfg: Any, state: PipelineState
    ) -> APIResponse:
        last_error: Optional[Exception] = None
        kwargs = self._call_kwargs(cfg, state)

        for attempt in range(self._retry.max_retries + 1):
            try:
                return await client.create_message(**kwargs)
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
                        "code": e.code.value,
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
                        "code": ExecutorErrorCode.from_category(category).value,
                        "delay": delay,
                    },
                )
                await asyncio.sleep(delay)

        raise last_error or APIError(
            "Max retries exceeded",
            category=ErrorCategory.UNKNOWN,
            code=ExecutorErrorCode.EXEC_API_RETRY_EXHAUSTED,
        )

    async def _call_streaming_with_retry(
        self, client: BaseClient, cfg: Any, state: PipelineState
    ) -> APIResponse:
        last_error: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                return await self._call_streaming(client, cfg, state)
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

        raise last_error or APIError(
            "Max retries exceeded",
            category=ErrorCategory.UNKNOWN,
            code=ExecutorErrorCode.EXEC_API_RETRY_EXHAUSTED,
        )

    async def _call_streaming(
        self, client: BaseClient, cfg: Any, state: PipelineState
    ) -> APIResponse:
        response: Optional[APIResponse] = None
        kwargs = self._call_kwargs(cfg, state)

        stream: AsyncIterator[Dict[str, Any]] = client.create_message_stream(**kwargs)
        async for chunk in stream:
            chunk_type = chunk.get("type")
            if chunk_type == "message_complete":
                response = chunk["response"]
            elif chunk_type == "text_delta" and chunk.get("text"):
                state.add_event("text.delta", {"text": chunk["text"]})

        if response is None:
            raise APIError(
                "Stream ended without message_complete",
                category=ErrorCategory.UNKNOWN,
                code=ExecutorErrorCode.EXEC_API_STREAM_INCOMPLETE,
            )
        return response

    # ── Response formatting ──

    def _build_assistant_content(self, response: APIResponse) -> List[Dict[str, Any]]:
        """Build assistant content for message history."""
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
