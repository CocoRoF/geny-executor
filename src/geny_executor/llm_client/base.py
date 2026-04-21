"""Base class for every LLM client.

Implementations adapt a vendor SDK to the canonical :class:`APIRequest` /
:class:`APIResponse` shape. Every :class:`BaseClient` MUST:

- Accept a :class:`ModelConfig` + canonical messages and run the vendor
  call without the caller needing to know which vendor is in use.
- Drop unsupported fields rather than raising, emitting a
  ``llm_client.feature_unsupported`` event on ``event_sink`` if one was
  provided.
- Translate vendor exceptions into
  :class:`geny_executor.core.errors.APIError` with a populated
  :class:`ErrorCategory` so upstream retry/classify logic does not need
  to branch on vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.types import APIRequest, APIResponse


@dataclass(frozen=True)
class ClientCapabilities:
    """Feature flags a client advertises.

    Stage code inspects these before sending fields not every vendor
    supports. Unsupported fields are silently dropped and the client
    emits a ``llm_client.feature_unsupported`` event.
    """

    supports_thinking: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_tool_choice: bool = False
    supports_stop_sequences: bool = True
    supports_top_k: bool = False
    supports_system_prompt: bool = True

    #: Fields this client will silently drop when present on the request.
    drops: tuple[str, ...] = field(default=())


class BaseClient(ABC):
    """Abstract LLM client. Concrete subclasses live in this package."""

    #: Provider name (stable identifier used by :class:`ClientRegistry`).
    provider: str = ""

    #: Capabilities advertised by this client. Subclasses override.
    capabilities: ClientCapabilities = ClientCapabilities()

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers
        self._event_sink = event_sink

    # ── High-level surface used by stages ───────────────────────────────

    async def create_message(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> APIResponse:
        """Send a non-streaming request built from a :class:`ModelConfig`."""
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )
        return await self._send(request, purpose=purpose)

    async def create_message_stream(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming variant. Default: fall back to non-streaming.

        Concrete clients override to use vendor streams.
        """
        response = await self.create_message(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            purpose=purpose,
        )
        yield {"type": "message_complete", "response": response}

    # ── Low-level surface — kept for s06_api parity during PR-3→PR-4 bridge

    @abstractmethod
    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        """Send a pre-built :class:`APIRequest`. Subclass implements vendor call."""

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_request(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
        stream: bool,
    ) -> APIRequest:
        """Assemble a canonical :class:`APIRequest`.

        Emits ``llm_client.feature_unsupported`` events for any field in
        ``model_config`` that this client drops.
        """
        request = APIRequest(
            model=model_config.model,
            messages=list(messages),
            max_tokens=model_config.max_tokens,
            system=system,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k if self.capabilities.supports_top_k else None,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=(
                list(model_config.stop_sequences) if model_config.stop_sequences else None
            ),
            stream=stream,
        )

        if model_config.thinking_enabled:
            if self.capabilities.supports_thinking:
                thinking: Dict[str, Any] = {"type": model_config.thinking_type}
                if model_config.thinking_type == "enabled":
                    thinking["budget_tokens"] = model_config.thinking_budget_tokens
                if model_config.thinking_display:
                    thinking["display"] = model_config.thinking_display
                request.thinking = thinking
            else:
                self._emit_unsupported("thinking_enabled")

        if model_config.top_k is not None and not self.capabilities.supports_top_k:
            self._emit_unsupported("top_k")

        if tool_choice and not self.capabilities.supports_tool_choice:
            self._emit_unsupported("tool_choice")

        return request

    def _emit_unsupported(self, field_name: str) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            {
                "type": "llm_client.feature_unsupported",
                "provider": self.provider,
                "field": field_name,
            }
        )

    def configure(self, **kwargs: Any) -> None:
        """Apply provider-specific runtime configuration."""
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)
