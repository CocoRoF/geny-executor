"""PR-3→PR-4 bridge adapter — wrap an s06_api ``APIProvider`` as a :class:`BaseClient`.

The pipeline has historically constructed its own ``AnthropicProvider``
inside ``s06_api``. While PR-3 adds the first-class ``llm_client`` package
but does not yet delete the artifact directories, we need a way for
``state.llm_client`` to be non-None without double-constructing a vendor
SDK client.

:class:`ProviderBackedClient` forwards ``_send`` to the wrapped provider's
``create_message``. PR-4 deletes this module along with the ``APIProvider``
interface.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIRequest, APIResponse


class ProviderBackedClient(BaseClient):
    """Back a :class:`BaseClient` by an already-constructed s06_api ``APIProvider``."""

    def __init__(
        self,
        provider: Any,
        *,
        capabilities: Optional[ClientCapabilities] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(api_key="", base_url=None, event_sink=event_sink)
        self._wrapped = provider
        provider_name = getattr(provider, "name", None) or "bridge"
        self.provider = provider_name
        if capabilities is not None:
            self.capabilities = capabilities
        elif provider_name == "anthropic":
            self.capabilities = ClientCapabilities(
                supports_thinking=True,
                supports_tools=True,
                supports_streaming=True,
                supports_tool_choice=True,
                supports_stop_sequences=True,
                supports_top_k=True,
                supports_system_prompt=True,
            )
        elif provider_name == "openai":
            self.capabilities = ClientCapabilities(
                supports_thinking=False,
                supports_tools=True,
                supports_streaming=True,
                supports_tool_choice=True,
                supports_stop_sequences=True,
                supports_top_k=False,
                supports_system_prompt=True,
                drops=("thinking_enabled", "top_k"),
            )
        elif provider_name == "google":
            self.capabilities = ClientCapabilities(
                supports_thinking=False,
                supports_tools=True,
                supports_streaming=True,
                supports_tool_choice=True,
                supports_stop_sequences=True,
                supports_top_k=True,
                supports_system_prompt=True,
                drops=("thinking_enabled",),
            )

    @property
    def wrapped(self) -> Any:
        """The underlying :class:`APIProvider` instance (for tests / introspection)."""
        return self._wrapped

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
    ) -> AsyncIterator[Dict[str, Any]]:
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
