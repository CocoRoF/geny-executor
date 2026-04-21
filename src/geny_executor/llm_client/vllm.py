"""vLLM client — thin subclass of :class:`OpenAIClient`.

vLLM exposes an OpenAI-compatible REST surface, so the bulk of the
adapter is identical; the differences are:

- ``provider = "vllm"``
- a required ``base_url`` (no public SaaS endpoint)
- conservative default capabilities (tool-calling depends on the
  serving model; override via :meth:`VLLMClient.configure_capabilities`
  if the deployed model supports it)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from geny_executor.llm_client.base import ClientCapabilities
from geny_executor.llm_client.openai import OpenAIClient


class VLLMClient(OpenAIClient):
    """vLLM client. Reuses the OpenAI SDK against a local ``base_url``."""

    provider = "vllm"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=False,
        supports_streaming=True,
        supports_tool_choice=False,
        supports_stop_sequences=True,
        supports_top_k=False,
        supports_system_prompt=True,
        drops=("thinking_enabled", "top_k", "tool_choice", "tools"),
    )

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        if not base_url:
            raise ValueError(
                "VLLMClient requires base_url (the vLLM server endpoint). "
                "Example: base_url='http://localhost:8000/v1'"
            )
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )

    def configure_capabilities(self, **overrides: bool) -> None:
        """Upgrade the client's capability flags when the deployed model supports them.

        Example: a vLLM instance running a tool-call-capable model can opt in::

            client = VLLMClient(base_url=...)
            client.configure_capabilities(supports_tools=True, supports_tool_choice=True)
        """
        self.capabilities = replace(self.capabilities, **overrides)
