"""vLLM client — thin subclass of :class:`OpenAIClient`.

vLLM exposes an OpenAI-compatible REST surface, so the bulk of the
adapter is identical; the differences are:

- ``provider = "vllm"``
- a required ``base_url`` (no public SaaS endpoint)
- conservative default capabilities (tool-calling depends on the
  serving model, so the deployment declares it: ``capabilities=`` at
  construction, or :meth:`BaseClient.configure_capabilities` later)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from geny_executor.llm_client.base import ClientCapabilities
from geny_executor.llm_client.openai import OpenAIClient


class VLLMClient(OpenAIClient):
    """vLLM client. Reuses the OpenAI SDK against a local ``base_url``."""

    provider = "vllm"
    #: A vLLM server implements Chat Completions only.
    speaks_responses = False
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=False,
        supports_streaming=True,
        supports_tool_choice=False,
        supports_stop_sequences=True,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=False,
        supports_session_continuity=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        requires_workspace=False,
        streaming_granularity="token",
        drops=("thinking_enabled", "top_k", "tool_choice", "tools"),
    )

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
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
        # What a vLLM server can do is a property of the model it loaded,
        # not of vLLM: the same endpoint class serves a tool-calling
        # Qwen-Coder and a plain text completion model. The defaults above
        # are the safe floor; the deployment that knows better says so.
        self.apply_capability_overrides(capabilities)
