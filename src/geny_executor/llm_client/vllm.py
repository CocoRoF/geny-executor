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
        supports_structured_output=False,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=False,
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

        The declared ``drops`` tuple is interpreted against the upgraded
        flags: the drop application skips any field whose matching
        ``supports_*`` flag is True on the instance (see
        ``BaseClient._apply_declared_drops``), so the opt-in above
        really does let ``tools`` / ``tool_choice`` reach the server —
        no need to rewrite ``drops`` here.
        """
        # type-ignore: dataclasses.replace stubs can't type heterogeneous
        # **kwargs against the per-field types; values are validated by
        # the dataclass itself at construction.
        self.capabilities = replace(self.capabilities, **overrides)  # type: ignore[arg-type]
