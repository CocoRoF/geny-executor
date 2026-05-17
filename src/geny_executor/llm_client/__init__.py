"""Unified LLM client package — one surface, many vendors.

See :class:`BaseClient` for the per-vendor interface stage code should
target. See :class:`ClientRegistry` for provider-name lookup. Hosts inject
credentials via :class:`CredentialBundle` (built from
:class:`ProviderCredentials` entries).
"""

from geny_executor.llm_client.anthropic import AnthropicClient
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient
from geny_executor.llm_client.copilot import CopilotCLIClient
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

__all__ = [
    "APIRequest",
    "APIResponse",
    "AnthropicClient",
    "BaseClient",
    "ClaudeCodeCLIClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ConfigError",
    "ContentBlock",
    "CopilotCLIClient",
    "CredentialBundle",
    "ProviderCredentials",
]
