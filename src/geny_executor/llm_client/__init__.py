"""Unified LLM client package — one surface, many vendors.

See :class:`BaseClient` for the per-vendor interface stage code should
target. See :class:`ClientRegistry` for provider-name lookup. Hosts inject
credentials via :class:`CredentialBundle` (built from
:class:`ProviderCredentials` entries).
"""

from geny_executor.llm_client._sandbox_handle import SandboxHandle
from geny_executor.llm_client.anthropic import AnthropicClient
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.claude_code_tokens import ClaudeCodeTokenClient
from geny_executor.llm_client.codex import CodexResponsesClient
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.llm_client.context_window import (
    DEFAULT_CONTEXT_WINDOW,
    binding_context_window,
    known_context_window,
    resolve_context_window,
)
from geny_executor.llm_client.local_probe import (
    probe_ollama_num_ctx,
    resolve_local_context_window,
)
from geny_executor.llm_client.model_discovery import (
    ModelDiscovery,
    ModelInfo,
    discover_models,
)
from geny_executor.llm_client.profiles import (
    BUILTIN_PROFILES,
    ProviderProfile,
    builtin_profiles,
)
from geny_executor.llm_client.registry import ROUTED_PROVIDERS, ClientRegistry
from geny_executor.llm_client.router import RouterClient
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

__all__ = [
    "APIRequest",
    "APIResponse",
    "AnthropicClient",
    "BaseClient",
    "BUILTIN_PROFILES",
    "ClaudeCodeTokenClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ConfigError",
    "CodexResponsesClient",
    "ContentBlock",
    "CredentialBundle",
    "ProviderCredentials",
    "ProviderProfile",
    "ROUTED_PROVIDERS",
    "RouterClient",
    "SandboxHandle",
    "builtin_profiles",
    "probe_ollama_num_ctx",
    "resolve_local_context_window",
    "DEFAULT_CONTEXT_WINDOW",
    "binding_context_window",
    "known_context_window",
    "resolve_context_window",
    "discover_models",
    "ModelDiscovery",
    "ModelInfo",
]
