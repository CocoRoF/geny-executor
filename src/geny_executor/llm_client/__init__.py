"""Unified LLM client package — one surface, many vendors.

See :class:`BaseClient` for the per-vendor interface stage code should
target. See :class:`ClientRegistry` for provider-name lookup. Hosts inject
credentials via :class:`CredentialBundle` (built from
:class:`ProviderCredentials` entries).
"""

from geny_executor.llm_client._cli_runtime import (
    CLIProcessRunner,
    ContainerCLIRunner,
    SandboxHandle,
)
from geny_executor.llm_client.anthropic import AnthropicClient
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.claude_code import (
    ClaudeCodeCLIClient,
    build_container_cli_client,
)
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
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
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

__all__ = [
    "APIRequest",
    "APIResponse",
    "AnthropicClient",
    "BaseClient",
    "BUILTIN_PROFILES",
    "CLIProcessRunner",
    "ClaudeCodeCLIClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ConfigError",
    "ContainerCLIRunner",
    "ContentBlock",
    "CredentialBundle",
    "ProviderCredentials",
    "ProviderProfile",
    "SandboxHandle",
    "build_container_cli_client",
    "builtin_profiles",
    "probe_ollama_num_ctx",
    "resolve_local_context_window",
    "discover_models",
    "ModelDiscovery",
    "ModelInfo",
]
