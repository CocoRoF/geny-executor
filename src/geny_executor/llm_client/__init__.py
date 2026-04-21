"""Unified LLM client package — one surface, many vendors.

See :class:`BaseClient` for the per-vendor interface stage code should
target. See :class:`ClientRegistry` for provider-name lookup.
"""

from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.bridge import ProviderBackedClient
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

__all__ = [
    "APIRequest",
    "APIResponse",
    "BaseClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ContentBlock",
    "ProviderBackedClient",
]
