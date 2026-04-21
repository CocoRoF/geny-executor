"""Provider-name → client-class lookup with lazy imports.

Each adapter's vendor SDK is optional — ``AnthropicClient`` is the only
client whose SDK is a hard dependency of geny-executor. Others are
lazily imported so a user installing only the anthropic extras is not
forced to pip-install ``google-genai`` or ``openai``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from geny_executor.llm_client.base import BaseClient


class ClientRegistry:
    """Provider-name → client-class lookup."""

    _factories: Dict[str, Callable[[], Type[BaseClient]]] = {}

    @classmethod
    def register(cls, provider: str, factory: Callable[[], Type[BaseClient]]) -> None:
        cls._factories[provider] = factory

    @classmethod
    def get(cls, provider: str) -> Type[BaseClient]:
        if provider not in cls._factories:
            raise ValueError(
                f"Unknown LLM client provider: {provider!r}. Registered: {sorted(cls._factories)}"
            )
        return cls._factories[provider]()

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._factories)


def _anthropic_factory() -> Type[BaseClient]:
    from geny_executor.llm_client.anthropic import AnthropicClient

    return AnthropicClient


def _openai_factory() -> Type[BaseClient]:
    try:
        from geny_executor.llm_client.openai import OpenAIClient
    except ImportError as e:
        raise ImportError(
            "OpenAI client requires the 'openai' package. "
            "Install with: pip install geny-executor[openai]"
        ) from e
    return OpenAIClient


def _google_factory() -> Type[BaseClient]:
    try:
        from geny_executor.llm_client.google import GoogleClient
    except ImportError as e:
        raise ImportError(
            "Google client requires 'google-genai'. Install with: pip install geny-executor[google]"
        ) from e
    return GoogleClient


def _vllm_factory() -> Type[BaseClient]:
    from geny_executor.llm_client.vllm import VLLMClient

    return VLLMClient


ClientRegistry.register("anthropic", _anthropic_factory)
ClientRegistry.register("openai", _openai_factory)
ClientRegistry.register("google", _google_factory)
ClientRegistry.register("vllm", _vllm_factory)
