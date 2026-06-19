"""Tests for ClientRegistry — provider-name → client-class lookup."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.llm_client import BaseClient, ClientRegistry


def test_available_lists_four_builtins():
    names = set(ClientRegistry.available())
    assert {"anthropic", "openai", "google", "vllm"} <= names


def test_available_lists_branded_local_providers():
    names = set(ClientRegistry.available())
    # Profile-driven OpenAI-compatible local backends + the custom alias.
    assert {"ollama", "lmstudio", "custom", "local"} <= names


def test_get_anthropic_returns_class():
    cls = ClientRegistry.get("anthropic")
    assert issubclass(cls, BaseClient)
    assert getattr(cls, "provider", None) == "anthropic"


def test_get_vllm_returns_class():
    cls = ClientRegistry.get("vllm")
    assert issubclass(cls, BaseClient)
    assert cls.provider == "vllm"


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError) as ei:
        ClientRegistry.get("nonexistent")
    assert "nonexistent" in str(ei.value)
    assert "anthropic" in str(ei.value)  # lists registered names


def test_register_custom_provider():
    # NB: ``custom`` is now a real built-in provider (the generic
    # OpenAI-compatible local backend), so this host-extension test uses a
    # clearly-fake name it can safely register + pop without clobbering a
    # shipped factory.
    class _Custom(BaseClient):
        provider = "host_probe_provider"

        async def _send(self, request, *, purpose=""):
            raise NotImplementedError

    ClientRegistry.register("host_probe_provider", lambda: _Custom)
    try:
        cls = ClientRegistry.get("host_probe_provider")
        assert cls is _Custom
    finally:
        ClientRegistry._factories.pop("host_probe_provider", None)


def test_vllm_requires_base_url():
    cls = ClientRegistry.get("vllm")
    with pytest.raises(ValueError):
        cls(api_key="EMPTY")


def test_vllm_with_base_url_ok():
    cls = ClientRegistry.get("vllm")
    client = cls(api_key="EMPTY", base_url="http://localhost:8000/v1")
    assert client.provider == "vllm"
    assert client._base_url == "http://localhost:8000/v1"
