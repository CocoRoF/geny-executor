"""Unit tests for the declarative provider-profile layer (P0-A-1/A-2)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client import ClientRegistry  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402
from geny_executor.llm_client.credentials import ProviderCredentials  # noqa: E402
from geny_executor.llm_client.profiles import (  # noqa: E402
    CUSTOM_PROFILE,
    OLLAMA_PROFILE,
    builtin_profiles,
    is_profiled_provider,
    profiled_client_kwargs,
    profiled_provider_names,
    resolve_profile,
)
from geny_executor.llm_client.types import APIRequest  # noqa: E402


# ── profile resolution ────────────────────────────────────────────────


def test_builtin_profiles_are_three():
    names = {p.name for p in builtin_profiles()}
    assert names == {"ollama", "lmstudio", "custom"}


def test_resolve_primary_and_alias():
    assert resolve_profile("ollama") is OLLAMA_PROFILE
    assert resolve_profile("custom") is CUSTOM_PROFILE
    # ``local`` is a declared alias of ``custom``.
    assert resolve_profile("local") is CUSTOM_PROFILE


def test_is_profiled_provider():
    assert is_profiled_provider("ollama")
    assert is_profiled_provider("local")
    assert not is_profiled_provider("anthropic")
    assert not is_profiled_provider("vllm")


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_profile("does-not-exist")


def test_profiled_provider_names_includes_alias():
    names = set(profiled_provider_names())
    assert {"ollama", "lmstudio", "custom", "local"} <= names


def test_profile_is_frozen_dataclass():
    with pytest.raises(Exception):
        OLLAMA_PROFILE.name = "mutated"  # type: ignore[misc]


# ── registry wiring ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["ollama", "lmstudio", "custom", "local"])
def test_registry_get_returns_baseclient_subclass(name):
    cls = ClientRegistry.get(name)
    assert issubclass(cls, BaseClient)


def test_registry_get_provider_names():
    assert ClientRegistry.get("ollama").provider == "ollama"
    assert ClientRegistry.get("lmstudio").provider == "lmstudio"
    # The ``local`` alias resolves to the custom client.
    assert ClientRegistry.get("local").provider == "custom"
    assert ClientRegistry.get("custom").provider == "custom"


# ── construction quirks ───────────────────────────────────────────────


def test_ollama_default_base_url():
    client = ClientRegistry.get("ollama")()
    assert client._base_url == "http://localhost:11434/v1"
    # keyless local server → AsyncOpenAI-safe placeholder key
    assert client._api_key == "EMPTY"


def test_lmstudio_default_base_url():
    client = ClientRegistry.get("lmstudio")()
    assert client._base_url == "http://127.0.0.1:1234/v1"


def test_custom_requires_base_url():
    with pytest.raises(ValueError):
        ClientRegistry.get("custom")()


def test_custom_with_base_url_ok():
    client = ClientRegistry.get("custom")(base_url="http://localhost:8080/v1")
    assert client.provider == "custom"
    assert client._base_url == "http://localhost:8080/v1"


def test_explicit_base_url_overrides_default():
    client = ClientRegistry.get("ollama")(base_url="http://gpu-box:11434/v1")
    assert client._base_url == "http://gpu-box:11434/v1"


# ── capabilities ──────────────────────────────────────────────────────


def test_local_caps_are_tool_capable_by_default():
    client = ClientRegistry.get("ollama")()
    assert client.supports("tools") is True
    assert client.supports("tool_choice") is True
    assert client.supports("thinking") is False
    assert client.supports("token_usage") is True


def test_configure_capabilities_downgrade():
    client = ClientRegistry.get("ollama")()
    client.configure_capabilities(supports_tools=False, supports_tool_choice=False)
    assert client.supports("tools") is False
    assert client.supports("tool_choice") is False


# ── _build_kwargs quirks: token floor + num_ctx/think extra_body ───────


def _req(max_tokens: int) -> APIRequest:
    return APIRequest(
        model="qwen2.5-coder:7b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
    )


def test_max_tokens_floor_applied_when_unset():
    client = ClientRegistry.get("ollama")()
    kwargs = client._build_kwargs(_req(max_tokens=0))
    # Floor prevents Ollama's num_predict=128 truncation.
    assert kwargs["max_tokens"] == OLLAMA_PROFILE.default_max_tokens


def test_explicit_small_max_tokens_is_respected():
    client = ClientRegistry.get("ollama")()
    kwargs = client._build_kwargs(_req(max_tokens=32))
    assert kwargs["max_tokens"] == 32


def test_num_ctx_and_think_extra_body():
    client = ClientRegistry.get("ollama")(num_ctx=32768, think=False)
    kwargs = client._build_kwargs(_req(max_tokens=256))
    assert kwargs["extra_body"]["options"]["num_ctx"] == 32768
    assert kwargs["extra_body"]["think"] is False


def test_no_extra_body_by_default():
    client = ClientRegistry.get("ollama")()
    kwargs = client._build_kwargs(_req(max_tokens=256))
    # Default request must be a plain OpenAI-compatible call.
    assert "extra_body" not in kwargs


# ── credential → kwargs mapping ───────────────────────────────────────


def test_profiled_client_kwargs_threads_base_url_and_knobs():
    creds = ProviderCredentials(
        api_key="",
        base_url="http://localhost:11434/v1",
        extras={"ollama_num_ctx": 16384, "think": False},
    )
    kwargs = profiled_client_kwargs("ollama", creds)
    assert kwargs["api_key"] == "EMPTY"
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["num_ctx"] == 16384
    assert kwargs["think"] is False


def test_profiled_client_kwargs_num_ctx_alias():
    creds = ProviderCredentials(base_url="http://x/v1", extras={"num_ctx": 4096})
    kwargs = profiled_client_kwargs("custom", creds)
    assert kwargs["num_ctx"] == 4096


def test_pipeline_creds_mapper_uses_profile_path():
    # The pipeline-level mapper must route profiled providers through the
    # profile kwargs (base_url + knobs), not the generic API-provider path.
    from geny_executor.core.pipeline import _creds_to_client_kwargs

    creds = ProviderCredentials(
        base_url="http://localhost:11434/v1", extras={"ollama_num_ctx": 8192}
    )
    kwargs = _creds_to_client_kwargs("ollama", creds)
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["num_ctx"] == 8192
    assert kwargs["api_key"] == "EMPTY"
