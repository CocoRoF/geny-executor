"""Tests for ProviderCredentials + CredentialBundle (Phase A2)."""

from __future__ import annotations

import sys
import os
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)


# ---------------------------------------------------------------------------
# ProviderCredentials
# ---------------------------------------------------------------------------


def test_provider_credentials_defaults_are_empty() -> None:
    c = ProviderCredentials()
    assert c.api_key == ""
    assert c.base_url is None
    assert c.default_headers is None
    assert c.binary_path is None
    assert c.extras == {}
    assert c.is_empty() is True


def test_provider_credentials_with_api_key_not_empty() -> None:
    c = ProviderCredentials(api_key="sk-xxx")
    assert c.is_empty() is False


def test_provider_credentials_with_base_url_not_empty() -> None:
    c = ProviderCredentials(base_url="http://localhost:8000/v1")
    assert c.is_empty() is False


def test_provider_credentials_with_binary_path_not_empty() -> None:
    c = ProviderCredentials(binary_path="/usr/local/bin/claude")
    assert c.is_empty() is False


def test_provider_credentials_with_extras_not_empty() -> None:
    c = ProviderCredentials(extras={"workspace_root": "/tmp/ws"})
    assert c.is_empty() is False


def test_provider_credentials_repr_redacts_api_key() -> None:
    c = ProviderCredentials(api_key="sk-supersecret", base_url="https://api.example.com")
    r = repr(c)
    assert "sk-supersecret" not in r
    assert "<redacted>" in r
    assert "https://api.example.com" in r


def test_provider_credentials_repr_empty_api_key_shows_empty() -> None:
    c = ProviderCredentials(base_url="https://x")
    r = repr(c)
    assert "<redacted>" not in r  # nothing to redact
    assert "''" in r


def test_provider_credentials_is_frozen() -> None:
    c = ProviderCredentials(api_key="x")
    with pytest.raises(FrozenInstanceError):
        c.api_key = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CredentialBundle
# ---------------------------------------------------------------------------


def test_bundle_get_missing_returns_empty_credentials() -> None:
    b = CredentialBundle()
    c = b.get("anthropic")
    assert isinstance(c, ProviderCredentials)
    assert c.is_empty() is True


def test_bundle_get_present_returns_value() -> None:
    creds = ProviderCredentials(api_key="sk-a")
    b = CredentialBundle(by_provider={"anthropic": creds})
    assert b.get("anthropic") is creds


def test_bundle_require_missing_raises_config_error() -> None:
    b = CredentialBundle()
    with pytest.raises(ConfigError, match="No credentials configured"):
        b.require("anthropic")


def test_bundle_require_empty_raises_config_error() -> None:
    b = CredentialBundle(by_provider={"anthropic": ProviderCredentials()})
    with pytest.raises(ConfigError):
        b.require("anthropic")


def test_bundle_require_present_returns_value() -> None:
    creds = ProviderCredentials(api_key="sk-a")
    b = CredentialBundle(by_provider={"anthropic": creds})
    assert b.require("anthropic") is creds


def test_bundle_has() -> None:
    creds = ProviderCredentials(api_key="sk-a")
    b = CredentialBundle(by_provider={
        "anthropic": creds,
        "openai": ProviderCredentials(),  # empty
    })
    assert b.has("anthropic") is True
    assert b.has("openai") is False
    assert b.has("google") is False


def test_bundle_providers_lists_non_empty_only() -> None:
    b = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-a"),
        "openai":    ProviderCredentials(),
        "vllm":      ProviderCredentials(base_url="http://x"),
        "google":    ProviderCredentials(),
    })
    assert b.providers() == ["anthropic", "vllm"]


def test_bundle_is_frozen() -> None:
    b = CredentialBundle()
    with pytest.raises(FrozenInstanceError):
        b.by_provider = {}  # type: ignore[misc]


def test_bundle_repr_does_not_leak_api_key() -> None:
    b = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-supersecret"),
    })
    assert "sk-supersecret" not in repr(b)
