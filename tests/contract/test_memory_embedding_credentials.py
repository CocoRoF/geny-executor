"""Embedding credentials flow through the CredentialBundle (audit §2.6).

The live 401-spam incident traced back to the embedding key living on
a parallel channel (config → env-var ladder) outside the
CredentialBundle the host already rotates. These tests pin the new
contract:

  - `MemoryProviderFactory(credentials=bundle)` sources the embedding
    api_key from `bundle.get('embedding')` whenever the embedding
    config didn't set one explicitly;
  - the bundle key **wins over env vars** (the deprecated ladder is the
    last resort, not a peer);
  - explicit config api_key still wins over the bundle (back-compat for
    hosts that already pass keys through config);
  - the env ladder logs its deprecation warning exactly once per
    process, naming the bundle channel;
  - bundle extras (`model` / `base_url`) fill config gaps.

Clients are constructed but never asked to embed — no network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import geny_executor.memory.embedding.client as embedding_client_mod
from geny_executor.llm_client.credentials import CredentialBundle, ProviderCredentials
from geny_executor.memory.factory import MemoryProviderFactory
from geny_executor.memory.providers import FileMemoryProvider


@pytest.fixture(autouse=True)
def _reset_env_ladder_warning(monkeypatch):
    """Each test starts with the one-time deprecation latch un-fired."""
    monkeypatch.setattr(embedding_client_mod, "_env_ladder_warned", False)


def _bundle(api_key: str = "bundle-key", **extras) -> CredentialBundle:
    return CredentialBundle(
        by_provider={
            "embedding": ProviderCredentials(api_key=api_key, extras=extras),
        }
    )


def _file_config(root: Path, embedding: dict) -> dict:
    return {"provider": "file", "root": str(root), "embedding": embedding}


# ── bundle as the credential source ─────────────────────────────────


def test_bundle_key_wins_over_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-lose")
    factory = MemoryProviderFactory(credentials=_bundle("bundle-key"))

    provider = factory.build(_file_config(tmp_path, {"provider": "openai"}))

    assert isinstance(provider, FileMemoryProvider)
    client = provider._embedding_client
    assert client._api_key == "bundle-key"
    assert client.descriptor.api_key_present is True


def test_explicit_config_key_wins_over_bundle(tmp_path: Path) -> None:
    """Hosts already passing api_key in the embedding config must not
    be silently re-keyed by a bundle — explicit config stays the most
    specific channel."""
    factory = MemoryProviderFactory(credentials=_bundle("bundle-key"))

    provider = factory.build(
        _file_config(tmp_path, {"provider": "openai", "api_key": "config-key"})
    )

    assert provider._embedding_client._api_key == "config-key"


def test_no_bundle_falls_back_to_env_with_one_deprecation_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    caplog.set_level(logging.WARNING, logger="geny_executor.memory.embedding.client")
    factory = MemoryProviderFactory()  # no credentials

    provider_a = factory.build(_file_config(tmp_path / "a", {"provider": "openai"}))
    provider_b = factory.build(_file_config(tmp_path / "b", {"provider": "openai"}))

    # The ladder still works (deployments keep running through 2.2.x)…
    assert provider_a._embedding_client._api_key == "env-key"
    assert provider_b._embedding_client._api_key == "env-key"
    # …but announces the migration exactly once, naming the channel.
    deprecations = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "DEPRECATED" in r.message
    ]
    assert len(deprecations) == 1, f"expected one deprecation, got {deprecations!r}"
    assert "CredentialBundle" in deprecations[0].message
    assert "embedding" in deprecations[0].message


def test_bundle_key_emits_no_deprecation_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """When the bundle supplies the key the env ladder is never
    consulted, so the deprecation warning must not fire."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    caplog.set_level(logging.WARNING, logger="geny_executor.memory.embedding.client")
    factory = MemoryProviderFactory(credentials=_bundle("bundle-key"))

    factory.build(_file_config(tmp_path, {"provider": "openai"}))

    assert not any("DEPRECATED" in r.message for r in caplog.records)


def test_env_warning_fires_once_even_for_direct_client_construction(
    monkeypatch, caplog
) -> None:
    """Direct (factory-less) client construction shares the same
    one-time latch — the warning is per-process, not per-call-site."""
    from geny_executor.memory.embedding.openai import OpenAIEmbeddingClient
    from geny_executor.memory.embedding.voyage import VoyageEmbeddingClient

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key-2")
    caplog.set_level(logging.WARNING, logger="geny_executor.memory.embedding.client")

    OpenAIEmbeddingClient()
    VoyageEmbeddingClient()

    deprecations = [r for r in caplog.records if "DEPRECATED" in r.message]
    assert len(deprecations) == 1


# ── bundle extras fill config gaps ──────────────────────────────────


def test_bundle_extras_supply_model(tmp_path: Path) -> None:
    factory = MemoryProviderFactory(
        credentials=_bundle("bundle-key", model="text-embedding-3-large")
    )

    provider = factory.build(_file_config(tmp_path, {"provider": "openai"}))

    descriptor = provider._embedding_client.descriptor
    assert descriptor.model == "text-embedding-3-large"
    assert descriptor.dimension == 3072


def test_config_model_wins_over_bundle_extras(tmp_path: Path) -> None:
    factory = MemoryProviderFactory(
        credentials=_bundle("bundle-key", model="text-embedding-3-large")
    )

    provider = factory.build(
        _file_config(
            tmp_path, {"provider": "openai", "model": "text-embedding-3-small"}
        )
    )

    assert provider._embedding_client.descriptor.model == "text-embedding-3-small"


def test_bundle_base_url_reaches_voyage_client(tmp_path: Path) -> None:
    factory = MemoryProviderFactory(
        credentials=CredentialBundle(
            by_provider={
                "embedding": ProviderCredentials(
                    api_key="bundle-key", base_url="https://proxy.example/v1/embeddings"
                ),
            }
        )
    )

    provider = factory.build(_file_config(tmp_path, {"provider": "voyage"}))

    assert provider._embedding_client._base_url == "https://proxy.example/v1/embeddings"
    assert provider._embedding_client._api_key == "bundle-key"


# ── behaviour preservation ──────────────────────────────────────────


def test_factory_without_credentials_is_unchanged(tmp_path: Path) -> None:
    """The credentials kwarg is additive: omitting it keeps the 2.1.x
    construction path byte-for-byte (local embedding, no key)."""
    factory = MemoryProviderFactory()
    provider = factory.build(
        _file_config(
            tmp_path, {"provider": "local", "model": "hash-v1", "dimension": 64}
        )
    )
    assert provider._embedding_client is not None
    assert provider._embedding_client.descriptor.provider == "local"


def test_bundle_without_embedding_entry_is_a_noop(tmp_path: Path, monkeypatch) -> None:
    """A bundle carrying only LLM-provider entries must not perturb
    embedding construction."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = MemoryProviderFactory(
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="llm-key")}
        )
    )

    provider = factory.build(_file_config(tmp_path, {"provider": "openai"}))

    assert provider._embedding_client._api_key == ""


def test_no_embedding_config_and_no_bundle_provider_builds_none(tmp_path: Path) -> None:
    """Embedding stays config-opt-in: a bundle key alone (no provider
    name anywhere) cannot conjure a vector layer."""
    factory = MemoryProviderFactory(credentials=_bundle("bundle-key"))
    provider = factory.build({"provider": "file", "root": str(tmp_path)})
    assert provider._embedding_client is None
    assert provider.vector() is None


def test_bundle_extras_provider_enables_bundle_only_construction(tmp_path: Path) -> None:
    """`extras['provider']` is the documented escape hatch for hosts
    that keep the entire embedding choice in the bundle."""
    factory = MemoryProviderFactory(
        credentials=_bundle("bundle-key", provider="openai")
    )
    provider = factory.build({"provider": "file", "root": str(tmp_path)})
    client = provider._embedding_client
    assert client is not None
    assert client.descriptor.provider == "openai"
    assert client._api_key == "bundle-key"
