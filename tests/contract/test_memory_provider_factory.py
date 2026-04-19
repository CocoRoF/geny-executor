"""Tests for `MemoryProviderFactory`.

The factory's job is config-in / provider-out. These tests pin down:
  - built-in dispatch for the four provider names ships out of the
    box (ephemeral, file, sql, composite);
  - error paths surface the right ValueError with a helpful message;
  - the composite builder reuses named sub-providers so two layers
    pointing at the same name share one underlying instance;
  - third-party builders register cleanly and override built-ins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from geny_executor.memory.composite import CompositeMemoryProvider
from geny_executor.memory.factory import MemoryProviderFactory
from geny_executor.memory.provider import Layer
from geny_executor.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)


# ── built-in dispatch ───────────────────────────────────────────────


def test_factory_lists_builtin_names():
    factory = MemoryProviderFactory()
    assert set(factory.names()) >= {"ephemeral", "file", "sql", "composite"}


def test_unknown_provider_raises():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="unknown memory provider"):
        factory.build({"provider": "nonexistent"})


def test_missing_provider_key_raises():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="non-empty string"):
        factory.build({"scope": "session"})


# ── ephemeral ───────────────────────────────────────────────────────


def test_build_ephemeral_returns_correct_class():
    factory = MemoryProviderFactory()
    provider = factory.build({"provider": "ephemeral"})
    assert isinstance(provider, EphemeralMemoryProvider)


def test_build_ephemeral_honours_scope():
    from geny_executor.memory.provider import Scope

    factory = MemoryProviderFactory()
    provider = factory.build({"provider": "ephemeral", "scope": "user"})
    assert provider.descriptor.scope == Scope.USER


# ── file ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_file_creates_initialisable_provider(tmp_path: Path):
    factory = MemoryProviderFactory()
    provider = factory.build({"provider": "file", "root": str(tmp_path / "fs")})
    assert isinstance(provider, FileMemoryProvider)
    await provider.initialize()
    assert (tmp_path / "fs").exists()


def test_build_file_requires_root():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="'root'"):
        factory.build({"provider": "file"})


@pytest.mark.asyncio
async def test_build_file_with_inline_embedding(tmp_path: Path):
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "file",
            "root": str(tmp_path / "fs"),
            "embedding": {"provider": "local", "model": "test", "dimension": 32},
        }
    )
    await provider.initialize()
    assert provider.vector() is not None
    assert provider.descriptor.embedding is not None
    assert provider.descriptor.embedding.dimension == 32


# ── sql ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_sql_creates_initialisable_provider(tmp_path: Path):
    factory = MemoryProviderFactory()
    provider = factory.build({"provider": "sql", "dsn": str(tmp_path / "main.db")})
    assert isinstance(provider, SQLMemoryProvider)
    await provider.initialize()


def test_build_sql_requires_dsn():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="'dsn'"):
        factory.build({"provider": "sql"})


# ── composite ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_composite_routes_named_subproviders(tmp_path: Path):
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "composite",
            "providers": {
                "stm_only": {"provider": "ephemeral"},
                "main": {"provider": "file", "root": str(tmp_path / "main")},
            },
            "layers": {
                "stm": "stm_only",
                "ltm": "main",
                "notes": "main",
                "index": "main",
            },
        }
    )
    assert isinstance(provider, CompositeMemoryProvider)
    await provider.initialize()
    distinct = provider.routing.distinct_providers()
    assert len(distinct) == 2  # one ephemeral + one file
    assert provider.routing.provider_for(Layer.STM) is not provider.routing.provider_for(Layer.LTM)


@pytest.mark.asyncio
async def test_build_composite_two_layers_share_named_provider(tmp_path: Path):
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "composite",
            "providers": {
                "main": {"provider": "sql", "dsn": str(tmp_path / "main.db")},
            },
            "layers": {
                "stm": "main",
                "ltm": "main",
                "notes": "main",
                "index": "main",
            },
        }
    )
    assert isinstance(provider, CompositeMemoryProvider)
    distinct = provider.routing.distinct_providers()
    assert len(distinct) == 1  # named "main" reused across all four layers


@pytest.mark.asyncio
async def test_build_composite_with_scope_provider(tmp_path: Path):
    from geny_executor.memory.provider import Scope

    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "composite",
            "providers": {
                "session_main": {"provider": "file", "root": str(tmp_path / "session")},
                "user_main": {"provider": "file", "root": str(tmp_path / "user")},
            },
            "layers": {
                "stm": "session_main",
                "ltm": "session_main",
                "notes": "session_main",
                "index": "session_main",
            },
            "scope_providers": {
                "user": "user_main",
            },
        }
    )
    assert isinstance(provider, CompositeMemoryProvider)
    await provider.initialize()
    user_prov = provider.routing.scope_provider(Scope.USER)
    assert user_prov is not None
    assert user_prov is not provider.routing.provider_for(Layer.NOTES)


def test_build_composite_unknown_layer_provider_name_raises(tmp_path: Path):
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="references unknown provider"):
        factory.build(
            {
                "provider": "composite",
                "providers": {
                    "main": {"provider": "ephemeral"},
                },
                "layers": {
                    "stm": "main",
                    "ltm": "ghost",
                    "notes": "main",
                    "index": "main",
                },
            }
        )


def test_build_composite_unknown_scope_provider_name_raises():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="references unknown provider"):
        factory.build(
            {
                "provider": "composite",
                "providers": {
                    "main": {"provider": "ephemeral"},
                },
                "layers": {
                    "stm": "main",
                    "ltm": "main",
                    "notes": "main",
                    "index": "main",
                },
                "scope_providers": {"user": "ghost"},
            }
        )


def test_build_composite_missing_providers_block_raises():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="non-empty 'providers'"):
        factory.build({"provider": "composite", "layers": {}})


def test_build_composite_missing_layers_block_raises():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="'layers' mapping"):
        factory.build(
            {
                "provider": "composite",
                "providers": {"main": {"provider": "ephemeral"}},
            }
        )


# ── third-party builder ─────────────────────────────────────────────


def test_register_custom_builder():
    sentinel = EphemeralMemoryProvider()

    def custom(_: MemoryProviderFactory, __: Mapping[str, Any]) -> EphemeralMemoryProvider:
        return sentinel

    factory = MemoryProviderFactory()
    factory.register("inproc-test", custom)
    assert factory.has("inproc-test")
    assert factory.build({"provider": "inproc-test"}) is sentinel


def test_register_overrides_builtin():
    sentinel = EphemeralMemoryProvider()

    def override(_: MemoryProviderFactory, __: Mapping[str, Any]) -> EphemeralMemoryProvider:
        return sentinel

    factory = MemoryProviderFactory()
    factory.register("ephemeral", override)
    assert factory.build({"provider": "ephemeral"}) is sentinel


def test_register_rejects_empty_name():
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError, match="non-empty string"):
        factory.register("", lambda *_: EphemeralMemoryProvider())


# ── embedding error path ────────────────────────────────────────────


def test_factory_rejects_non_mapping_embedding(tmp_path: Path):
    factory = MemoryProviderFactory()
    with pytest.raises(TypeError, match="embedding config must be a mapping"):
        factory.build(
            {
                "provider": "file",
                "root": str(tmp_path / "fs"),
                "embedding": "local",  # wrong type
            }
        )
