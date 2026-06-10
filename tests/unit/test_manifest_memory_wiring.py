"""2.2.0 Wave 3 — manifest ``memory`` block → built + wired provider.

Audit §1-1: memory provider construction was host-code-only. A
non-empty ``manifest.memory`` block now builds the provider via
``MemoryProviderFactory`` (``provider_from_manifest_memory``) inside
``Pipeline.from_manifest`` and wires it through ``_apply_runtime`` —
the exact slot path ``attach_runtime``'s memory kwargs use, so a host
that attaches runtime memory objects afterwards wins (runtime objects
beat declarations).
"""

from __future__ import annotations

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.core.state import PipelineState
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.memory.factory import provider_from_manifest_memory
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file import FileMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever
from geny_executor.memory.strategy import ProviderDrivenStrategy

from tests._fixtures.manifest_entries import required_stage_entries


def _manifest(*, memory=None) -> EnvironmentManifest:
    """Required stages + the two memory-wiring targets (s02 / s18)."""
    stages = required_stage_entries() + [
        StageManifestEntry(order=2, name="context", active=True).to_dict(),
        StageManifestEntry(order=18, name="memory", active=True).to_dict(),
    ]
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_mem", name="memory-wiring"),
        stages=stages,
        tools=ToolsSnapshot(),
        memory=dict(memory or {}),
    )


def _bundle() -> CredentialBundle:
    return CredentialBundle(
        by_provider={"anthropic": ProviderCredentials(api_key="sk-a")}
    )


def _slot_strategy(pipeline, stage_name, slot_name):
    stage = next(s for s in pipeline.stages if s.name == stage_name)
    return stage.get_strategy_slots()[slot_name].strategy


# ── provider_from_manifest_memory (the factory glue) ─────────


class TestProviderFromManifestMemory:
    def test_file_block_builds_file_provider(self, tmp_path):
        p = provider_from_manifest_memory(
            {"provider": "file", "config": {"root": str(tmp_path)}}
        )
        assert isinstance(p, FileMemoryProvider)

    def test_ephemeral_block_needs_no_config(self):
        p = provider_from_manifest_memory({"provider": "ephemeral"})
        assert isinstance(p, EphemeralMemoryProvider)

    def test_missing_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            provider_from_manifest_memory({"config": {"root": "/x"}})

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown memory provider"):
            provider_from_manifest_memory({"provider": "redis"})


# ── from_manifest builds + wires ─────────────────────────────


class TestFromManifestMemoryWiring:
    def test_provider_built_and_exposed(self, tmp_path):
        m = _manifest(memory={"provider": "file", "config": {"root": str(tmp_path)}})
        p = Pipeline.from_manifest(m, credentials=_bundle())
        assert isinstance(p.memory_provider, FileMemoryProvider)

    def test_slots_wired_like_attach_runtime(self, tmp_path):
        m = _manifest(memory={"provider": "file", "config": {"root": str(tmp_path)}})
        p = Pipeline.from_manifest(m, credentials=_bundle())
        retriever = _slot_strategy(p, "context", "retriever")
        strategy = _slot_strategy(p, "memory", "strategy")
        assert isinstance(retriever, MemoryAwareRetriever)
        assert isinstance(strategy, ProviderDrivenStrategy)

    def test_empty_block_leaves_pipeline_untouched(self):
        p = Pipeline.from_manifest(_manifest(), credentials=_bundle())
        assert p.memory_provider is None
        assert not isinstance(
            _slot_strategy(p, "context", "retriever"), MemoryAwareRetriever
        )

    @pytest.mark.asyncio
    async def test_end_to_end_file_backend_records_turns(self, tmp_path):
        """The manifest-built provider actually persists: drive the
        wired ProviderDrivenStrategy with a state and find the turn in
        the file backend under tmp_path."""
        m = _manifest(
            memory={
                "provider": "file",
                "config": {"root": str(tmp_path), "session_id": "s-e2e"},
            }
        )
        p = Pipeline.from_manifest(m, credentials=_bundle())
        strategy = _slot_strategy(p, "memory", "strategy")

        state = PipelineState(session_id="s-e2e")
        state.messages = [{"role": "user", "content": "remember me"}]
        await strategy.update(state)

        recent = await p.memory_provider.stm().recent(5)
        assert [t.content for t in recent] == ["remember me"]
        assert any(tmp_path.rglob("*.jsonl"))

    def test_host_attach_runtime_wins_over_manifest_block(self, tmp_path):
        """Documented precedence: runtime objects beat declarations."""

        class _HostRetriever:
            name = "host_retriever"

            async def retrieve(self, query, state):
                return []

        class _HostStrategy:
            name = "host_strategy"

            async def update(self, state):
                return None

        m = _manifest(memory={"provider": "file", "config": {"root": str(tmp_path)}})
        p = Pipeline.from_manifest(m, credentials=_bundle())
        host_retriever = _HostRetriever()
        host_strategy = _HostStrategy()
        p.attach_runtime(
            memory_retriever=host_retriever, memory_strategy=host_strategy
        )
        assert _slot_strategy(p, "context", "retriever") is host_retriever
        assert _slot_strategy(p, "memory", "strategy") is host_strategy

    def test_strict_build_raises_on_unbuildable_block(self):
        # 'file' without 'root' passes validate_manifest (root is a
        # config-shape concern, not a name check) but the factory
        # refuses — strict surfaces it as ConfigError at build time.
        m = _manifest(memory={"provider": "file"})
        with pytest.raises(ConfigError, match="manifest.memory could not be built"):
            Pipeline.from_manifest(m, credentials=_bundle())

    def test_lenient_build_drops_unbuildable_block_loudly(self, caplog):
        import logging

        m = _manifest(memory={"provider": "file"})
        with caplog.at_level(logging.WARNING, logger="geny_executor.core.pipeline"):
            p = Pipeline.from_manifest(m, credentials=_bundle(), strict=False)
        assert p.memory_provider is None
        assert "manifest.memory failed to build" in caplog.text

    def test_strict_build_refuses_unknown_provider_via_validation(self):
        m = _manifest(memory={"provider": "redis"})
        with pytest.raises(ConfigError, match="memory.unknown_provider"):
            Pipeline.from_manifest(m, credentials=_bundle())

    @pytest.mark.asyncio
    async def test_from_manifest_async_inherits_memory_wiring(self, tmp_path):
        m = _manifest(memory={"provider": "file", "config": {"root": str(tmp_path)}})
        p = await Pipeline.from_manifest_async(m, credentials=_bundle())
        try:
            assert isinstance(p.memory_provider, FileMemoryProvider)
        finally:
            await p.aclose()

    def test_credentials_reach_the_memory_factory(self, tmp_path):
        """Wave 1's bundle-sourced embedding keys flow through: the
        bundle's 'embedding' entry fills the api_key the config omits."""
        bundle = CredentialBundle(
            by_provider={
                "anthropic": ProviderCredentials(api_key="sk-a"),
                "embedding": ProviderCredentials(
                    api_key="sk-embed", extras={"provider": "openai"}
                ),
            }
        )
        m = _manifest(
            memory={
                "provider": "file",
                "config": {
                    "root": str(tmp_path),
                    "embedding": {"provider": "openai"},
                },
            }
        )
        p = Pipeline.from_manifest(m, credentials=bundle)
        client = p.memory_provider._embedding_client
        assert client is not None
        assert client._api_key == "sk-embed"
