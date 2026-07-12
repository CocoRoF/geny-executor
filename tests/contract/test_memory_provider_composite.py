"""Behavioural-contract suite for `CompositeMemoryProvider`.

Same `MemoryProviderContract` mixin every other backend uses. The
fixture wires a composite where every required layer is routed to a
single underlying `FileMemoryProvider` — that is the simplest setup
that still exercises the routing layer end-to-end. Routing-specific
behaviour (different backends per layer, scope-bound promote, partial
restore, layer-skip on retrieve) lives in
`test_memory_provider_composite_routing.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.composite import CompositeMemoryProvider, LayerRouting
from geny_executor.memory.provider import Layer, MemoryProvider
from geny_executor.memory.providers import FileMemoryProvider
from tests.contract.memory_provider_contract import MemoryProviderContract


def _build_composite(root: Path) -> CompositeMemoryProvider:
    delegate = FileMemoryProvider(root=root)
    routing = LayerRouting(
        layers={
            Layer.STM: delegate,
            Layer.LTM: delegate,
            Layer.NOTES: delegate,
            Layer.INDEX: delegate,
        }
    )
    return CompositeMemoryProvider(routing=routing)


class TestCompositeProviderContract(MemoryProviderContract):
    @pytest.fixture
    async def provider(self, tmp_path: Path) -> MemoryProvider:
        p = _build_composite(tmp_path / "single")
        await p.initialize()
        return p

    async def _fresh_from(self, provider: MemoryProvider) -> MemoryProvider:
        existing = provider.routing.distinct_providers()[0]  # type: ignore[attr-defined]
        original = Path(existing.root)  # type: ignore[attr-defined]
        restored_root = original.with_name(original.stem + "-restored")
        fresh = _build_composite(restored_root)
        await fresh.initialize()
        return fresh


class TestCompositeRecordCompaction:
    """audit D5: composite must persist compaction snapshots (was file-only)."""

    @pytest.mark.asyncio
    async def test_record_compaction_writes_note(self, tmp_path: Path):
        p = _build_composite(tmp_path / "rc")
        await p.initialize()
        fname = await p.record_compaction(
            "Summarized 30 earlier turns about the deploy.",
            replaced_count=30,
            strategy="llm_summary",
            saved_tokens=4000,
            trigger="proactive",
        )
        assert fname  # a note filename, not None
        # It lands in the "compactions" category and is retrievable.
        hits = await p.notes().search("deploy", limit=5)
        assert any("deploy" in (h.content or "") for h in hits)

    @pytest.mark.asyncio
    async def test_record_compaction_empty_is_noop(self, tmp_path: Path):
        p = _build_composite(tmp_path / "rc2")
        await p.initialize()
        assert await p.record_compaction("", replaced_count=0) is None
