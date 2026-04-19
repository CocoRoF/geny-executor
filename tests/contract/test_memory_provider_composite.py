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
