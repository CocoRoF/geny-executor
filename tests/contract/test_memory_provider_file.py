"""Run the MemoryProvider contract suite against FileMemoryProvider.

One subclass, one fixture — reuses every assertion in
`MemoryProviderContract` verbatim. Divergence between the file
provider and the ephemeral reference surfaces as a test failure,
not a style choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import MemoryProvider
from geny_executor.memory.providers import FileMemoryProvider
from tests.contract.memory_provider_contract import MemoryProviderContract


class TestFileProviderContract(MemoryProviderContract):
    @pytest.fixture
    async def provider(self, tmp_path: Path) -> MemoryProvider:
        p = FileMemoryProvider(tmp_path / "session")
        await p.initialize()
        return p

    async def _fresh_from(self, provider: MemoryProvider) -> MemoryProvider:
        """Snapshot round-trip test needs a *different* root so the
        restore payload can re-populate a clean directory."""
        assert isinstance(provider, FileMemoryProvider)
        fresh_root = provider.root.parent / (provider.root.name + "-restored")
        fresh = FileMemoryProvider(fresh_root)
        await fresh.initialize()
        return fresh
