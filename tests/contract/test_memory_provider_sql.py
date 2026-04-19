"""Run the MemoryProvider contract suite against SQLMemoryProvider.

One subclass, one fixture — reuses every assertion in
`MemoryProviderContract` verbatim. The SQL provider is a separate
backend implementation: passing this suite is the cross-backend
parity guarantee.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import MemoryProvider
from geny_executor.memory.providers import SQLMemoryProvider
from tests.contract.memory_provider_contract import MemoryProviderContract


class TestSQLProviderContract(MemoryProviderContract):
    @pytest.fixture
    async def provider(self, tmp_path: Path) -> MemoryProvider:
        p = SQLMemoryProvider(tmp_path / "session.db")
        await p.initialize()
        return p

    async def _fresh_from(self, provider: MemoryProvider) -> MemoryProvider:
        """Snapshot round-trip needs a *different* DSN so the restore
        payload populates a clean database.
        """
        assert isinstance(provider, SQLMemoryProvider)
        original = Path(provider.dsn)
        fresh_path = original.with_name(original.stem + "-restored.db")
        fresh = SQLMemoryProvider(fresh_path)
        await fresh.initialize()
        return fresh
