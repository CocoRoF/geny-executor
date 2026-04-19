"""Run the MemoryProvider contract suite against EphemeralMemoryProvider.

Phase 2 will add sibling files (`test_memory_provider_file.py`,
`test_memory_provider_sql.py`, ...) that subclass the same contract
with a different `provider` fixture — the assertion set stays
identical.
"""

from __future__ import annotations

import pytest

from geny_executor.memory.provider import MemoryProvider
from geny_executor.memory.providers import EphemeralMemoryProvider
from tests.contract.memory_provider_contract import MemoryProviderContract


class TestEphemeralProviderContract(MemoryProviderContract):
    @pytest.fixture
    async def provider(self) -> MemoryProvider:
        p = EphemeralMemoryProvider()
        await p.initialize()
        return p
