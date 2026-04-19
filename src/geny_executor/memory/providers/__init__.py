"""Concrete `MemoryProvider` implementations.

Phase 1 ships only `EphemeralMemoryProvider` — a fully in-memory
reference implementation used by tests and by sessions that must
not write to disk. Phase 2 adds `FileMemoryProvider`,
`SQLMemoryProvider`, `CompositeMemoryProvider`.
"""

from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider

__all__ = ["EphemeralMemoryProvider"]
