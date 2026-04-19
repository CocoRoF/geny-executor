"""Quarantined `GenyManagerAdapter` — C7 parity fixture.

The adapter is a `MemoryProvider`-conforming wrapper that, in Phase 3,
will sit in front of a real Geny `SessionMemoryManager` and replay
scenarios alongside the native providers so the two implementations
can be compared chunk-for-chunk (see
`tests/completeness/test_c7_adapter_parity.py`).

For Phase 2e the adapter is deliberately kept under `tests/` — not
`src/` — so the runtime artifact carries zero Geny-shaped code. The
body here is a thin delegate over `EphemeralMemoryProvider` so C7
scenarios can import it today without requiring Geny as a test
dependency; swapping the delegate for a real `SessionMemoryManager`
wrapper in Phase 3 is a single-class change that doesn't touch the
C1·C2·C3·C5·C6 activations.
"""

from tests.completeness.fixtures.adapter.adapter import GenyManagerAdapter

__all__ = ["GenyManagerAdapter"]
