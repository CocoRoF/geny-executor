"""Test-only fixtures for the completeness suite.

Everything under `tests/completeness/fixtures/` is strictly validation
infrastructure — it must never be imported from `src/`. Quarantining
here keeps the runtime artifact free of Geny-shaped adapter code while
still giving C7 (Phase 3) a concrete provider to parity-check against.
"""
