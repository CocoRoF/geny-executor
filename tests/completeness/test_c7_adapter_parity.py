"""C7 — Adapter parity.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[6]`):
Running the same scenario through `GenyManagerAdapter` (reference
fixture) and through the native `FileMemoryProvider` yields
semantically equivalent outputs: identical chunk composition (modulo
float tolerance), identical note set, identical link graph, identical
vector top-k for each probe query.

This is the **G3 gate** test. When green, executor v1.0.0-rc is
declared complete. The adapter is then quarantined as a test fixture
and excluded from the runtime artifact.

State: **red** until Phase 3.
"""

from __future__ import annotations

import pytest

PHASE_REASON = (
    "C7 awaits Phase 3: GenyManagerAdapter as test fixture + "
    "FileMemoryProvider, with shared scenario suite running both."
)

GOLDEN_DATASET_PATH = "tests/completeness/fixtures/geny_golden/"


def _provider_module_available() -> bool:
    try:
        from geny_executor.memory.provider import MemoryProvider  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c7_native_and_adapter_produce_identical_outputs(spec):
    """When activated:

    For each scenario in `tests/completeness/fixtures/geny_golden/`:
    1. Mount the same fixture with both `GenyManagerAdapter(...)` and
       `FileMemoryProvider(root=<copy>)`.
    2. Replay the recorded turn sequence through both.
    3. Assert per-turn:
       - retrieve(...) returns the same chunk keys in the same order
         (relevance scores within 1e-3).
       - notes().list() returns the same NoteMeta set.
       - notes().graph() yields the same edge set.
       - vector().search(probe, top_k=5) returns the same NoteRefs.
    4. Aggregate any mismatch into a single readable diff.
    """
    raise AssertionError("C7 acceptance not yet implemented")  # noqa: TRY003


@pytest.mark.skip(reason="Phase 3 — performance gate, requires Geny baseline.")
def test_c7_native_perf_within_20pct_of_geny_baseline(spec):
    """Native FileMemoryProvider must complete the same retrieval
    workload within ±20% of Geny `build_memory_context_async` time.
    """
    raise AssertionError("C7 perf gate not yet implemented")  # noqa: TRY003
