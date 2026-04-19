"""C1 — Six-layer retrieval round-trip.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[0]`):
A fresh session + user query produces a RetrievalResult composed of
STM session_summary, LTM main, vector top-k, keyword recall with
importance boosts, recent N messages, and optional curated. After
the response, both turns are STM-appended.

State: **red** until Phase 1 ships `geny_executor.memory.provider`
and `EphemeralMemoryProvider`.
"""

from __future__ import annotations

import pytest

PHASE_REASON = (
    "C1 awaits Phase 1: `geny_executor.memory.provider.MemoryProvider` "
    "and `EphemeralMemoryProvider` must exist before this gate can run."
)


def _provider_module_available() -> bool:
    try:
        import geny_executor.memory.provider  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c1_retrieval_composes_six_layers(spec, registered_providers):
    """When activated this test will:

    1. Construct each provider in `registered_providers` with a
       known fixture corpus (LTM main, a few notes with tags +
       importance, optional vector chunks, optional curated entries).
    2. Build a `RetrievalQuery(text="...", max_chars=N)`.
    3. Assert the resulting `RetrievalResult.layer_breakdown` covers
       the layers declared `required: true` in
       `spec["retrieval"]["layer_order"]` for that provider.
    4. Run the next turn and assert STM grew by exactly two entries
       (user + assistant).
    """
    if not registered_providers:
        pytest.skip(PHASE_REASON)
    raise AssertionError("C1 acceptance not yet implemented")  # noqa: TRY003
