"""C5 — Embedding migration safety.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[4]`):
Switching embedding provider triggers compatibility_check, surfaces a
reindex_plan, and only after explicit approval performs the reindex
in the background while emitting `memory.reindexed`. Silent rebuild
is forbidden.

State: **red** until Phase 2 ships embedding clients + descriptor
compatibility check.
"""

from __future__ import annotations

import pytest

PHASE_REASON = (
    "C5 awaits Phase 2: EmbeddingClient backends + "
    "MemoryDescriptor.compatibility_check + reindex_plan."
)


def _provider_module_available() -> bool:
    try:
        from geny_executor.memory.provider import MemoryProvider  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c5_embedding_swap_requires_explicit_approval(spec, registered_providers):
    """When activated:

    1. Build a provider with embedding=openai/text-embedding-3-small.
    2. Index a small corpus.
    3. Call `provider.descriptor.compatibility_check(new_embedding)`
       with a different-dimension model — assert it returns a
       ReindexPlan, NOT a silent rebuild.
    4. Without approval call provider.vector().search(...) — must
       still return results from the original index.
    5. Approve the plan, await reindex, assert `memory.reindexed`
       event with new dimension and `cost > 0`.
    """
    if not registered_providers:
        pytest.skip(PHASE_REASON)
    raise AssertionError("C5 acceptance not yet implemented")  # noqa: TRY003
