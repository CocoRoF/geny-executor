"""C3 — LLM reflection and auto-promotion.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[2]`):
`reflect(...)` yields Insights of (title, content, category, tags,
importance). High/critical Insights auto-promote to curated scope
via `promote(...)`. Each promotion emits `memory.promoted`.

State: **red** until Phase 2 ships reflection hook + promote().
"""

from __future__ import annotations

import pytest

PHASE_REASON = "C3 awaits Phase 2: reflect() + promote() + auto-promotion hook."


def _provider_module_available() -> bool:
    try:
        from geny_executor.memory.provider import MemoryProvider  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c3_reflection_extracts_insights_and_promotes(spec, registered_providers):
    """When activated:

    1. Seed STM with a multi-turn conversation containing a clearly
       quotable fact (suitable for high-importance Insight).
    2. Drive `provider.reflect(ReflectionContext.from_state(state))`
       with a deterministic stub LLM that returns a fixed Insight
       payload (importance=high).
    3. Assert at least one Insight in returned sequence, schema-valid.
    4. Assert `provider.curated().list()` now contains the promoted
       note, and that a `memory.promoted` event was emitted with the
       new NoteRef and to_scope=Scope.USER (or USER_CURATED).
    """
    if not registered_providers:
        pytest.skip(PHASE_REASON)
    raise AssertionError("C3 acceptance not yet implemented")  # noqa: TRY003
