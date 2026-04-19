"""C3 — LLM reflection and auto-promotion.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[2]`):
`reflect(...)` yields Insights of (title, content, category, tags,
importance). High/critical Insights auto-promote to curated scope via
`promote(...)`. Each promotion emits `memory.promoted`.

Native providers keep `reflect()` passive — insights come from a stage
wiring in an LLM callable via `MemoryHooks`. The C3 activation pins
the parts the provider *does* own: the `Insight.should_auto_promote`
gate plus the `promote()` machinery that moves a SESSION-scoped note
into USER scope. A stage wrapping these calls is what emits
`memory.promoted`; the event payload is the returned NoteRef, which
this test asserts is correctly re-scoped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    Insight,
    MemoryProvider,
    NoteDraft,
    Scope,
)


async def _seed_and_promote(provider: MemoryProvider) -> None:
    # Build an Insight that would be auto-promoted by the stage.
    insight = Insight(
        title="Stage-2 separation drift is systemic",
        content="Across 12 flights, stage-2 separation introduced 0.8° drift.",
        category="insights",
        tags=["telemetry", "anomaly"],
        importance=Importance.HIGH,
    )
    assert insight.should_auto_promote(), "HIGH importance must auto-promote"

    # Materialise the Insight as a session-scoped note.
    notes = provider.notes()
    meta = await notes.write(
        NoteDraft(
            title=insight.title,
            body=insight.content,
            importance=insight.importance,
            tags=list(insight.tags),
            category=insight.category,
            scope=Scope.SESSION,
        )
    )

    # Providers without PROMOTE capability return the ref unchanged.
    new_ref = await provider.promote(meta.ref, Scope.USER)
    assert new_ref.scope == Scope.USER, f"promote() returned scope={new_ref.scope}, expected USER"


async def test_c3_reflection_extracts_insights_and_promotes(
    tmp_path: Path,
    registered_providers,
):
    if not registered_providers:
        pytest.skip("no providers registered for C3")

    for name, factory in registered_providers:
        root = tmp_path / f"c3-{name}"
        root.mkdir()
        provider = await factory(root)
        try:
            await _seed_and_promote(provider)
        finally:
            await provider.close()
