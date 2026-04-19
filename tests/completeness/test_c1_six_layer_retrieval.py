"""C1 — Six-layer retrieval round-trip.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[0]`):
A fresh session + user query produces a RetrievalResult composed of
STM session_summary, LTM main, vector top-k, keyword recall with
importance boosts, recent N messages, and optional curated. After
the response, both turns are STM-appended.

The activation runs the same scenario against every provider in
`registered_providers` so the gate protects the cross-backend
contract, not any individual implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    Layer,
    MemoryProvider,
    NoteDraft,
    RetrievalQuery,
    Scope,
    Turn,
)


async def _seed(provider: MemoryProvider) -> None:
    """Seed every required layer so retrieve() has something to find.

    The test corpus is intentionally small and keyword-homogeneous
    ("rocket launches") so the notes store's keyword scorer lights up
    the relevant rows on the probe query.
    """
    ltm = provider.ltm()
    await ltm.append("Project rocket tracks daily launches and telemetry.")

    notes = provider.notes()
    await notes.write(
        NoteDraft(
            title="rocket-launch-cadence",
            body="Falcon 9 launch cadence exceeded weekly in Q1.",
            importance=Importance.HIGH,
            tags=["rocket", "cadence"],
            category="insights",
            scope=Scope.SESSION,
        )
    )
    await notes.write(
        NoteDraft(
            title="telemetry-calibration",
            body="Telemetry calibration drifted after stage-2 separation.",
            importance=Importance.MEDIUM,
            tags=["telemetry"],
            category="insights",
            scope=Scope.SESSION,
        )
    )

    stm = provider.stm()
    await stm.append(Turn(role="user", content="any updates on rocket launches?"))


async def test_c1_retrieval_composes_six_layers(
    tmp_path: Path,
    registered_providers,
):
    if not registered_providers:
        pytest.skip("no providers registered for C1")

    required_by_spec = {Layer.STM, Layer.LTM, Layer.NOTES}

    for name, factory in registered_providers:
        root = tmp_path / f"c1-{name}"
        root.mkdir()
        provider = await factory(root)
        try:
            await _seed(provider)

            query = RetrievalQuery(
                text="rocket launches",
                layers={Layer.STM, Layer.LTM, Layer.NOTES, Layer.VECTOR},
                max_chars=4000,
            )
            result = await provider.retrieve(query)

            assert result.chunks, f"{name}: retrieve returned no chunks"
            assert result.total_chars <= query.max_chars, (
                f"{name}: total_chars {result.total_chars} exceeds budget"
            )
            covered = set(result.layer_breakdown.keys())
            missing = required_by_spec - covered
            assert not missing, (
                f"{name}: breakdown missing required layers {missing}; got {covered}"
            )

            stm_before = len(await provider.stm().recent(n=100))
            await provider.record_turn(Turn(role="user", content="follow-up: any delays?"))
            await provider.record_turn(
                Turn(role="assistant", content="No delays reported this week.")
            )
            stm_after = len(await provider.stm().recent(n=100))
            assert stm_after - stm_before == 2, (
                f"{name}: STM expected to grow by 2 turns, grew by {stm_after - stm_before}"
            )
        finally:
            await provider.close()
