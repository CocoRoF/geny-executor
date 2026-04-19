"""C2 — Execution-end recording.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[1]`):
On terminal state, `record_execution(...)` writes a dated LTM entry,
creates a structured note, and incrementally indexes new content
into the vector layer (when present). Emits `memory.execution_recorded`
with non-zero counts.

The activation exercises every registered provider. Event emission
itself is a stage-level concern (MemoryEvent.EXECUTION_RECORDED is the
payload the stage would then dispatch); this test pins the
provider-side contract — the shape of `RecordReceipt` the event
carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import ExecutionSummary, MemoryProvider


async def _run_and_record(provider: MemoryProvider, session_id: str) -> None:
    summary = ExecutionSummary(
        session_id=session_id,
        user_input="what's the Falcon 9 cadence this quarter?",
        final_text="Falcon 9 averaged 1.2 launches per week in Q1, peaking at 3/week in March.",
        iterations=1,
        duration_ms=420,
        tags=["rocket", "cadence"],
    )
    receipt = await provider.record_execution(summary)

    assert receipt.notes_written >= 1, (
        f"{session_id}: record_execution produced no notes (receipt={receipt})"
    )
    assert receipt.files_updated, f"{session_id}: record_execution reported no files_updated"


async def test_c2_execution_writes_dated_ltm_note_and_vector(
    tmp_path: Path,
    registered_providers,
):
    if not registered_providers:
        pytest.skip("no providers registered for C2")

    for name, factory in registered_providers:
        root = tmp_path / f"c2-{name}"
        root.mkdir()
        provider = await factory(root)
        try:
            await _run_and_record(provider, session_id=f"session-{name}")

            notes = await provider.notes().list()
            assert notes, (
                f"{name}: notes store empty after record_execution — expected the "
                "structured insights note to be written"
            )
        finally:
            await provider.close()
