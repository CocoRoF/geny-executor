"""C2 — Execution-end recording.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[1]`):
On terminal state, `record_execution(...)` writes a dated LTM entry,
creates a structured note, and incrementally indexes new content
into the vector layer (when present). Emits `memory.execution_recorded`
with non-zero counts.

State: **red** until Phase 2 ships `FileMemoryProvider`.
"""

from __future__ import annotations

import pytest

PHASE_REASON = "C2 awaits Phase 2: native FileMemoryProvider with LTM/notes/vector backends."


def _provider_module_available() -> bool:
    try:
        from geny_executor.memory.provider import MemoryProvider  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c2_execution_writes_dated_ltm_note_and_vector(spec, registered_providers):
    """When activated:

    1. Build a session, run one full request/response.
    2. Call `provider.record_execution(ExecutionSummary.from_state(state))`.
    3. Assert returned `RecordReceipt` has notes_written >= 1,
       vector_chunks > 0 (if vector handle present), files_updated
       includes a dated path matching `YYYY-MM-DD.md`.
    4. Assert a `memory.execution_recorded` event was emitted with
       matching counts.
    """
    if not registered_providers:
        pytest.skip(PHASE_REASON)
    raise AssertionError("C2 acceptance not yet implemented")  # noqa: TRY003
