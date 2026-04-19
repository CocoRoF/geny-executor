"""C6 — Server restart resume.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[5]`):
With session persistence enabled, restarting the host process
reloads PipelineState and reattaches the same MemoryProvider instance
so the prior conversation is recoverable verbatim (messages, system,
token_usage, cost).

State: **red** until Phase 2 wires `FileSessionPersistence` into
`SessionService` + provider factory rehydration.
"""

from __future__ import annotations

import pytest

PHASE_REASON = (
    "C6 awaits Phase 2: SessionService persistence integration + "
    "provider factory rehydration on boot."
)


def _provider_module_available() -> bool:
    try:
        from geny_executor.memory.provider import MemoryProvider  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _provider_module_available(), reason=PHASE_REASON)
def test_c6_session_state_survives_process_restart(tmp_path, spec, registered_providers):
    """When activated:

    1. SessionManager(storage_root=tmp_path, persistence_mode='file').
    2. Create session, run one turn (records messages, cost, tokens).
    3. Drop the SessionManager (simulate restart).
    4. New SessionManager(storage_root=tmp_path) reloads the session.
    5. Assert state.messages, state.system, state.token_usage,
       state.total_cost_usd all match pre-restart values byte-for-byte
       (modulo timestamp fields).
    6. Run one additional turn — assert it builds on the restored
       conversation (e.g. references a fact from turn 1).
    """
    if not registered_providers:
        pytest.skip(PHASE_REASON)
    raise AssertionError("C6 acceptance not yet implemented")  # noqa: TRY003
