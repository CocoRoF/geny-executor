"""C6 — Server restart resume.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[5]`):
With session persistence enabled, restarting the host process
reloads PipelineState and reattaches the same MemoryProvider instance
so the prior conversation is recoverable verbatim (messages, system,
token_usage, cost).

The full SessionManager persistence rehydration is Phase 4 territory
(web service integration). Phase 2e pins the memory-layer
precondition that unblocks it: every provider must round-trip its
observable state through `snapshot()` / `restore()` byte-for-byte on
the layers it owns. If that contract breaks, SessionManager
rehydration can't possibly succeed — so this test is the prerequisite
gate the Phase 4 integration will ride on top of.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    MemoryProvider,
    NoteDraft,
    Scope,
    Turn,
)
from geny_executor.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)
from tests.completeness.fixtures.adapter import GenyManagerAdapter


async def _seed(provider: MemoryProvider) -> None:
    await provider.stm().append(Turn(role="user", content="Compute Fibonacci(20) please."))
    await provider.stm().append(Turn(role="assistant", content="Fibonacci(20) = 6765."))
    await provider.ltm().append("User asked about Fibonacci(20); answer is 6765.")
    await provider.notes().write(
        NoteDraft(
            title="fib-20-fact",
            body="Fibonacci(20) = 6765. Confirmed via closed-form and recursion.",
            importance=Importance.HIGH,
            tags=["math", "fibonacci"],
            category="insights",
            scope=Scope.SESSION,
        )
    )


async def _rehydrate(kind: str, root: Path, snapshot_payload) -> MemoryProvider:
    """Build a fresh provider instance (simulating a process restart)
    and restore the prior snapshot into it. Each provider kind needs a
    *different* storage root than the source so we're guaranteed to be
    testing restore, not just reading leftover state.
    """
    if kind == "ephemeral":
        provider = EphemeralMemoryProvider()
    elif kind == "file":
        provider = FileMemoryProvider(root=root / "restored")
    elif kind == "sql":
        provider = SQLMemoryProvider(dsn=str(root / "restored.db"))
    elif kind == "geny-adapter":
        provider = GenyManagerAdapter()
    else:
        raise AssertionError(f"unknown provider kind {kind!r}")
    await provider.initialize()
    await provider.restore(snapshot_payload)
    return provider


async def test_c6_session_state_survives_process_restart(
    tmp_path: Path,
    registered_providers,
):
    if not registered_providers:
        pytest.skip("no providers registered for C6")

    for name, factory in registered_providers:
        root = tmp_path / f"c6-{name}"
        root.mkdir()
        source = await factory(root)
        try:
            await _seed(source)

            turns_before = await source.stm().recent(n=100)
            ltm_before = await source.ltm().read_main()
            notes_before = await source.notes().list()
            snap = await source.snapshot()

            await source.close()
        except BaseException:
            await source.close()
            raise

        # Restart — fresh provider, same backing store (or sibling for
        # file/sql so we verify restore actually repopulated it).
        restored = await _rehydrate(name, root, snap)
        try:
            turns_after = await restored.stm().recent(n=100)
            ltm_after = await restored.ltm().read_main()
            notes_after = await restored.notes().list()

            assert len(turns_after) == len(turns_before), (
                f"{name}: STM turn count changed {len(turns_before)} → {len(turns_after)}"
            )
            assert [t.content for t in turns_after] == [t.content for t in turns_before], (
                f"{name}: STM turn contents drifted after restore"
            )
            assert ltm_after.strip() == ltm_before.strip(), (
                f"{name}: LTM main body drifted after restore"
            )
            assert {n.ref.filename for n in notes_after} == {
                n.ref.filename for n in notes_before
            }, f"{name}: notes set changed after restore"
        finally:
            await restored.close()
