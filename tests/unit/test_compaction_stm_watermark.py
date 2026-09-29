"""Stage-18 STM watermark survives compaction (audit D3, 2.51.0).

Compaction shrinks ``state.messages``; the recorded-watermark index must
be remapped so recording doesn't silently stop until the list regrows.
"""

from __future__ import annotations

from geny_executor.core.compaction import (
    _STATE_LAST_RECORDED,
    UNRECORDED_KEY,
    reconcile_recorded_index,
)


def _msgs(n):
    # Distinct dict objects so identity matching is meaningful.
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_watermark_remaps_when_boundary_inside_kept_suffix():
    before = _msgs(61)  # indices 0..60
    # LLMSummary-style: 2 synthetic + last 4 real messages (same objects).
    kept = before[-4:]
    after = [{"role": "user", "content": "S1"}, {"role": "assistant", "content": "S2"}] + kept
    meta = {_STATE_LAST_RECORDED: 60}  # msgs 0..59 recorded, msg60 pending

    reconcile_recorded_index(before, after, meta)

    # kept suffix starts at before-index 57; synthetic=2 → new = 2 + (60-57) = 5.
    assert meta[_STATE_LAST_RECORDED] == 5
    # after[5:] == [msg60] → the pending message is still recordable, and
    # the already-recorded msg57/58/59 (after indices 2,3,4) are not re-run.
    assert after[5:] == [before[60]]


def test_watermark_clamps_when_boundary_before_kept():
    before = _msgs(20)
    kept = before[-3:]  # start=17
    after = [{"role": "assistant", "content": "S"}] + kept  # synthetic=1
    meta = {_STATE_LAST_RECORDED: 10}  # boundary well before the kept region

    reconcile_recorded_index(before, after, meta)

    # old_idx(10) <= start(17) → new = n_synthetic = 1; kept suffix all recordable.
    assert meta[_STATE_LAST_RECORDED] == 1
    assert after[1:] == kept


def test_watermark_never_exceeds_new_length():
    before = _msgs(30)
    after = _msgs(3)  # truncate, no shared identity (worst case)
    meta = {_STATE_LAST_RECORDED: 30}
    reconcile_recorded_index(before, after, meta)
    assert 0 <= meta[_STATE_LAST_RECORDED] <= 3


def test_nothing_recorded_yet_holds_the_removed_messages_for_the_recorder():
    """2.76.0: a turn compacted before anything was recorded used to keep
    the watermark at 0 — the recorder then wrote the summary head into
    memory as if it had been said, and the removed messages nowhere."""
    before = _msgs(10)
    kept = before[-3:]
    after = [{"role": "user", "content": "S"}, {"role": "assistant", "content": "ack"}] + kept
    meta = {}
    reconcile_recorded_index(before, after, meta)
    assert meta[_STATE_LAST_RECORDED] == 2  # the summary head is not recorded
    assert meta[UNRECORDED_KEY] == before[:7]

    # Nothing removed: nothing to do.
    same = _msgs(4)
    meta2 = {_STATE_LAST_RECORDED: 0}
    reconcile_recorded_index(same, list(same), meta2)
    assert meta2 == {_STATE_LAST_RECORDED: 0}


def test_unrecorded_messages_come_first_and_are_taken_once():
    from geny_executor.core.compaction import mark_recorded, unrecorded_messages
    from geny_executor.core.state import PipelineState

    before = _msgs(12)
    state = PipelineState()
    state.metadata[_STATE_LAST_RECORDED] = 5  # m0..m4 recorded
    kept = before[-4:]
    state.messages = [{"role": "user", "content": "S"}, {"role": "assistant", "content": "ack"}] + kept
    reconcile_recorded_index(before, state.messages, state.metadata)
    # m5..m7 were removed unrecorded; m8..m11 kept, unrecorded.
    assert [m["content"] for m in unrecorded_messages(state)] == [f"m{i}" for i in range(5, 12)]
    mark_recorded(state)
    assert unrecorded_messages(state) == []
    assert UNRECORDED_KEY not in state.metadata


def test_truncate_compactor_keeps_recording_after_shrink():
    """End-to-end: after a real TruncateCompactor shrink, the watermark
    points such that the newest message is still recordable."""
    import asyncio

    from geny_executor.core.compaction import run_compaction
    from geny_executor.core.state import PipelineState
    from geny_executor.stages.s02_context.artifact.default.compactors import (
        TruncateCompactor,
    )

    state = PipelineState()
    state.messages = _msgs(40)
    state.metadata[_STATE_LAST_RECORDED] = 40  # everything recorded so far
    newest = state.messages[-1]

    asyncio.run(run_compaction(state, TruncateCompactor(keep_last=5), trigger="proactive"))

    idx = state.metadata[_STATE_LAST_RECORDED]
    assert 0 <= idx <= len(state.messages)
    # The newest message survived the truncation and must NOT be behind
    # the watermark forever (the pre-2.51 bug left idx=40 > len=5).
    assert newest in state.messages
    assert idx <= len(state.messages)
