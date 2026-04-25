"""Unit tests for Stage 20 Persist (S9b.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s20_persist import (
    CHECKPOINT_HISTORY_KEY,
    CheckpointRecord,
    EveryNTurnsFrequency,
    EveryTurnFrequency,
    FilePersister,
    LAST_CHECKPOINT_KEY,
    NoPersister,
    OnSignificantFrequency,
    PersistStage,
)


def _state(*, session_id: str = "s", iteration: int = 0) -> PipelineState:
    state = PipelineState(session_id=session_id)
    state.iteration = iteration
    state.token_usage = TokenUsage(input_tokens=10, output_tokens=5)
    return state


# ── CheckpointRecord ──────────────────────────────────────────────


class TestCheckpointRecord:
    def test_default_id_prefixed(self):
        r = CheckpointRecord()
        assert r.checkpoint_id.startswith("ckpt_")

    def test_to_dict(self):
        r = CheckpointRecord(
            session_id="s",
            iteration=3,
            payload={"x": 1},
        )
        d = r.to_dict()
        assert d["session_id"] == "s"
        assert d["iteration"] == 3
        assert d["payload"] == {"x": 1}
        assert d["created_at"]


# ── NoPersister ──────────────────────────────────────────────────


class TestNoPersister:
    @pytest.mark.asyncio
    async def test_write_no_op(self):
        p = NoPersister()
        await p.write(CheckpointRecord(), _state())  # must not raise


# ── FilePersister ────────────────────────────────────────────────


class TestFilePersister:
    @pytest.mark.asyncio
    async def test_write_creates_session_dir_and_file(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        record = CheckpointRecord(session_id="sess", iteration=1, payload={"k": "v"})
        await p.write(record, _state())

        path = tmp_path / "sess" / f"{record.checkpoint_id}.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["session_id"] == "sess"
        assert data["payload"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_unknown_session_bucket(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        record = CheckpointRecord(session_id="", iteration=0)
        await p.write(record, _state(session_id=""))
        assert (tmp_path / "_unknown").exists()

    @pytest.mark.asyncio
    async def test_read_round_trip(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        record = CheckpointRecord(session_id="sess", iteration=2, payload={"x": 1})
        await p.write(record, _state())
        loaded = await p.read(record.checkpoint_id)
        assert loaded is not None
        assert loaded.session_id == "sess"
        assert loaded.iteration == 2
        assert loaded.payload == {"x": 1}

    @pytest.mark.asyncio
    async def test_read_missing_returns_none(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        assert await p.read("ghost") is None

    @pytest.mark.asyncio
    async def test_list_all_sessions(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        await p.write(
            CheckpointRecord(session_id="a", iteration=0), _state()
        )
        await p.write(
            CheckpointRecord(session_id="a", iteration=1), _state()
        )
        await p.write(
            CheckpointRecord(session_id="b", iteration=0), _state()
        )
        records = await p.list_checkpoints()
        assert len(records) == 3

    @pytest.mark.asyncio
    async def test_list_by_session(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        await p.write(CheckpointRecord(session_id="a", iteration=0), _state())
        await p.write(CheckpointRecord(session_id="b", iteration=0), _state())
        records = await p.list_checkpoints(session_id="a")
        assert len(records) == 1
        assert records[0].session_id == "a"

    @pytest.mark.asyncio
    async def test_atomic_write_no_temp_files_left(self, tmp_path):
        p = FilePersister(base_dir=tmp_path)
        for i in range(3):
            await p.write(
                CheckpointRecord(session_id="sess", iteration=i), _state()
            )
        leftover = list((tmp_path / "sess").glob("*.json.tmp"))
        assert leftover == []


# ── Frequency policies ───────────────────────────────────────────


class TestFrequencies:
    def test_every_turn_always_true(self):
        f = EveryTurnFrequency()
        assert f.should_persist(_state()) is True

    def test_every_n_turns(self):
        f = EveryNTurnsFrequency(n=3)
        assert f.should_persist(_state(iteration=0)) is True
        assert f.should_persist(_state(iteration=1)) is False
        assert f.should_persist(_state(iteration=3)) is True

    def test_every_n_turns_validation(self):
        with pytest.raises(ValueError):
            EveryNTurnsFrequency(n=0)

    def test_every_n_turns_configure(self):
        f = EveryNTurnsFrequency(n=3)
        f.configure({"n": 5})
        assert f.n == 5
        with pytest.raises(ValueError):
            f.configure({"n": -1})

    def test_on_significant_no_signals(self):
        f = OnSignificantFrequency()
        assert f.should_persist(_state()) is False

    def test_on_significant_completion_signal(self):
        f = OnSignificantFrequency()
        s = _state()
        s.completion_signal = "done"
        assert f.should_persist(s) is True

    def test_on_significant_event_match(self):
        f = OnSignificantFrequency()
        s = _state(iteration=2)
        s.add_event("hitl.decision", {})
        # add_event records iteration from current_stage; force the
        # match by patching the most recent event's iteration.
        s.events[-1]["iteration"] = 2
        assert f.should_persist(s) is True

    def test_on_significant_review_error(self):
        f = OnSignificantFrequency()
        s = _state()
        s.shared["tool_review_flags"] = [{"severity": "error"}]
        assert f.should_persist(s) is True

    def test_on_significant_high_importance_summary(self):
        f = OnSignificantFrequency()
        s = _state()
        # Use a dict-shaped record.
        s.shared["turn_summary"] = {"importance": "high"}
        assert f.should_persist(s) is True

    def test_on_significant_low_importance_summary_skipped(self):
        f = OnSignificantFrequency()
        s = _state()
        s.shared["turn_summary"] = {"importance": "low"}
        assert f.should_persist(s) is False

    def test_on_significant_event_match_wrong_iteration(self):
        f = OnSignificantFrequency()
        s = _state(iteration=2)
        s.add_event("hitl.decision", {})
        # Event recorded with iteration=0 (state.iteration was 2 only
        # when add_event ran but state.current_stage is unset, etc.).
        # Force mismatch.
        s.events[-1]["iteration"] = 1
        assert f.should_persist(s) is False


# ── PersistStage ─────────────────────────────────────────────────


class TestPersistStage:
    def test_default_bypasses(self):
        stage = PersistStage()
        assert stage.should_bypass(_state()) is True

    @pytest.mark.asyncio
    async def test_default_direct_execute_writes_noop_record(self):
        """Calling execute() directly bypasses should_bypass; the
        NoPersister write itself is a no-op so this is a cheap
        bookkeeping path, not real IO."""
        stage = PersistStage()
        s = _state()
        await stage.execute(input=None, state=s)
        # Bookkeeping happens but the persister wrote nothing real.
        assert s.shared[LAST_CHECKPOINT_KEY].startswith("ckpt_")

    @pytest.mark.asyncio
    async def test_file_persister_writes_and_records(self, tmp_path):
        stage = PersistStage(persister=FilePersister(base_dir=tmp_path))
        s = _state(session_id="sess", iteration=1)
        s.messages = [{"role": "user", "content": "hi"}]
        await stage.execute(input=None, state=s)
        # Last id recorded.
        last_id = s.shared[LAST_CHECKPOINT_KEY]
        assert last_id and last_id.startswith("ckpt_")
        # File on disk.
        assert (tmp_path / "sess" / f"{last_id}.json").exists()
        # History updated.
        history = s.shared[CHECKPOINT_HISTORY_KEY]
        assert len(history) == 1
        assert history[0]["checkpoint_id"] == last_id
        # Event emitted.
        evts = [e for e in s.events if e["type"] == "checkpoint.written"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_frequency_skip_emits_event(self, tmp_path):
        stage = PersistStage(
            persister=FilePersister(base_dir=tmp_path),
            frequency=EveryNTurnsFrequency(n=10),
        )
        s = _state(iteration=1)  # not a multiple of 10
        await stage.execute(input=None, state=s)
        assert LAST_CHECKPOINT_KEY not in s.shared
        skips = [e for e in s.events if e["type"] == "checkpoint.skipped"]
        assert len(skips) == 1

    @pytest.mark.asyncio
    async def test_persister_exception_isolated(self, tmp_path):
        class BoomPersister(FilePersister):
            async def write(self, record, state):
                raise RuntimeError("kaboom")

        stage = PersistStage(persister=BoomPersister(base_dir=tmp_path))
        s = _state(session_id="sess")
        await stage.execute(input=None, state=s)
        assert LAST_CHECKPOINT_KEY not in s.shared
        errs = [e for e in s.events if e["type"] == "checkpoint.persister_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_payload_excludes_runtime_refs(self, tmp_path):
        stage = PersistStage(persister=FilePersister(base_dir=tmp_path))
        s = _state(session_id="sess")

        # session_runtime / llm_client should not appear in the
        # persisted payload (they're runtime-only).
        class _Runtime:
            secret = "should-not-be-saved"

        s.session_runtime = _Runtime()

        await stage.execute(input=None, state=s)
        last_id = s.shared[LAST_CHECKPOINT_KEY]
        path = tmp_path / "sess" / f"{last_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        # Payload only contains the documented keys.
        payload_keys = set(data["payload"].keys())
        assert "session_runtime" not in payload_keys
        assert "llm_client" not in payload_keys
        assert payload_keys >= {
            "session_id",
            "iteration",
            "messages",
            "shared",
            "metadata",
            "loop_decision",
        }

    @pytest.mark.asyncio
    async def test_history_accumulates(self, tmp_path):
        stage = PersistStage(persister=FilePersister(base_dir=tmp_path))
        s = _state(session_id="sess")
        await stage.execute(input=None, state=s)
        s.iteration = 1
        await stage.execute(input=None, state=s)
        assert len(s.shared[CHECKPOINT_HISTORY_KEY]) == 2

    def test_slot_registries(self):
        stage = PersistStage()
        slots = stage.get_strategy_slots()
        assert set(slots["persister"].registry) == {"no_persist", "file"}
        assert set(slots["frequency"].registry) == {
            "every_turn",
            "every_n_turns",
            "on_significant",
        }

    def test_default_strategies(self):
        stage = PersistStage()
        slots = stage.get_strategy_slots()
        assert isinstance(slots["persister"].strategy, NoPersister)
        assert isinstance(slots["frequency"].strategy, EveryTurnFrequency)
