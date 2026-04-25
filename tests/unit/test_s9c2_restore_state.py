"""Unit tests for restore_state_from_checkpoint helpers (S9c.2)."""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s20_persist import (
    CheckpointNotFound,
    CheckpointRecord,
    FilePersister,
    NoPersister,
    PersistStage,
    restore_state_from_checkpoint,
    state_from_payload,
    state_from_record,
)


def _seed_state() -> PipelineState:
    state = PipelineState(session_id="sess-1")
    state.iteration = 7
    state.model = "claude-opus-4-7"
    state.messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    state.shared = {"k": "v", "tasks_by_status": {}}
    state.metadata = {"trace_id": "abc"}
    state.loop_decision = "complete"
    state.completion_signal = "MAX_ITERATIONS"
    state.completion_detail = "limit hit"
    state.total_cost_usd = 0.123
    state.token_usage = TokenUsage(input_tokens=42, output_tokens=21)
    state.final_text = "done"
    return state


# ── state_from_payload ────────────────────────────────────────────


class TestStateFromPayload:
    def test_round_trip_full(self):
        original = _seed_state()
        from geny_executor.stages.s20_persist.artifact.default.stage import (
            _build_payload,
        )

        payload = _build_payload(original)
        restored = state_from_payload(payload)
        assert restored.session_id == "sess-1"
        assert restored.iteration == 7
        assert restored.model == "claude-opus-4-7"
        assert restored.messages == original.messages
        assert restored.shared == original.shared
        assert restored.metadata == original.metadata
        assert restored.loop_decision == "complete"
        assert restored.completion_signal == "MAX_ITERATIONS"
        assert restored.completion_detail == "limit hit"
        assert restored.total_cost_usd == pytest.approx(0.123)
        assert restored.token_usage.input_tokens == 42
        assert restored.token_usage.output_tokens == 21
        assert restored.final_text == "done"

    def test_empty_payload_yields_defaults(self):
        s = state_from_payload({})
        assert s.session_id == ""
        assert s.iteration == 0
        assert s.messages == []
        assert s.shared == {}
        assert s.metadata == {}
        assert s.loop_decision == "continue"
        assert s.completion_signal is None
        assert s.completion_detail is None
        assert s.total_cost_usd == 0.0
        assert s.token_usage.input_tokens == 0
        assert s.token_usage.output_tokens == 0

    def test_unknown_extra_keys_ignored(self):
        s = state_from_payload({"future_field": "ignored", "session_id": "x"})
        assert s.session_id == "x"
        # No exception, no field added.

    def test_token_usage_defaults_when_missing(self):
        s = state_from_payload({"token_usage": {}})
        assert s.token_usage.input_tokens == 0

    def test_token_usage_non_dict_falls_back(self):
        s = state_from_payload({"token_usage": "garbage"})
        assert s.token_usage.input_tokens == 0


class TestStateFromRecord:
    def test_passes_payload_through(self):
        record = CheckpointRecord(
            session_id="sess",
            iteration=3,
            payload={"session_id": "sess", "iteration": 3, "final_text": "ok"},
        )
        s = state_from_record(record)
        assert s.session_id == "sess"
        assert s.iteration == 3
        assert s.final_text == "ok"


# ── restore_state_from_checkpoint ────────────────────────────────


class TestRestoreFromCheckpoint:
    @pytest.mark.asyncio
    async def test_round_trip_via_file_persister(self, tmp_path):
        persister = FilePersister(base_dir=tmp_path)
        original_state = _seed_state()

        # Run the stage once to write a real checkpoint.
        stage = PersistStage(persister=persister)
        await stage.execute(input=None, state=original_state)
        checkpoint_id = original_state.shared["last_checkpoint"]

        # Restore.
        restored = await restore_state_from_checkpoint(persister, checkpoint_id)
        assert restored.session_id == original_state.session_id
        assert restored.iteration == original_state.iteration
        assert restored.messages == original_state.messages
        assert restored.completion_signal == original_state.completion_signal
        assert restored.token_usage.input_tokens == 42

    @pytest.mark.asyncio
    async def test_missing_checkpoint_raises(self, tmp_path):
        persister = FilePersister(base_dir=tmp_path)
        with pytest.raises(CheckpointNotFound):
            await restore_state_from_checkpoint(persister, "ghost")

    @pytest.mark.asyncio
    async def test_persister_returning_none_raises_checkpoint_not_found(self):
        # NoPersister.read returns None by default — so restore must raise.
        with pytest.raises(CheckpointNotFound):
            await restore_state_from_checkpoint(NoPersister(), "anything")

    @pytest.mark.asyncio
    async def test_runtime_fields_not_restored(self, tmp_path):
        """The payload doesn't carry llm_client / session_runtime by
        design; restored state has the raw defaults (None)."""
        persister = FilePersister(base_dir=tmp_path)
        state = _seed_state()
        state.session_runtime = object()  # set on live state but excluded from payload
        stage = PersistStage(persister=persister)
        await stage.execute(input=None, state=state)
        ckpt = state.shared["last_checkpoint"]

        restored = await restore_state_from_checkpoint(persister, ckpt)
        assert restored.session_runtime is None
        assert restored.llm_client is None

    @pytest.mark.asyncio
    async def test_persister_propagates_non_lookup_errors(self, tmp_path):
        class BoomPersister(NoPersister):
            async def read(self, checkpoint_id):
                raise RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await restore_state_from_checkpoint(BoomPersister(), "x")
