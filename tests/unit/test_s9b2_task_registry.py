"""Unit tests for Stage 13 Task Registry (S9b.2)."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_task_registry import (
    EagerWaitPolicy,
    FireAndForgetPolicy,
    InMemoryRegistry,
    PENDING_TASKS_KEY,
    TASKS_BY_STATUS_KEY,
    TaskRecord,
    TaskRegistryStage,
    TaskStatus,
    TimedWaitPolicy,
)


# ── TaskRecord / TaskStatus ────────────────────────────────────────────


class TestTaskRecord:
    def test_default_status_pending(self):
        r = TaskRecord(task_id="t1")
        assert r.status == TaskStatus.PENDING
        assert r.is_terminal is False

    def test_mark_running_sets_started_at(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.RUNNING)
        assert r.started_at is not None

    def test_mark_done_sets_completed_at_and_result(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.DONE, result={"value": 42})
        assert r.is_terminal
        assert r.completed_at is not None
        assert r.result == {"value": 42}

    def test_mark_failed_sets_error(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.FAILED, error="boom")
        assert r.error == "boom"
        assert r.is_terminal

    def test_to_dict_round_trip_keys(self):
        r = TaskRecord(task_id="t1", kind="K", payload={"x": 1})
        r.mark(TaskStatus.DONE, result="ok")
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["kind"] == "K"
        assert d["status"] == "done"
        assert d["payload"] == {"x": 1}
        assert d["completed_at"] is not None


# ── InMemoryRegistry ───────────────────────────────────────────────────


class TestInMemoryRegistry:
    def test_register_and_get(self):
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        assert r.get("t1") is rec

    def test_get_unknown_returns_none(self):
        assert InMemoryRegistry().get("ghost") is None

    def test_re_register_replaces(self):
        r = InMemoryRegistry()
        a = TaskRecord(task_id="t1", kind="A")
        b = TaskRecord(task_id="t1", kind="B")
        r.register(a)
        r.register(b)
        assert r.get("t1") is b

    def test_update_status(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1"))
        updated = r.update_status("t1", TaskStatus.DONE, result="ok")
        assert updated.status == TaskStatus.DONE
        assert updated.result == "ok"

    def test_update_unknown_returns_none(self):
        assert InMemoryRegistry().update_status("ghost", TaskStatus.DONE) is None

    def test_remove(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1"))
        assert r.remove("t1") is True
        assert r.remove("t1") is False

    def test_by_status_groups(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1", status=TaskStatus.PENDING))
        r.register(TaskRecord(task_id="t2", status=TaskStatus.DONE))
        r.register(TaskRecord(task_id="t3", status=TaskStatus.PENDING))
        out = r.by_status()
        assert sorted(out.keys()) == ["done", "pending"]
        assert {t.task_id for t in out["pending"]} == {"t1", "t3"}


# ── Policies ───────────────────────────────────────────────────────────


def _state() -> PipelineState:
    return PipelineState(session_id="s")


class TestFireAndForgetPolicy:
    @pytest.mark.asyncio
    async def test_no_op(self):
        p = FireAndForgetPolicy()
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        await p.apply([rec], r, _state())
        # status untouched
        assert r.get("t1").status == TaskStatus.PENDING


class TestEagerWaitPolicy:
    @pytest.mark.asyncio
    async def test_runs_executor_to_completion(self):
        async def exe(record):
            return f"done-{record.task_id}"

        p = EagerWaitPolicy(executor=exe)
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        s = _state()
        await p.apply([rec], r, s)
        assert r.get("t1").status == TaskStatus.DONE
        assert r.get("t1").result == "done-t1"
        evts = [e for e in s.events if e["type"] == "task.done"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_executor_failure_marks_failed(self):
        async def exe(record):
            raise RuntimeError("boom")

        p = EagerWaitPolicy(executor=exe)
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        s = _state()
        await p.apply([rec], r, s)
        assert r.get("t1").status == TaskStatus.FAILED
        assert "boom" in r.get("t1").error

    @pytest.mark.asyncio
    async def test_no_executor_leaves_pending(self):
        p = EagerWaitPolicy()  # no executor
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        await p.apply([rec], r, _state())
        assert r.get("t1").status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_configure_sets_executor(self):
        async def exe(record):
            return "ok"

        p = EagerWaitPolicy()
        p.configure({"executor": exe})
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        await p.apply([rec], r, _state())
        assert r.get("t1").status == TaskStatus.DONE


class TestTimedWaitPolicy:
    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        async def fast(record):
            return "ok"

        p = TimedWaitPolicy(executor=fast, timeout_seconds=1.0)
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        await p.apply([rec], r, _state())
        assert r.get("t1").status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_timeout_emits_event_and_leaves_running(self):
        async def slow(record):
            await asyncio.sleep(1.0)
            return "ok"

        p = TimedWaitPolicy(executor=slow, timeout_seconds=0.05)
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        s = _state()
        await p.apply([rec], r, s)
        # Status was set to RUNNING when work started; timeout doesn't
        # downgrade it (host scheduler may still complete the task).
        assert r.get("t1").status == TaskStatus.RUNNING
        evts = [e for e in s.events if e["type"] == "task.timeout"]
        assert len(evts) == 1

    def test_invalid_timeout_rejected(self):
        with pytest.raises(ValueError):
            TimedWaitPolicy(timeout_seconds=0)

    def test_configure_validates_timeout(self):
        p = TimedWaitPolicy(timeout_seconds=1.0)
        with pytest.raises(ValueError):
            p.configure({"timeout_seconds": -1})


# ── TaskRegistryStage ─────────────────────────────────────────────────


class TestTaskRegistryStage:
    @pytest.mark.asyncio
    async def test_drains_pending_queue(self):
        stage = TaskRegistryStage()
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1", "kind": "K"}]
        await stage.execute(input=None, state=s)
        # queue cleared; registry now has the task
        assert s.shared[PENDING_TASKS_KEY] == []
        registry = stage.get_strategy_slots()["registry"].strategy
        assert registry.get("t1") is not None

    @pytest.mark.asyncio
    async def test_emits_registered_event(self):
        stage = TaskRegistryStage()
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1"}]
        await stage.execute(input=None, state=s)
        evts = [e for e in s.events if e["type"] == "task.registered"]
        assert len(evts) == 1
        assert evts[0]["data"]["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_publishes_status_view(self):
        stage = TaskRegistryStage()
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [
            {"task_id": "t1", "status": "pending"},
            {"task_id": "t2", "status": "done"},
        ]
        await stage.execute(input=None, state=s)
        view = s.shared[TASKS_BY_STATUS_KEY]
        assert "pending" in view and len(view["pending"]) == 1
        assert "done" in view and len(view["done"]) == 1

    @pytest.mark.asyncio
    async def test_invalid_payload_skipped_with_event(self):
        stage = TaskRegistryStage()
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [
            {"task_id": ""},  # blank id rejected
            "not a dict",
            {"task_id": "ok"},
        ]
        await stage.execute(input=None, state=s)
        invalid = [e for e in s.events if e["type"] == "task_registry.invalid_payload"]
        assert len(invalid) == 2  # blank id + non-dict
        registry = stage.get_strategy_slots()["registry"].strategy
        assert registry.get("ok") is not None

    @pytest.mark.asyncio
    async def test_synced_event_summary(self):
        stage = TaskRegistryStage()
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1"}, {"task_id": "t2"}]
        await stage.execute(input=None, state=s)
        synced = [e for e in s.events if e["type"] == "task_registry.synced"]
        assert len(synced) == 1
        assert synced[0]["data"]["new"] == 2
        assert synced[0]["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_eager_wait_executes_via_policy(self):
        async def exe(record):
            return f"x{record.task_id}"

        stage = TaskRegistryStage(policy=EagerWaitPolicy(executor=exe))
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1"}]
        await stage.execute(input=None, state=s)
        registry = stage.get_strategy_slots()["registry"].strategy
        assert registry.get("t1").status == TaskStatus.DONE
        assert registry.get("t1").result == "xt1"

    @pytest.mark.asyncio
    async def test_policy_exception_does_not_block_loop(self):
        class BoomPolicy(FireAndForgetPolicy):
            async def apply(self, *args, **kwargs):
                raise RuntimeError("kaboom")

        stage = TaskRegistryStage(policy=BoomPolicy())
        s = _state()
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1"}]
        # Must not raise.
        await stage.execute(input=None, state=s)
        errs = [e for e in s.events if e["type"] == "task_registry.policy_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_empty_queue_still_publishes_view(self):
        stage = TaskRegistryStage()
        s = _state()
        await stage.execute(input=None, state=s)
        # Empty view, but key exists.
        assert s.shared.get(TASKS_BY_STATUS_KEY) == {}

    @pytest.mark.asyncio
    async def test_iteration_seen_recorded(self):
        stage = TaskRegistryStage()
        s = _state()
        s.iteration = 7
        s.shared[PENDING_TASKS_KEY] = [{"task_id": "t1"}]
        await stage.execute(input=None, state=s)
        registry = stage.get_strategy_slots()["registry"].strategy
        assert registry.get("t1").iteration_seen == 7

    def test_slot_registry_exposes_all_built_ins(self):
        stage = TaskRegistryStage()
        slots = stage.get_strategy_slots()
        assert "registry" in slots and "in_memory" in slots["registry"].registry
        assert "policy" in slots
        for policy_name in ("fire_and_forget", "eager_wait", "timed_wait"):
            assert policy_name in slots["policy"].registry

    def test_default_strategies(self):
        stage = TaskRegistryStage()
        slots = stage.get_strategy_slots()
        assert isinstance(slots["registry"].strategy, InMemoryRegistry)
        assert isinstance(slots["policy"].strategy, FireAndForgetPolicy)
