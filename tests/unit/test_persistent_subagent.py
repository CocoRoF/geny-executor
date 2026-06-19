"""Persistent sub-agent (the owned, autonomous, notify-on-completion delegate).

Pins the executor-level mechanism: spawn → assign (autonomous) → completion
lands in the owner's inbox (the alarm); multi-turn state accumulates; failures
report; session_store persists; stop closes the pipeline.
"""

from __future__ import annotations

import asyncio

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.subagent_type import (
    SubagentTypeDescriptor,
    SubagentTypeRegistry,
)
from geny_executor.stages.s12_agent.persistent_subagent import (
    SubAgentInbox,
    SubAgentManager,
)


class _Result:
    def __init__(self, text="ok", success=True, error=None):
        self.text = text
        self.success = success
        self.error = error


class _FakePipeline:
    """Records run() calls; accumulates into state.messages for multi-turn."""

    def __init__(self, *, fail=False):
        self.runs = []
        self.closed = False
        self._fail = fail

    async def run(self, task, state):
        self.runs.append(task)
        state.messages.append({"role": "user", "content": str(task)})
        state.messages.append({"role": "assistant", "content": f"did:{task}"})
        if self._fail:
            raise RuntimeError("boom")
        return _Result(text=f"done:{task}")

    async def aclose(self):
        self.closed = True


def _registry(pipeline):
    reg = SubagentTypeRegistry()
    reg.register(
        SubagentTypeDescriptor(agent_type="worker", factory=lambda ctx: pipeline)
    )
    return reg


@pytest.mark.asyncio
async def test_spawn_creates_idle_instance():
    pipe = _FakePipeline()
    mgr = SubAgentManager(_registry(pipe))
    agent = await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    assert agent.sub_agent_id == "sa1"
    assert agent.status == "idle"
    assert mgr.get("sa1") is agent
    assert mgr.list("owner1")[0]["sub_agent_id"] == "sa1"


@pytest.mark.asyncio
async def test_spawn_unknown_type_raises():
    mgr = SubAgentManager(SubagentTypeRegistry())
    with pytest.raises(KeyError):
        await mgr.spawn("nope", "owner1")


@pytest.mark.asyncio
async def test_assign_sync_returns_record_and_notifies_owner():
    pipe = _FakePipeline()
    mgr = SubAgentManager(_registry(pipe))
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")

    rec = await mgr.assign("sa1", "summarize X", background=False)
    assert rec["success"] is True
    assert rec["text"] == "done:summarize X"
    # completion delivered to owner's inbox (the alarm)
    msgs = mgr.read_inbox("owner1")
    assert len(msgs) == 1
    assert msgs[0]["kind"] == "completion"
    assert msgs[0]["sender"] == "sa1"
    # drained
    assert mgr.read_inbox("owner1") == []


@pytest.mark.asyncio
async def test_assign_background_runs_and_notifies():
    pipe = _FakePipeline()
    mgr = SubAgentManager(_registry(pipe))
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")

    out = await mgr.assign("sa1", "task1", background=True)
    assert out["status"] == "running"
    # let the background task finish
    for _ in range(50):
        if mgr.inbox.count("owner1"):
            break
        await asyncio.sleep(0.01)
    msgs = mgr.read_inbox("owner1")
    assert len(msgs) == 1 and msgs[0]["kind"] == "completion"


@pytest.mark.asyncio
async def test_multi_turn_state_accumulates():
    pipe = _FakePipeline()
    mgr = SubAgentManager(_registry(pipe))
    agent = await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    await mgr.assign("sa1", "t1", background=False)
    await mgr.assign("sa1", "t2", background=False)
    # same kept-alive pipeline + state across assignments
    assert pipe.runs == ["t1", "t2"]
    assert len(agent.state.messages) == 4  # 2 turns × (user+assistant)


@pytest.mark.asyncio
async def test_failure_reports_failed_in_inbox():
    pipe = _FakePipeline(fail=True)
    mgr = SubAgentManager(_registry(pipe))
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    rec = await mgr.assign("sa1", "t", background=False)
    assert rec["success"] is False
    assert "run_error" in rec["error"]
    msgs = mgr.read_inbox("owner1")
    assert msgs[0]["kind"] == "failed"


@pytest.mark.asyncio
async def test_session_store_load_and_save():
    pipe = _FakePipeline()
    saved = {}

    class _Store:
        def load(self, sid):
            return None

        def save(self, sid, state):
            saved[sid] = state

    mgr = SubAgentManager(_registry(pipe), session_store=_Store())
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    await mgr.assign("sa1", "t", background=False)
    assert "sa1" in saved  # state persisted after assignment


@pytest.mark.asyncio
async def test_restore_state_on_spawn():
    pipe = _FakePipeline()
    prior = PipelineState(session_id="sa1")
    prior.messages.append({"role": "user", "content": "earlier"})

    class _Store:
        def load(self, sid):
            return prior if sid == "sa1" else None

        def save(self, sid, state):
            pass

    mgr = SubAgentManager(_registry(pipe), session_store=_Store())
    agent = await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    assert agent.state is prior
    assert len(agent.state.messages) == 1  # restored conversation


@pytest.mark.asyncio
async def test_stop_closes_pipeline_and_removes():
    pipe = _FakePipeline()
    mgr = SubAgentManager(_registry(pipe))
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    ok = await mgr.stop("sa1")
    assert ok is True
    assert pipe.closed is True
    assert mgr.get("sa1") is None


@pytest.mark.asyncio
async def test_on_event_callback_fires():
    pipe = _FakePipeline()
    events = []
    mgr = SubAgentManager(
        _registry(pipe), on_event=lambda et, p: events.append(et)
    )
    await mgr.spawn("worker", "owner1", sub_agent_id="sa1")
    await mgr.assign("sa1", "t", background=False)
    assert "subagent.spawned" in events
    assert "subagent.assigned" in events
    assert "subagent.completed" in events


def test_inbox_bounded():
    inbox = SubAgentInbox(max_per_owner=3)
    from geny_executor.stages.s12_agent.persistent_subagent import InboxMessage

    for i in range(5):
        inbox.deliver(InboxMessage(id=str(i), owner="o", sender="s", kind="message", body=str(i)))
    msgs = inbox.peek("o")
    assert len(msgs) == 3
    assert [m.id for m in msgs] == ["2", "3", "4"]  # oldest dropped


@pytest.mark.asyncio
async def test_spawn_applies_model_and_system_prompt_overrides():
    """model / system_prompt are applied to the descriptor the factory sees."""
    seen = {}

    def _factory(ctx):
        seen["model_override"] = ctx.descriptor.model_override
        seen["system_prompt"] = ctx.descriptor.system_prompt
        return _FakePipeline()

    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(agent_type="worker", factory=_factory))
    mgr = SubAgentManager(reg)
    await mgr.spawn(
        "worker", "owner1", sub_agent_id="sa1",
        model="claude-opus-4-8", system_prompt="You are a strict reviewer.",
    )
    assert seen["model_override"] == "claude-opus-4-8"
    assert seen["system_prompt"] == "You are a strict reviewer."
