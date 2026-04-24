"""Phase 7 Sprint S7.5 — SubagentTypeOrchestrator tests."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_agent import (
    AgentStage,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)


# ─────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────


class _FakeRunResult:
    """Minimal stand-in for PipelineResult used by the orchestrator."""

    def __init__(self, *, text: str = "ok", success: bool = True, error: str | None = None):
        self.text = text
        self.success = success
        self.error = error


class _FakePipeline:
    """Records the (input, state) that each ``run`` is called with."""

    def __init__(self, *, result: _FakeRunResult | None = None, raise_exc: Exception | None = None):
        self._result = result or _FakeRunResult()
        self._raise = raise_exc
        self.calls: List[tuple] = []

    async def run(self, input: Any, state: PipelineState):
        self.calls.append((input, state))
        if self._raise is not None:
            raise self._raise
        return self._result


def _state(*requests: dict) -> PipelineState:
    s = PipelineState(session_id="parent")
    s.delegate_requests = list(requests)
    return s


def _descriptor(
    agent_type: str = "code-reviewer",
    *,
    factory=None,
    description: str = "Reviews code",
    allowed_tools=("Read", "Grep"),
    model_override: str | None = "claude-opus-4-7",
    extras: dict | None = None,
) -> SubagentTypeDescriptor:
    return SubagentTypeDescriptor(
        agent_type=agent_type,
        factory=factory or (lambda: _FakePipeline()),
        description=description,
        allowed_tools=allowed_tools,
        model_override=model_override,
        extras=extras or {},
    )


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_and_get(self):
        reg = SubagentTypeRegistry()
        d = _descriptor("a")
        reg.register(d)
        assert reg.get("a") is d
        assert reg.get("missing") is None

    def test_duplicate_rejected(self):
        reg = SubagentTypeRegistry()
        reg.register(_descriptor("a"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_descriptor("a"))

    def test_unregister_allows_re_register(self):
        reg = SubagentTypeRegistry()
        reg.register(_descriptor("a", description="first"))
        reg.unregister("a")
        reg.register(_descriptor("a", description="second"))
        assert reg.get("a").description == "second"

    def test_list_types_sorted(self):
        reg = SubagentTypeRegistry()
        reg.register(_descriptor("c"))
        reg.register(_descriptor("a"))
        reg.register(_descriptor("b"))
        assert reg.list_types() == ["a", "b", "c"]

    def test_contains_and_len(self):
        reg = SubagentTypeRegistry()
        reg.register(_descriptor("a"))
        assert "a" in reg
        assert "b" not in reg
        assert len(reg) == 1


# ─────────────────────────────────────────────────────────────────
# Orchestrator strategy metadata
# ─────────────────────────────────────────────────────────────────


class TestOrchestratorMetadata:
    def test_name(self):
        o = SubagentTypeOrchestrator(SubagentTypeRegistry())
        assert o.name == "subagent_type"

    def test_description_includes_count(self):
        reg = SubagentTypeRegistry()
        reg.register(_descriptor("a"))
        reg.register(_descriptor("b"))
        o = SubagentTypeOrchestrator(reg)
        assert "2" in o.description

    def test_registry_property(self):
        reg = SubagentTypeRegistry()
        o = SubagentTypeOrchestrator(reg)
        assert o.registry is reg


# ─────────────────────────────────────────────────────────────────
# Empty / no-delegations path
# ─────────────────────────────────────────────────────────────────


class TestEmpty:
    @pytest.mark.asyncio
    async def test_no_requests_returns_undelegated(self):
        reg = SubagentTypeRegistry()
        o = SubagentTypeOrchestrator(reg)
        state = _state()
        result = await o.orchestrate(state)
        assert result.delegated is False
        assert result.sub_results == []


# ─────────────────────────────────────────────────────────────────
# Happy path — sync + async factory
# ─────────────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_dispatches_one_request_via_sync_factory(self):
        pipe = _FakePipeline(result=_FakeRunResult(text="reviewed!"))
        reg = SubagentTypeRegistry().register(
            _descriptor("code-reviewer", factory=lambda: pipe)
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state(
            {"agent_type": "code-reviewer", "task": "review file X"}
        )

        result = await o.orchestrate(state)

        assert result.delegated is True
        assert len(result.sub_results) == 1
        sub = result.sub_results[0]
        assert sub["agent_type"] == "code-reviewer"
        assert sub["task"] == "review file X"
        assert sub["success"] is True
        assert sub["text"] == "reviewed!"
        # Descriptor metadata surfaced
        assert sub["subagent_metadata"]["description"] == "Reviews code"
        assert sub["subagent_metadata"]["allowed_tools"] == ["Read", "Grep"]
        assert sub["subagent_metadata"]["model_override"] == "claude-opus-4-7"
        # delegate_requests was consumed
        assert state.delegate_requests == []
        # Pipeline received the task
        assert pipe.calls and pipe.calls[0][0] == "review file X"

    @pytest.mark.asyncio
    async def test_async_factory_supported(self):
        pipe = _FakePipeline()

        async def _async_factory():
            return pipe

        reg = SubagentTypeRegistry().register(
            _descriptor("a", factory=_async_factory)
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state({"agent_type": "a", "task": "t"})
        result = await o.orchestrate(state)
        assert result.sub_results[0]["success"] is True
        assert pipe.calls

    @pytest.mark.asyncio
    async def test_session_id_namespaced_per_subagent(self):
        # The sub-pipeline's state should carry a session_id derived
        # from parent + agent_type, so audit logs can stitch a tree.
        pipe = _FakePipeline()
        reg = SubagentTypeRegistry().register(
            _descriptor("planner", factory=lambda: pipe)
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state({"agent_type": "planner", "task": "plan stuff"})
        await o.orchestrate(state)
        sub_state = pipe.calls[0][1]
        assert sub_state.session_id.startswith("parent-planner-")

    @pytest.mark.asyncio
    async def test_multiple_requests_dispatched_in_order(self):
        pipe_a = _FakePipeline(result=_FakeRunResult(text="A done"))
        pipe_b = _FakePipeline(result=_FakeRunResult(text="B done"))
        reg = (
            SubagentTypeRegistry()
            .register(_descriptor("a", factory=lambda: pipe_a))
            .register(_descriptor("b", factory=lambda: pipe_b))
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state(
            {"agent_type": "a", "task": "do A"},
            {"agent_type": "b", "task": "do B"},
        )
        result = await o.orchestrate(state)
        assert [r["text"] for r in result.sub_results] == ["A done", "B done"]


# ─────────────────────────────────────────────────────────────────
# Failure paths
# ─────────────────────────────────────────────────────────────────


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_unknown_agent_type_records_error(self, caplog):
        reg = SubagentTypeRegistry()
        o = SubagentTypeOrchestrator(reg)
        state = _state({"agent_type": "ghost", "task": "go"})
        caplog.set_level("WARNING")
        result = await o.orchestrate(state)
        assert result.delegated is True
        sub = result.sub_results[0]
        assert sub["success"] is False
        assert "unknown_agent_type" in sub["error"]
        assert sub["subagent_metadata"] is None  # no descriptor → no metadata
        assert any("unknown agent_type" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_factory_error_isolated(self, caplog):
        def _broken_factory():
            raise RuntimeError("factory broke")

        reg = SubagentTypeRegistry().register(
            _descriptor("broken", factory=_broken_factory)
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state({"agent_type": "broken", "task": "t"})
        caplog.set_level("WARNING")
        result = await o.orchestrate(state)
        sub = result.sub_results[0]
        assert sub["success"] is False
        assert "factory_error" in sub["error"]
        assert "factory broke" in sub["error"]

    @pytest.mark.asyncio
    async def test_run_error_isolated(self, caplog):
        pipe = _FakePipeline(raise_exc=RuntimeError("run boom"))
        reg = SubagentTypeRegistry().register(
            _descriptor("crashy", factory=lambda: pipe)
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state({"agent_type": "crashy", "task": "t"})
        caplog.set_level("WARNING")
        result = await o.orchestrate(state)
        sub = result.sub_results[0]
        assert sub["success"] is False
        assert "run_error" in sub["error"]
        assert "run boom" in sub["error"]

    @pytest.mark.asyncio
    async def test_one_failure_does_not_block_others(self):
        pipe_ok = _FakePipeline(result=_FakeRunResult(text="ok"))
        reg = (
            SubagentTypeRegistry()
            .register(_descriptor("good", factory=lambda: pipe_ok))
        )
        o = SubagentTypeOrchestrator(reg)
        state = _state(
            {"agent_type": "ghost", "task": "go"},  # unknown
            {"agent_type": "good", "task": "yay"},
        )
        result = await o.orchestrate(state)
        assert len(result.sub_results) == 2
        assert result.sub_results[0]["success"] is False
        assert result.sub_results[1]["success"] is True


# ─────────────────────────────────────────────────────────────────
# Stage 11 strategy registry wiring
# ─────────────────────────────────────────────────────────────────


class TestStageRegistration:
    def test_subagent_type_in_strategy_registry(self):
        stage = AgentStage()
        registry = stage.get_strategy_slots()["orchestrator"].registry
        assert "subagent_type" in registry
        assert registry["subagent_type"] is SubagentTypeOrchestrator

    def test_can_inject_subagent_type_orchestrator(self):
        reg = SubagentTypeRegistry()
        orchestrator = SubagentTypeOrchestrator(reg)
        stage = AgentStage(orchestrator=orchestrator)
        assert stage._slots["orchestrator"].strategy is orchestrator
