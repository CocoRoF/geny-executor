"""Phase 3 Week 6 — Stage 10 state_mutation wiring tests.

Confirms that ``ToolResult.state_mutations`` flows from tool result
through Stage 10 into ``PipelineState.shared`` across all four
executors, respecting the is_error and namespace gates.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool import (
    ParallelExecutor,
    PartitionExecutor,
    SequentialExecutor,
    StreamingToolExecutor,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage
from geny_executor.stages.s10_tool.state_mutation import apply_state_mutations
from geny_executor.tools.base import (
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
)
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Tool fixture
# ─────────────────────────────────────────────────────────────────


class _MutatingTool(Tool):
    """Returns a fixed ToolResult whose state_mutations are the test-facing knob."""

    def __init__(
        self,
        name: str,
        *,
        mutations: Dict[str, Any],
        is_error: bool = False,
        concurrency_safe: bool = True,
    ):
        self._name = name
        self._mutations = mutations
        self._is_error = is_error
        self._safe = concurrency_safe

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "mutating"

    @property
    def input_schema(self):
        return {"type": "object"}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=self._safe, max_result_chars=0)

    async def execute(self, input, context):
        return ToolResult(
            content="ok" if not self._is_error else "nope",
            is_error=self._is_error,
            state_mutations=dict(self._mutations),
        )


def _registry_with(tool: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


# ─────────────────────────────────────────────────────────────────
# apply_state_mutations helper — standalone
# ─────────────────────────────────────────────────────────────────


class TestApplyHelper:
    def test_applies_executor_namespaced(self):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={"executor.todos": [1, 2]})
        applied = apply_state_mutations(r, shared, tool_name="T")
        assert shared == {"executor.todos": [1, 2]}
        assert applied == {"executor.todos": [1, 2]}

    def test_applies_memory_namespaced(self):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={"memory.context_chunks": ["a"]})
        apply_state_mutations(r, shared, tool_name="T")
        assert shared == {"memory.context_chunks": ["a"]}

    def test_applies_geny_namespaced(self):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={"geny.creature_state": {"lvl": 1}})
        apply_state_mutations(r, shared, tool_name="T")
        assert shared == {"geny.creature_state": {"lvl": 1}}

    def test_applies_plugin_namespaced(self):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={"plugin.foo.bar": "z"})
        apply_state_mutations(r, shared, tool_name="T")
        assert shared == {"plugin.foo.bar": "z"}

    def test_skips_unknown_namespace(self, caplog):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={"random_key": 1})
        caplog.set_level("WARNING")
        applied = apply_state_mutations(r, shared, tool_name="T")
        assert shared == {}
        assert applied == {}
        assert any("unknown namespace" in rec.message for rec in caplog.records)

    def test_skips_on_error(self):
        shared: Dict[str, Any] = {}
        r = ToolResult(
            content="nope",
            is_error=True,
            state_mutations={"executor.todos": [1]},
        )
        applied = apply_state_mutations(r, shared, tool_name="T")
        # Error → nothing applied; shared untouched
        assert shared == {}
        assert applied == {}

    def test_empty_mutations_no_op(self):
        shared: Dict[str, Any] = {"existing": "v"}
        r = ToolResult(content="x", state_mutations={})
        apply_state_mutations(r, shared, tool_name="T")
        assert shared == {"existing": "v"}

    def test_non_string_key_skipped(self, caplog):
        shared: Dict[str, Any] = {}
        r = ToolResult(content="x", state_mutations={123: "v"})  # type: ignore[dict-item]
        caplog.set_level("WARNING")
        apply_state_mutations(r, shared, tool_name="T")
        assert shared == {}
        assert any("non-string key" in rec.message for rec in caplog.records)


# ─────────────────────────────────────────────────────────────────
# Executor integration
# ─────────────────────────────────────────────────────────────────


class TestSequentialExecutorApplies:
    @pytest.mark.asyncio
    async def test_success_mutations_reach_shared(self):
        tool = _MutatingTool("M", mutations={"executor.todos": [{"id": "x"}]})
        reg = _registry_with(tool)
        router = RegistryRouter(reg)
        execu = SequentialExecutor()

        state = PipelineState(session_id="s")
        stage = ToolStage(registry=reg, executor=execu)
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "M", "tool_input": {}}
        ]
        await stage.execute(None, state)
        assert state.shared.get("executor.todos") == [{"id": "x"}]

    @pytest.mark.asyncio
    async def test_error_result_does_not_mutate_shared(self):
        tool = _MutatingTool(
            "M",
            mutations={"executor.todos": ["leaked"]},
            is_error=True,
        )
        reg = _registry_with(tool)
        stage = ToolStage(registry=reg, executor=SequentialExecutor())
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "M", "tool_input": {}}
        ]
        await stage.execute(None, state)
        assert "executor.todos" not in state.shared


class TestParallelExecutorApplies:
    @pytest.mark.asyncio
    async def test_two_tools_both_apply(self):
        tool_a = _MutatingTool("A", mutations={"executor.todos": ["a"]})
        tool_b = _MutatingTool("B", mutations={"memory.context_chunks": ["b"]})
        reg = ToolRegistry()
        reg.register(tool_a)
        reg.register(tool_b)
        stage = ToolStage(
            registry=reg,
            executor=ParallelExecutor(max_concurrency=2),
        )
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "A", "tool_input": {}},
            {"tool_use_id": "u2", "tool_name": "B", "tool_input": {}},
        ]
        await stage.execute(None, state)
        assert state.shared.get("executor.todos") == ["a"]
        assert state.shared.get("memory.context_chunks") == ["b"]


class TestPartitionExecutorApplies:
    @pytest.mark.asyncio
    async def test_safe_and_unsafe_both_apply(self):
        safe_tool = _MutatingTool(
            "S",
            mutations={"executor.todos": [1]},
            concurrency_safe=True,
        )
        unsafe_tool = _MutatingTool(
            "U",
            mutations={"geny.mutation_buffer": ["edit"]},
            concurrency_safe=False,
        )
        reg = ToolRegistry()
        reg.register(safe_tool)
        reg.register(unsafe_tool)
        stage = ToolStage(
            registry=reg,
            executor=PartitionExecutor(registry=reg, max_concurrency=2),
        )
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "S", "tool_input": {}},
            {"tool_use_id": "u2", "tool_name": "U", "tool_input": {}},
        ]
        await stage.execute(None, state)
        assert state.shared.get("executor.todos") == [1]
        assert state.shared.get("geny.mutation_buffer") == ["edit"]


class TestStreamingExecutorApplies:
    @pytest.mark.asyncio
    async def test_streaming_apply_via_context(self):
        tool = _MutatingTool("M", mutations={"executor.todos": ["streamed"]})
        reg = _registry_with(tool)
        router = RegistryRouter(reg)
        execu = StreamingToolExecutor(registry=reg, router=router)

        # Hand-build a ctx with state_apply — mirrors what ToolStage
        # would do.
        shared: Dict[str, Any] = {}

        def _apply(mutations, tool_name):
            return apply_state_mutations(
                ToolResult(content=None, state_mutations=mutations),
                shared,
                tool_name=tool_name,
            )

        ctx = ToolContext(session_id="s", working_dir="", state_apply=_apply)
        await execu.add({"tool_use_id": "s1", "tool_name": "M", "tool_input": {}}, ctx)
        await execu.drain(ctx)
        assert shared == {"executor.todos": ["streamed"]}


class TestAbsentSink:
    @pytest.mark.asyncio
    async def test_no_sink_silently_drops(self):
        """When ToolContext.state_apply is None, mutations are silently
        discarded. Confirms the executor doesn't crash or log at error
        level — this is the path when a tool is invoked outside ToolStage
        (e.g. a host test harness)."""
        tool = _MutatingTool("M", mutations={"executor.todos": ["x"]})
        reg = _registry_with(tool)
        router = RegistryRouter(reg)
        execu = SequentialExecutor()

        ctx = ToolContext(session_id="s", working_dir="")
        assert ctx.state_apply is None
        results = await execu.execute_all(
            [{"tool_use_id": "u1", "tool_name": "M", "tool_input": {}}],
            router,
            ctx,
        )
        assert len(results) == 1
        # No crash, no state applied (there was no state to apply to).
