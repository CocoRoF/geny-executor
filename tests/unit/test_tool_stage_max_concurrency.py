"""Phase 2 Week 4 Checkpoint 4 — ToolStage.max_concurrency config tests.

Confirms that the stage exposes a ``max_concurrency`` ConfigField,
propagates it to the active executor on construction, re-applies it on
every ``execute()`` call (so swapped-in executors inherit the budget),
and that ``update_config`` takes effect immediately.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from geny_executor.core.schema import ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool import (
    ParallelExecutor,
    PartitionExecutor,
    SequentialExecutor,
)
from geny_executor.stages.s10_tool.artifact.default.stage import (
    _DEFAULT_MAX_CONCURRENCY,
    ToolStage,
)
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry


class _CountingTool(Tool):
    """Safe tool — capability flag ensures PartitionExecutor batches in parallel."""

    def __init__(self, name: str = "counter"):
        self._name = name
        self.in_flight = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "counting"

    @property
    def input_schema(self):
        return {"type": "object"}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, max_result_chars=0)

    async def execute(self, input, context):
        async with self._lock:
            self.in_flight += 1
            if self.in_flight > self.peak:
                self.peak = self.in_flight
        await asyncio.sleep(0.02)
        async with self._lock:
            self.in_flight -= 1
        return ToolResult(content="ok")


def _calls(n: int) -> List[Dict[str, Any]]:
    return [
        {"tool_use_id": f"u{i}", "tool_name": "counter", "tool_input": {}}
        for i in range(n)
    ]


def _state_with(calls: List[Dict[str, Any]]) -> PipelineState:
    state = PipelineState(session_id="s")
    state.pending_tool_calls = list(calls)
    return state


# ─────────────────────────────────────────────────────────────────
# Schema surface
# ─────────────────────────────────────────────────────────────────


class TestSchemaSurface:
    def test_schema_exposes_max_concurrency(self):
        stage = ToolStage()
        schema = stage.get_config_schema()
        assert isinstance(schema, ConfigSchema)
        names = {f.name for f in schema.fields}
        assert "max_concurrency" in names
        field = next(f for f in schema.fields if f.name == "max_concurrency")
        assert field.type == "integer"
        assert field.default == _DEFAULT_MAX_CONCURRENCY
        assert field.min_value == 1

    def test_get_config_defaults(self):
        stage = ToolStage()
        cfg = stage.get_config()
        assert cfg == {"max_concurrency": _DEFAULT_MAX_CONCURRENCY}

    def test_update_config_stores_value(self):
        stage = ToolStage()
        stage.update_config({"max_concurrency": 3})
        assert stage.get_config() == {"max_concurrency": 3}

    def test_update_config_clamps_below_one_to_one(self):
        stage = ToolStage()
        stage.update_config({"max_concurrency": 0})
        assert stage.get_config() == {"max_concurrency": 1}

    def test_ctor_arg_propagates(self):
        stage = ToolStage(max_concurrency=2)
        assert stage.get_config() == {"max_concurrency": 2}

    def test_schema_validates_below_min(self):
        schema = ToolStage().get_config_schema()
        errors = schema.validate({"max_concurrency": 0})
        assert errors and any("max_concurrency" in e for e in errors)


# ─────────────────────────────────────────────────────────────────
# Runtime propagation to executors
# ─────────────────────────────────────────────────────────────────


class TestPropagationToExecutors:
    def test_default_executor_unchanged_is_sequential(self):
        # Default stays SequentialExecutor to preserve legacy callers — the
        # max_concurrency knob is a no-op until a host swaps to parallel /
        # partition. Check ctor doesn't raise when the executor lacks the
        # attribute.
        stage = ToolStage(max_concurrency=7)
        executor = stage.get_strategy_slots()["executor"].strategy
        assert isinstance(executor, SequentialExecutor)
        assert stage.get_config() == {"max_concurrency": 7}

    def test_injected_parallel_executor_gets_updated(self):
        stage = ToolStage(executor=ParallelExecutor(max_concurrency=20), max_concurrency=3)
        # Ctor value wins — stage knob is the authority
        executor = stage.get_strategy_slots()["executor"].strategy
        assert executor._max_concurrency == 3

    def test_sequential_executor_accepts_stage_without_breakage(self):
        stage = ToolStage(executor=SequentialExecutor(), max_concurrency=5)
        # SequentialExecutor has no _max_concurrency — nothing to set,
        # nothing should raise.
        assert stage.get_config() == {"max_concurrency": 5}

    @pytest.mark.asyncio
    async def test_update_config_takes_effect_on_next_execute(self, tmp_path):
        """After update_config, the next execute() run must respect the new cap."""
        tool = _CountingTool()
        reg = ToolRegistry()
        reg.register(tool)
        stage = ToolStage(
            registry=reg,
            executor=ParallelExecutor(max_concurrency=_DEFAULT_MAX_CONCURRENCY),
            max_concurrency=_DEFAULT_MAX_CONCURRENCY,
        )

        # First: cap down to 2 and fire a wide batch
        stage.update_config({"max_concurrency": 2})
        state = _state_with(_calls(8))
        await stage.execute(None, state)
        assert tool.peak <= 2, f"expected <=2, got {tool.peak}"

        # Reset counter and widen to 8
        tool.peak = 0
        stage.update_config({"max_concurrency": 8})
        state = _state_with(_calls(8))
        await stage.execute(None, state)
        assert tool.peak >= 3, (
            f"expected broader parallelism, peak stayed at {tool.peak}"
        )

    @pytest.mark.asyncio
    async def test_budget_survives_executor_swap(self, tmp_path):
        """Swapping the executor via StrategySlot.swap must inherit the
        current max_concurrency from the stage."""
        tool = _CountingTool()
        reg = ToolRegistry()
        reg.register(tool)
        stage = ToolStage(registry=reg, max_concurrency=2)

        # Swap to PartitionExecutor (zero-arg constructor inside swap())
        slot = stage.get_strategy_slots()["executor"]
        slot.swap("partition")
        new_exec = slot.strategy
        assert isinstance(new_exec, PartitionExecutor)
        # Straight after swap the executor carries its own class default
        # (10) — the stage reapplies our budget on execute().
        state = _state_with(_calls(6))
        await stage.execute(None, state)
        assert tool.peak <= 2, f"swap budget leaked — peak {tool.peak}"
        # And the executor field now reflects the stage knob
        assert new_exec._max_concurrency == 2


# ─────────────────────────────────────────────────────────────────
# ToolContext still carries storage_path etc unchanged
# ─────────────────────────────────────────────────────────────────


class TestContextUnchanged:
    @pytest.mark.asyncio
    async def test_context_fields_forwarded(self, tmp_path):
        tool = _CountingTool()
        reg = ToolRegistry()
        reg.register(tool)
        stage = ToolStage(
            registry=reg,
            context=ToolContext(storage_path=str(tmp_path), working_dir="/w"),
            max_concurrency=2,
        )
        state = _state_with(_calls(2))
        await stage.execute(None, state)
        # No assertion on result — just making sure the pipeline ran end to end
        # with max_concurrency wired and no context regression.
        assert state.tool_results and len(state.tool_results) == 2
