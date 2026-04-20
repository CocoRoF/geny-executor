"""Per-call tool event tests (v0.23.0).

Exercises the ``on_event`` kwarg on ``SequentialExecutor.execute_all``
and ``ParallelExecutor.execute_all`` — the additive event contract
introduced in executor v0.23.0 so host-side log consumers can render
per-call details (previously only summary ``tool.execute_*`` events
were emitted).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from geny_executor.stages.s10_tool.artifact.default.executors import (
    ParallelExecutor,
    SequentialExecutor,
)
from geny_executor.stages.s10_tool.interface import ToolRouter
from geny_executor.tools.base import ToolContext, ToolResult


class _RecordingRouter(ToolRouter):
    """Router that returns pre-seeded results keyed by tool name."""

    def __init__(self, outcomes: Dict[str, ToolResult], sleep_map: Dict[str, float] | None = None):
        self._outcomes = outcomes
        self._sleep_map = sleep_map or {}

    @property
    def name(self) -> str:
        return "recording"

    @property
    def description(self) -> str:
        return "test router"

    async def route(
        self, tool_name: str, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        delay = self._sleep_map.get(tool_name, 0.0)
        if delay:
            await asyncio.sleep(delay)
        return self._outcomes[tool_name]


def _tc(tool_name: str, tool_use_id: str, **input_kw: Any) -> Dict[str, Any]:
    return {
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "tool_input": input_kw,
    }


def _make_recorder() -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
    events: List[Tuple[str, Dict[str, Any]]] = []

    def _on(event_type: str, data: Dict[str, Any]) -> None:
        events.append((event_type, data))

    return events, _on


# ───────────────────────── Sequential ─────────────────────────


@pytest.mark.asyncio
async def test_sequential_emits_start_and_complete_per_call():
    router = _RecordingRouter(
        {
            "alpha": ToolResult(content="A"),
            "beta": ToolResult(content="B"),
        }
    )
    executor = SequentialExecutor()
    events, on_event = _make_recorder()

    calls = [_tc("alpha", "id_a", q="x"), _tc("beta", "id_b", q="y")]
    results = await executor.execute_all(calls, router, ToolContext(), on_event=on_event)

    assert len(results) == 2
    types = [e[0] for e in events]
    assert types == [
        "tool.call_start",
        "tool.call_complete",
        "tool.call_start",
        "tool.call_complete",
    ]

    start_a, complete_a = events[0][1], events[1][1]
    assert start_a == {"tool_use_id": "id_a", "name": "alpha", "input": {"q": "x"}}
    assert complete_a["tool_use_id"] == "id_a"
    assert complete_a["name"] == "alpha"
    assert complete_a["is_error"] is False
    assert isinstance(complete_a["duration_ms"], int)
    assert complete_a["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_sequential_reports_is_error_true_for_failing_tool():
    router = _RecordingRouter({"busted": ToolResult(content="boom", is_error=True)})
    events, on_event = _make_recorder()

    await SequentialExecutor().execute_all(
        [_tc("busted", "id_x")], router, ToolContext(), on_event=on_event
    )

    [(start_t, _), (_, complete_data)] = events
    assert start_t == "tool.call_start"
    assert complete_data["is_error"] is True


@pytest.mark.asyncio
async def test_sequential_on_event_is_optional():
    """Omitting on_event matches pre-0.23.0 behavior — no errors, no events."""
    router = _RecordingRouter({"x": ToolResult(content="ok")})
    results = await SequentialExecutor().execute_all([_tc("x", "id_only")], router, ToolContext())
    assert len(results) == 1


# ───────────────────────── Parallel ─────────────────────────


@pytest.mark.asyncio
async def test_parallel_emits_paired_events_per_call():
    router = _RecordingRouter(
        {
            "p1": ToolResult(content="1"),
            "p2": ToolResult(content="2"),
            "p3": ToolResult(content="3"),
        }
    )
    events, on_event = _make_recorder()

    calls = [
        _tc("p1", "u1", q=1),
        _tc("p2", "u2", q=2),
        _tc("p3", "u3", q=3),
    ]
    results = await ParallelExecutor(max_concurrency=3).execute_all(
        calls, router, ToolContext(), on_event=on_event
    )

    assert len(results) == 3

    starts = [d for t, d in events if t == "tool.call_start"]
    completes = [d for t, d in events if t == "tool.call_complete"]
    assert len(starts) == 3
    assert len(completes) == 3

    start_ids = {d["tool_use_id"] for d in starts}
    complete_ids = {d["tool_use_id"] for d in completes}
    assert start_ids == complete_ids == {"u1", "u2", "u3"}

    for d in completes:
        assert d["is_error"] is False
        assert isinstance(d["duration_ms"], int)
        assert d["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_parallel_on_event_is_optional():
    router = _RecordingRouter({"x": ToolResult(content="ok")})
    results = await ParallelExecutor().execute_all([_tc("x", "id_only")], router, ToolContext())
    assert len(results) == 1


# ───────────── Stage-level pairing with tool.execute_* ─────────────


@pytest.mark.asyncio
async def test_stage_wraps_call_events_inside_execute_events():
    """``tool.execute_start`` must precede first ``tool.call_start`` and
    ``tool.execute_complete`` must follow last ``tool.call_complete`` —
    the stage remains the outermost bracket, per-call events nest inside.
    """
    from geny_executor import PipelineState
    from geny_executor.stages.s10_tool import ToolStage
    from geny_executor.tools.registry import ToolRegistry
    from geny_executor.tools.base import Tool

    class _Echo(Tool):
        @property
        def name(self) -> str:
            return "echo"

        @property
        def description(self) -> str:
            return "echo"

        @property
        def input_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, input, context):
            return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(_Echo())

    stage = ToolStage(registry=registry)
    state = PipelineState(
        messages=[],
        pending_tool_calls=[
            _tc("echo", "u1"),
            _tc("echo", "u2"),
        ],
    )

    await stage.execute(None, state)

    event_types = [e["type"] for e in state.events]
    assert event_types[0] == "tool.execute_start"
    assert event_types[-1] == "tool.execute_complete"

    call_indices = [i for i, t in enumerate(event_types) if t.startswith("tool.call_")]
    assert call_indices, "expected per-call events between execute_start/complete"
    assert call_indices[0] > 0
    assert call_indices[-1] < len(event_types) - 1
    mid = [event_types[i] for i in call_indices]
    assert mid == [
        "tool.call_start",
        "tool.call_complete",
        "tool.call_start",
        "tool.call_complete",
    ]
