"""Calls that cannot produce anything new are not made — and a turn that
keeps making them ends.

Geny had no guard at all. The thresholds are the sibling runtime's, which
were chosen on measured turns (13 of 14 wasteful loops caught, one false
positive in 55 normal turns), not here; what these tests pin is that the
behaviour holds in THIS pipeline, through the real Stage 10 and Stage 16.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool import repeat_guard
from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage
from geny_executor.stages.s16_loop.artifact.default.stage import LoopStage
from geny_executor.stages.s16_loop.repeat_stop import repeat_stopped
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry


class _Tool(Tool):
    """Answers from a script; counts how often it actually ran."""

    def __init__(self, name: str, answer=lambda i: ToolResult(content="same")) -> None:
        self._name = name
        self._answer = answer
        self.runs = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "scripted"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.runs += 1
        return self._answer(input)


def _stage(*tools: Tool) -> ToolStage:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return ToolStage(registry=registry)


def _round(stage: ToolStage, state: PipelineState, name: str, tool_input: Dict[str, Any], n: int):
    state.pending_tool_calls = [
        {"tool_use_id": f"{name}-{n}", "tool_name": name, "tool_input": tool_input}
    ]
    asyncio.run(stage.execute(None, state))
    return state.tool_results[0]


def _types(state: PipelineState) -> List[str]:
    return [e["type"] for e in state.events]


class TestTheSameCallTheSameAnswer:
    def test_the_fifth_identical_call_is_not_run(self) -> None:
        """``pwd`` 21 times, one memory write 62 times — and a repeated write
        is a repeated side effect, not only a wasted round-trip."""
        tool = _Tool("pwd")
        stage, state = _stage(tool), PipelineState()
        results = [_round(stage, state, "pwd", {}, n) for n in range(6)]
        assert tool.runs == 4
        assert "same call, same result, 4 times" in results[3]["content"]
        assert "not executed" in results[4]["content"] and "same" in results[4]["content"]
        assert "tool.same_result" in _types(state)

    def test_a_different_answer_is_new_information(self) -> None:
        counter = iter(range(100))
        tool = _Tool("poll", answer=lambda i: ToolResult(content=f"state {next(counter)}"))
        stage, state = _stage(tool), PipelineState()
        for n in range(8):
            _round(stage, state, "poll", {}, n)
        assert tool.runs == 8

    def test_different_inputs_are_different_calls(self) -> None:
        tool = _Tool("Read")
        stage, state = _stage(tool), PipelineState()
        for n in range(8):
            _round(stage, state, "Read", {"file_path": f"f{n}"}, n)
        assert tool.runs == 8


class TestTheSameFailureAgain:
    def test_an_input_error_blocks_even_as_the_arguments_change(self) -> None:
        """The recorded loop changed only the query — fifteen times."""
        tool = _Tool("search", answer=lambda i: ToolResult(
            content="ERROR invalid_input: '3' is not of type 'integer'", is_error=True))
        stage, state = _stage(tool), PipelineState()
        results = [_round(stage, state, "search", {"q": f"try {n}", "max": "3"}, n)
                   for n in range(6)]
        assert tool.runs == 4
        assert "failed the same way 3 times" in results[2]["content"]
        assert results[4]["content"].startswith("ERROR repeated_failure_blocked")
        assert {"tool.repeat_failure", "tool.repeat_blocked"} <= set(_types(state))

    def test_a_fix_and_rebuild_loop_is_not_blocked(self) -> None:
        """Editing a file between failing builds is the job, not a loop."""
        build = _Tool("Bash", answer=lambda i: ToolResult(content="build failed", is_error=True))
        edit = _Tool("Edit", answer=lambda i: ToolResult(content="edited"))
        stage, state = _stage(build, edit), PipelineState()
        for n in range(6):
            _round(stage, state, "Bash", {"command": "make"}, n)
            _round(stage, state, "Edit", {"file_path": "a.c", "n": n}, n)
        assert build.runs == 6

    def test_one_failure_for_many_items_is_not_one_loop(self) -> None:
        """Five products each "not found" is five answers, not a loop —
        until it clearly is one."""
        tool = _Tool("lookup", answer=lambda i: ToolResult(content="not found", is_error=True))
        stage, state = _stage(tool), PipelineState()
        for n in range(5):
            _round(stage, state, "lookup", {"id": n}, n)
        assert tool.runs == 5
        for n in range(5, 12):
            _round(stage, state, "lookup", {"id": n}, n)
        assert tool.runs == repeat_guard.ANY_INPUT_BLOCK_AT


class TestTheTurnEnds:
    def test_after_three_refusals_the_next_response_ends_the_turn(self) -> None:
        tool = _Tool("pwd")
        tools, loop, state = _stage(tool), LoopStage(), PipelineState()
        for n in range(7):
            state.iteration = n
            _round(tools, state, "pwd", {}, n)
            asyncio.run(loop.execute(None, state))
            if state.loop_decision != "continue":
                break
        note = str(state.messages[-1]["content"][-1]["content"])
        assert "Do not call any more tools" in note
        # the response to the note comes next, and ends the turn
        state.iteration += 1
        state.loop_decision = None
        state.add_message("assistant", [{"type": "text", "text": "Done: pwd is /w."}])
        asyncio.run(loop.execute(None, state))
        assert state.loop_decision == "complete"
        assert state.completion_signal == "REPEAT_STOP"
        assert repeat_stopped(state)["refused"] >= 3

    def test_it_can_be_turned_off(self) -> None:
        tool = _Tool("pwd")
        tools, loop, state = _stage(tool), LoopStage(repeat_stop_after=0), PipelineState()
        for n in range(9):
            state.iteration = n
            _round(tools, state, "pwd", {}, n)
            asyncio.run(loop.execute(None, state))
        assert repeat_stopped(state) is None
        assert "loop.repeat_stop" not in _types(state)
