"""2.77.0 — the tool loop's safeguards.

Each test pins one defect found by the 2026-09-29 harness audit, most of
them measured first on XGEN's runtime (the same stack):

* parallel tool calls ran out of the order the model wrote them;
* a string where the schema wants a number, a JSON string where it wants an
  array, a mixed-up parameter name — each a rejected call, often repeated;
* unreadable tool-call arguments were reported as a missing field;
* a refused call was asked again, and again;
* ``Write`` replaced files the agent had never seen;
* an MCP call and a stalled model could hold a turn forever;
* a turn could read without bound, and one that reached its step limit
  stopped mid-work without answering;
* a long command's output kept its start and dropped the error at its end;
* a tool failure reached the host without its reason.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s10_tool.artifact.default.executors import (
    PartitionExecutor,
    SequentialExecutor,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.stages.s10_tool import repeat_guard
from geny_executor.stages.s10_tool.state_mutation import apply_state_mutations
from geny_executor.stages.s16_loop import LoopStage
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.built_in.bash_tool import _head_tail
from geny_executor.tools.built_in.read_tool import ReadTool
from geny_executor.tools.built_in.tool_search_tool import _named_in_query
from geny_executor.tools.built_in.write_tool import WriteTool
from geny_executor.tools.errors import ToolError, make_error_result
from geny_executor.tools.input_repair import UNPARSED_ARGUMENTS_KEY, coerce_input
from geny_executor.tools.registry import ToolRegistry


def _run(coro):
    return asyncio.run(coro)


class _Echo(Tool):
    def __init__(self, name: str, schema: Dict[str, Any], *, safe: bool = False, fail: str = ""):
        self._name = name
        self._schema = schema
        self._safe = safe
        self._fail = fail
        self.calls: List[Dict[str, Any]] = []
        self.log: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._schema

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=self._safe)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.calls.append(dict(input))
        if self._fail:
            return make_error_result(ToolError.access_denied(self._name, self._fail))
        return ToolResult(content=f"{self._name} ok {input}")


def _router(*tools: Tool) -> RegistryRouter:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return RegistryRouter(reg)


# ── input repair ────────────────────────────────────────────────────

_SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "exact": {"type": "boolean"},
    },
    "required": ["query"],
}


def test_plain_meaning_strings_are_fixed_before_validation():
    tool = _Echo("Search", _SEARCH)
    result = _run(
        _router(tool).route(
            "Search",
            {"query": "q", "max_results": "3", "tags": '["a", "b"]', "exact": "TRUE"},
            ToolContext(),
        )
    )
    assert not result.is_error
    assert tool.calls == [{"query": "q", "max_results": 3, "tags": ["a", "b"], "exact": True}]
    # A string field keeps its string.
    assert coerce_input({"type": "string"}, "3") == "3"


def test_a_mixed_up_parameter_name_is_moved_and_said():
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    }
    tool = _Echo("Open", schema)
    result = _run(_router(tool).route("Open", {"file_path": "/a.txt"}, ToolContext()))
    assert tool.calls == [{"path": "/a.txt"}]
    assert result.content.startswith("[input repaired] 'file_path' is not a parameter")


def test_a_missing_field_error_says_what_was_sent():
    tool = _Echo("Open", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
    result = _run(_router(tool).route("Open", {"a": 1, "b": 2}, ToolContext()))
    assert result.is_error
    text = result.to_api_format("t")["content"]
    assert "required: path; you sent: a, b" in text


def test_unreadable_arguments_are_not_called_a_missing_field():
    tool = _Echo("Open", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
    result = _run(
        _router(tool).route("Open", {UNPARSED_ARGUMENTS_KEY: '{"path": "/a'}, ToolContext())
    )
    text = result.to_api_format("t")["content"]
    assert "not valid JSON" in text and "required" not in text
    assert tool.calls == []


# ── order, refusals, failure text ───────────────────────────────────


class _Ordered(Tool):
    def __init__(self, name: str, safe: bool, log: List[str]):
        self._name, self._safe, self._log = name, safe, log

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=self._safe)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self._log.append(f"start {self._name}")
        await asyncio.sleep(0.02)
        self._log.append(f"end {self._name}")
        return ToolResult(content=self._name)


def test_a_write_then_a_read_run_in_that_order():
    log: List[str] = []
    write, read = _Ordered("Write", False, log), _Ordered("Read", True, log)
    reg = ToolRegistry()
    reg.register(write)
    reg.register(read)
    calls = [
        {"tool_use_id": "1", "tool_name": "Write", "tool_input": {}},
        {"tool_use_id": "2", "tool_name": "Read", "tool_input": {}},
    ]
    _run(PartitionExecutor(registry=reg).execute_all(calls, RegistryRouter(reg), ToolContext()))
    assert log == ["start Write", "end Write", "start Read", "end Read"]


def test_a_refused_call_is_not_asked_again_this_turn():
    shared: Dict[str, Any] = {}
    state = PipelineState(session_id="s")
    state.shared = shared
    tool = _Echo("Bash", {"type": "object"}, fail="the user said no")
    router = _router(tool)
    call = {"tool_use_id": "1", "tool_name": "Bash", "tool_input": {"command": "rm -rf x"}}

    async def round_(tc):
        pre, runnable, blocked, skipped = repeat_guard.guard_calls([tc], shared)
        executed = await SequentialExecutor().execute_all(runnable, router, ToolContext())
        results = repeat_guard.merge_in_order([tc], pre, executed)
        repeat_guard.report(state, [tc], results, blocked, skipped)
        return results

    _run(round_(call))
    again = dict(call, tool_use_id="2", tool_input={"command": "rm  -rf 'x'"})
    results = _run(round_(again))
    assert len(tool.calls) == 1  # the retry never ran — no second prompt
    assert results[0]["content"].startswith("ERROR user_denied_repeat")
    assert any(e["type"] == "tool.user_denied" for e in state.events)


def test_a_failure_carries_its_reason_to_the_host():
    events: List[tuple] = []
    tool = _Echo("Bash", {"type": "object"}, fail="blocked by policy")
    calls = [{"tool_use_id": "1", "tool_name": "Bash", "tool_input": {}}]
    _run(
        SequentialExecutor().execute_all(
            calls, _router(tool), ToolContext(), on_event=lambda t, d: events.append((t, d))
        )
    )
    done = [d for t, d in events if t == "tool.call_complete"][0]
    assert "blocked by policy" in done["error"]


# ── read before overwrite ───────────────────────────────────────────


def _file_ctx(tmp_path, state):
    return ToolContext(working_dir=str(tmp_path), state_view=state)


def _apply(result, state):
    apply_state_mutations(result, state.shared, tool_name="t")
    return result


def test_an_unread_file_is_not_overwritten(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("the user's work")
    state = PipelineState(session_id="s")
    ctx = _file_ctx(tmp_path, state)

    refused = _run(WriteTool().execute({"file_path": "notes.md", "content": "new"}, ctx))
    assert refused.is_error and "have not read it" in refused.content
    assert target.read_text() == "the user's work"

    _apply(_run(ReadTool().execute({"file_path": "notes.md"}, ctx)), state)
    written = _apply(_run(WriteTool().execute({"file_path": str(target), "content": "new"}, ctx)), state)
    assert not written.is_error and target.read_text() == "new"
    # Its own write is known: writing again is fine.
    again = _run(WriteTool().execute({"file_path": "notes.md", "content": "newer"}, ctx))
    assert not again.is_error


def test_new_files_and_parallel_reads(tmp_path):
    state = PipelineState(session_id="s")
    ctx = _file_ctx(tmp_path, state)
    assert not _run(WriteTool().execute({"file_path": "fresh.txt", "content": "x"}, ctx)).is_error
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(name)
    # Two reads computed before either applies — both must stay in the ledger.
    ra = _run(ReadTool().execute({"file_path": "a.txt"}, ctx))
    rb = _run(ReadTool().execute({"file_path": "b.txt"}, ctx))
    _apply(ra, state)
    _apply(rb, state)
    for name in ("a.txt", "b.txt"):
        assert not _run(WriteTool().execute({"file_path": name, "content": "y"}, ctx)).is_error


# ── timeouts ────────────────────────────────────────────────────────


def test_an_mcp_call_that_never_answers_times_out():
    from geny_executor.tools.mcp.manager import MCPServerConfig, MCPServerConnection
    from geny_executor.tools.mcp.state import MCPConnectionState

    class _Hung:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(30)

    conn = MCPServerConnection(MCPServerConfig(name="slow", call_timeout_s=0.1))
    conn._client_session = _Hung()
    conn._state = MCPConnectionState.CONNECTED
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="did not answer"):
        _run(conn.call_tool("search", {}))
    assert time.monotonic() - t0 < 5


def test_a_stalled_stream_times_out_and_is_retried_once(monkeypatch):
    from geny_executor.stages.s06_api.artifact.default import stage as s06

    monkeypatch.setattr(s06, "first_chunk_timeout_s", lambda: 0.1)
    monkeypatch.setattr(s06, "idle_timeout_s", lambda: 0.1)

    opened: List[int] = []

    async def stalled():
        opened.append(1)
        yield {"type": "message_start"}
        await asyncio.sleep(30)
        yield {"type": "text_delta", "text": "never"}

    async def go():
        seen = []
        with pytest.raises(APIError) as info:
            async for chunk in s06._watched_stream(stalled(), 0.1, 0.1):
                seen.append(chunk)
        return info.value

    err = _run(go())
    assert err.category is ErrorCategory.TIMEOUT
    assert "did not start answering" in str(err)

    stage = s06.APIStage()
    calls: List[int] = []

    async def fake_streaming(client, cfg, state, *, extra_messages=None):
        calls.append(1)
        raise APIError("stalled", category=ErrorCategory.TIMEOUT)

    monkeypatch.setattr(stage, "_call_streaming", fake_streaming)
    monkeypatch.setattr(s06.asyncio, "sleep", _no_sleep)
    with pytest.raises(APIError):
        _run(stage._call_streaming_with_retry(None, None, PipelineState(session_id="s")))
    assert len(calls) == 2  # the call and ONE retry, not four stalls


async def _no_sleep(_delay):
    return None


# ── how a turn ends ─────────────────────────────────────────────────


def _loop_state(used: int, iteration: int = 1, max_iterations: int = 50) -> PipelineState:
    state = PipelineState(session_id="s")
    state.iteration = iteration
    state.max_iterations = max_iterations
    state.pending_tool_calls = [{"id": "t"}]
    state.turn_token_usage = [TokenUsage(input_tokens=used)]
    state.messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "X", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
    ]
    return state


def _note(state) -> str:
    return state.messages[-1]["content"][0]["content"]


def test_a_turn_past_its_input_budget_reports_and_ends():
    stage = LoopStage(turn_soft_input_tokens=100, turn_hard_input_tokens=1000)
    soft = _loop_state(500)
    _run(stage.execute("in", soft))
    assert "Wrap up now" in _note(soft) and soft.loop_decision == "continue"

    hard = _loop_state(5000)
    _run(stage.execute("in", hard))
    assert "Do not call any more tools" in _note(hard) and hard.loop_decision == "continue"
    # The next response — even one calling a tool — ends the turn.
    hard.turn_token_usage.append(TokenUsage(input_tokens=10))
    hard.loop_decision = "continue"
    _run(stage.execute("in", hard))
    assert hard.loop_decision == "complete"
    assert hard.completion_signal == "TURN_INPUT_BUDGET"


def test_the_step_before_the_last_says_so():
    stage = LoopStage(turn_hard_input_tokens=0)
    early = _loop_state(10, iteration=3, max_iterations=10)
    _run(stage.execute("in", early))
    assert "Step limit" not in _note(early)
    late = _loop_state(10, iteration=8, max_iterations=10)
    _run(stage.execute("in", late))
    assert "Step limit" in _note(late)


# ── output and search ───────────────────────────────────────────────


def test_a_long_output_keeps_its_end():
    text = "progress\n" * 50_000 + "ERROR: the real failure"
    out = _head_tail(text, 1000)
    assert out.endswith("ERROR: the real failure")
    assert out.startswith("progress") and "characters omitted" in out
    assert _head_tail("short", 1000) == "short"


def test_several_exact_names_find_them_all():
    descs = [{"name": n} for n in ("WebFetch", "WebSearch", "Read")]
    assert [d["name"] for d in _named_in_query(descs, "WebSearch, webfetch")] == [
        "WebSearch",
        "WebFetch",
    ]
    assert _named_in_query(descs, "web search tools") == []
    assert _named_in_query(descs, "Read") == []  # one name: the normal ranking
