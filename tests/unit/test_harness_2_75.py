"""2.75.0 — the previous turns reach the model as messages only, a stop
stops, and a turn that breaks is still remembered.

Each test pins one defect found by the 2026-09-29 harness audit (Geny's
production turns, read against the code):

* a second retrieval pass rendered the previous turns as text —
  ``[recent_message] stm-N: [assistant] [{'type': 'tool_use', …}]`` —
  next to the same turns replayed as messages;
* the recent-turns text layer came back whenever the replay stood down,
  counting rows and keeping text only;
* cancelling a streaming consumer left the run task calling tools and
  the model in the background;
* a turn that failed or was stopped recorded nothing;
* ``[COMPLETE]`` written next to a tool call ended the turn before the
  model saw the tool's result;
* a history with an unanswered call anywhere but the end 400'd every
  request after it;
* Stage 19 rewrote ``summary.md`` on every turn, wiping the rolling
  digest kept there.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List

import pytest

from geny_executor import Pipeline, PipelineState
from geny_executor.core.message_repair import (
    normalize_messages_for_request,
    repair_all_tool_pairs,
)
from geny_executor.core.stage import Stage
from geny_executor.memory.provider import MemoryHooks, Turn
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever
from geny_executor.memory.short_term_window import (
    UNANSWERED_NOTE,
    WindowConfig,
    build_window,
    render_turns_as_text,
)
from geny_executor.memory.turn_text import turn_text
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
from geny_executor.stages.s14_evaluate.artifact.default.strategies import SignalBasedEvaluation


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc)


def _rows(*pairs: Any) -> List[Turn]:
    return [Turn(role=r, content=c, timestamp=_now()) for r, c in pairs]


def _tool_turn(question: str, tool: str, command: str, result: str, answer: str) -> list:
    return [
        ("user", question),
        (
            "assistant",
            [{"type": "tool_use", "id": f"id-{question}", "name": tool, "input": {"command": command}}],
        ),
        ("user", [{"type": "tool_result", "tool_use_id": f"id-{question}", "content": result}]),
        ("assistant", answer),
    ]


# ── the second retrieval pass ────────────────────────────────────────


def test_the_provider_pass_does_not_run_when_the_retriever_reads_memory():
    """Geny attaches the provider to Stage 2 so the replay can read it; the
    retriever already reads the same memory. Nothing from the provider's
    short-term layer may come back as text."""
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        for role, content in _tool_turn("q1", "Bash", "ls", "a.txt", "done"):
            await provider.stm().append(Turn(role=role, content=content, timestamp=_now()))
        stage = ContextStage(retriever=MemoryAwareRetriever(provider, hooks=MemoryHooks(recent_turns=0)))
        stage.provider = provider
        state = PipelineState(session_id="s")
        state.messages = [{"role": "user", "content": "what did you run?"}]
        await stage.execute(None, state)
        return state

    state = _run(go())
    context = state.metadata.get("memory_context") or ""
    assert "recent_message" not in context
    assert "stm-" not in context
    assert "'type'" not in context  # no Python repr of content blocks


def test_a_provider_row_is_prose_not_a_repr():
    text = turn_text(
        "assistant",
        [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "ls"}, "cache_control": {}},
        ],
    )
    assert text == '[assistant] checking [called Bash({"command": "ls"})]'
    assert "cache_control" not in text


# ── the recent-turns text layer ──────────────────────────────────────


def test_the_text_layer_counts_logical_turns_and_shows_the_tools():
    rows = _rows(
        *_tool_turn("first", "Bash", "ls", "a.txt", "found a.txt"),
        *_tool_turn("second", "Read", "a.txt", "hello", "it says hello"),
        ("user", "third"),
        ("assistant", "[SILENT]"),
    )
    body, count = render_turns_as_text(rows, turns=2, max_chars=4000)
    # Two turns asked for, eight rows of tool work in them; the silent one
    # takes no slot.
    assert count == 2
    assert "[user] first" in body and "[user] second" in body
    assert "[used tools: Bash(ls)]" in body
    assert "[SILENT]" not in body


def test_the_text_layer_stands_down_when_the_history_is_already_in_the_messages():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        for role, content in [("user", "earlier"), ("assistant", "earlier answer")]:
            await provider.stm().append(Turn(role=role, content=content, timestamp=_now()))
        retriever = MemoryAwareRetriever(
            provider, hooks=MemoryHooks(recent_turns=3, slim_mode=True, always_render_vault_map=False)
        )
        state = PipelineState(session_id="s")
        state.messages = [{"role": "user", "content": "now"}]
        state.metadata["memory.history_in_messages"] = "host"
        return await retriever.retrieve("now", state)

    chunks = _run(go())
    assert not [c for c in chunks if (c.metadata or {}).get("layer") == "recent_turns"]


def test_a_turn_that_was_never_answered_is_kept_and_closed_off():
    rows = _rows(("user", "please remember my cat is Nabi"), ("user", "and what's her name?"), ("assistant", "Nabi"))
    window = build_window(rows, WindowConfig(full_turns=2, dialogue_turns=3, max_tokens=5000))
    texts = [
        "".join(b.get("text", "") for b in m["content"]) if isinstance(m["content"], list) else m["content"]
        for m in window.messages
    ]
    assert "please remember my cat is Nabi" in texts[0]
    assert texts[1] == UNANSWERED_NOTE
    roles = [m["role"] for m in window.messages]
    assert all(a != b for a, b in zip(roles, roles[1:]))


# ── request-boundary pairing ────────────────────────────────────────


def test_every_unanswered_call_is_answered_on_the_wire_not_in_the_store():
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "old", "name": "X", "input": {}}]},
        {"role": "user", "content": "a stop happened, then a new question"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "new", "name": "Y", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new", "content": "ok"}]},
    ]
    sent = normalize_messages_for_request(history)
    answered = {
        b["tool_use_id"]
        for m in sent
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    }
    assert answered == {"old", "new"}
    assert len(history) == 5  # the stored history is untouched
    assert normalize_messages_for_request(sent) == sent


def test_repair_in_place_reports_whether_it_changed_anything():
    clean = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert repair_all_tool_pairs(clean) is False
    broken = [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "X", "input": {}}]}]
    assert repair_all_tool_pairs(broken) is True
    assert broken[-1]["content"][0]["tool_use_id"] == "x"


# ── completion signal vs tool results ────────────────────────────────


def test_complete_next_to_a_tool_call_does_not_end_the_turn():
    state = PipelineState(session_id="s")
    state.iteration = 0
    state.tool_iteration = 0
    state.completion_signal = "complete"
    state.tool_results = [{"tool_use_id": "t", "content": "42"}]
    result = _run(SignalBasedEvaluation().evaluate(state))
    assert result.decision == "continue"


# ── a stop stops; a broken turn is remembered ────────────────────────


class _Gate(Stage):
    def __init__(self, order: int) -> None:
        self._order = order
        self.entered = asyncio.Event()
        self.cancelled = False

    @property
    def name(self) -> str:
        return f"gate{self._order}"

    @property
    def order(self) -> int:
        return self._order

    async def execute(self, input, state):  # noqa: ANN001
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return input


class _Recorder(Stage):
    """Stands in for Stage 18: remembers what it was asked to record."""

    def __init__(self) -> None:
        self.recorded: List[Any] = []

    @property
    def name(self) -> str:
        return "memory"

    @property
    def order(self) -> int:
        return 18

    async def execute(self, input, state):  # noqa: ANN001
        self.recorded = list(state.messages)
        return input


class _Boom(Stage):
    @property
    def name(self) -> str:
        return "boom"

    @property
    def order(self) -> int:
        return 3

    async def execute(self, input, state):  # noqa: ANN001
        state.messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "Write", "input": {}}]}
        )
        raise RuntimeError("provider exploded")


def test_cancelling_the_stream_consumer_cancels_the_run():
    gate = _Gate(2)
    recorder = _Recorder()
    pipeline = Pipeline()
    pipeline.register_stage(InputStage())
    pipeline.register_stage(gate)
    pipeline.register_stage(recorder)

    async def go():
        state = PipelineState(session_id="s")
        events: List[str] = []

        async def consume():
            async for event in pipeline.run_stream("hello", state):
                events.append(event.type)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(gate.entered.wait(), 5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await asyncio.sleep(0)
        return state, events

    state, _ = _run(go())
    assert gate.cancelled, "the run kept going after its consumer was stopped"
    assert pipeline._runs_in_flight == 0
    # The stopped turn is recorded, closed off as stopped.
    assert recorder.recorded
    last = recorder.recorded[-1]
    assert last["role"] == "assistant"
    assert "stopped before it finished" in last["content"][0]["text"]
    assert state.metadata["turn.interrupted"]["reason"] == "stopped"


def test_a_failed_turn_is_recorded_with_its_calls_answered():
    recorder = _Recorder()
    pipeline = Pipeline()
    pipeline.register_stage(InputStage())
    pipeline.register_stage(_Boom())
    pipeline.register_stage(recorder)

    async def go():
        state = PipelineState(session_id="s")
        return await pipeline.run("write the file", state)

    result = _run(go())
    assert not result.success
    kinds = [
        (m["role"], b.get("type"))
        for m in recorder.recorded
        if isinstance(m.get("content"), list)
        for b in m["content"]
    ]
    assert ("assistant", "tool_use") in kinds and ("user", "tool_result") in kinds
    assert "ended with an error" in recorder.recorded[-1]["content"][0]["text"]
    assert recorder.recorded[0]["content"] == "write the file"
