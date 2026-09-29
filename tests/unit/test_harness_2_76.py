"""2.76.0 — compaction keeps what it removes, and a closed pipeline stops.

Each test pins one defect found by the 2026-09-29 harness audit:

* a turn long enough to compact lost its own request and first tool calls
  from memory — the watermark jumped over them;
* a turn compacted before anything was recorded wrote the summary into
  memory as if it had been said;
* without a per-stage model the "summary" was a fixed sentence;
* the summariser saw text only — tool work reached it as nothing;
* the kept tail was ten messages whatever they held, and the request being
  worked on was summarised away as soon as the loop ran past ten;
* ``aclose`` left a running turn calling tools against a closed pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from geny_executor import Pipeline, PipelineState
from geny_executor.core.compaction import UNRECORDED_KEY, run_compaction
from geny_executor.core.stage import Stage
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.strategy import STM_RECORDED_KEY, ProviderDrivenStrategy
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s02_context.artifact.default.compactors import (
    SUMMARY_PREFIX,
    LLMSummaryCompactor,
    SummaryCompactor,
)


def _run(coro):
    return asyncio.run(coro)


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _Client:
    provider = "fake"

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.models: List[str] = []

    async def create_message(self, *, model_config, messages, purpose=""):  # noqa: ANN001
        self.models.append(model_config.model)
        self.prompts.append(messages[0]["content"])
        return _Resp("the user asked for the build; it ran and failed on lint")


def _tool_round(i: int, result: str) -> list:
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"step {i}"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": result}]},
    ]


def _long_turn(rounds: int, result: str = "ok") -> list:
    messages: list = [{"role": "user", "content": "build the project and fix what fails"}]
    for i in range(rounds):
        messages += _tool_round(i, result)
    return messages


# ── what compaction removes is still recorded ────────────────────────


def test_a_turn_that_compacts_still_records_its_own_request():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        state = PipelineState(session_id="s")
        state.messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ] + _long_turn(12)
        # The two replayed messages are already in memory; this turn is not.
        state.metadata[STM_RECORDED_KEY] = 2
        await run_compaction(state, SummaryCompactor(keep_recent=4, keep_recent_ratio=0), trigger="guard")
        await ProviderDrivenStrategy(provider).update(state)
        return [t.content for t in await provider.stm().recent(200)], state

    recorded, state = _run(go())
    assert recorded[0] == "build the project and fix what fails"
    uses = [
        b["input"]["command"]
        for c in recorded
        if isinstance(c, list)
        for b in c
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    assert uses == [f"step {i}" for i in range(12)]  # every call, once, in order
    assert not any(isinstance(c, str) and c.startswith(("[", "earlier")) for c in recorded[1:])
    assert UNRECORDED_KEY not in state.metadata


def test_a_turn_compacted_before_anything_was_recorded_does_not_record_the_summary():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        state = PipelineState(session_id="s")
        state.messages = _long_turn(8)
        await run_compaction(state, SummaryCompactor(keep_recent=4, keep_recent_ratio=0), trigger="guard")
        await ProviderDrivenStrategy(provider).update(state)
        return [t.content for t in await provider.stm().recent(200)]

    recorded = _run(go())
    texts = [c for c in recorded if isinstance(c, str)]
    assert texts == ["build the project and fix what fails"]  # no summary, no acknowledgement


def test_a_pipeline_run_does_not_carry_removed_messages_into_the_next():
    pipeline = Pipeline()
    pipeline.register_stage(InputStage())

    async def go():
        state = PipelineState(session_id="s")
        state.metadata[UNRECORDED_KEY] = [{"role": "user", "content": "stale"}]
        await pipeline.run("hi", state)
        return state

    assert UNRECORDED_KEY not in _run(go()).metadata


# ── the summary ─────────────────────────────────────────────────────


def test_the_summary_is_written_by_the_session_model_from_the_tool_work():
    client = _Client()
    state = PipelineState(session_id="s")
    state.model = "session-model"
    state.llm_client = client
    state.messages = _long_turn(10, result="lint: 3 errors in app.py")
    compactor = LLMSummaryCompactor(keep_recent=4, keep_recent_ratio=0)
    _run(compactor.compact(state))

    assert client.models == ["session-model"]
    prompt = client.prompts[0]
    assert '[called Bash({"command": "step 0"})]' in prompt
    assert "lint: 3 errors in app.py" in prompt
    assert state.messages[0]["content"].startswith(SUMMARY_PREFIX)
    assert "failed on lint" in state.messages[0]["content"]


def test_the_request_being_worked_on_is_kept_word_for_word():
    state = PipelineState(session_id="s")
    state.messages = _long_turn(10)
    request = state.messages[0]
    _run(SummaryCompactor(keep_recent=4, keep_recent_ratio=0).compact(state))

    assert state.messages[2] is request
    roles = [m["role"] for m in state.messages]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # still alternates
    # The tail opens on a call, never on a result whose call is gone.
    assert state.messages[3]["role"] == "assistant"


def test_the_kept_tail_is_sized_by_tokens():
    big = "x" * 40_000  # ~10k tokens a result
    state = PipelineState(session_id="s")
    state.context_window_budget = 100_000
    state.messages = _long_turn(8, result=big)
    _run(SummaryCompactor(keep_recent=10).compact(state))
    # A quarter of 100k holds two of those rounds, not five.
    kept_results = [
        m for m in state.messages if isinstance(m["content"], list) and m["content"][0]["type"] == "tool_result"
    ]
    assert 1 <= len(kept_results) <= 2

    small = PipelineState(session_id="s")
    small.context_window_budget = 100_000
    small.messages = [{"role": "user", "content": "q"}] + [
        m for i in range(60) for m in _tool_round(i, "tiny")
    ] + [{"role": "user", "content": "x" * 400_000}]
    _run(SummaryCompactor(keep_recent=10).compact(small))
    # The huge last message is kept however large; everything small before it
    # that fits the quarter is gone because the last one used it all.
    assert small.messages[-1]["content"].startswith("xxx")


def test_many_small_messages_keep_far_more_than_ten():
    state = PipelineState(session_id="s")
    state.context_window_budget = 4_000
    state.messages = [{"role": "user", "content": "x" * 12_000}] + _long_turn(40)
    _run(SummaryCompactor(keep_recent=10).compact(state))
    assert len(state.messages) > 40


# ── aclose stops the runs ───────────────────────────────────────────


class _Gate(Stage):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False

    @property
    def name(self) -> str:
        return "gate"

    @property
    def order(self) -> int:
        return 2

    async def execute(self, input, state):  # noqa: ANN001
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return input


def test_closing_the_pipeline_stops_a_running_turn():
    gate = _Gate()
    pipeline = Pipeline()
    pipeline.register_stage(InputStage())
    pipeline.register_stage(gate)

    async def go():
        events: List[str] = []

        async def consume():
            async for event in pipeline.run_stream("hello", PipelineState(session_id="s")):
                events.append(event.type)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(gate.entered.wait(), 5)
        await pipeline.aclose()
        # The consumer ends on its own — it was not the one cancelled.
        await asyncio.wait_for(consumer, 5)
        return events

    events: Any = _run(go())
    assert gate.cancelled
    assert "pipeline.cancelled" in events
    assert pipeline._runs_in_flight == 0
