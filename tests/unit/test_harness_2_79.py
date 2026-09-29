"""2.79.0 — quality fixes from the 2026-09-29 harness audit.

* the date line said UTC with no weekday;
* STM reads re-read the whole transcript after every append, on the event
  loop; LTM search and large tool results did their file I/O there too;
* Stage 5's cache markers were recorded into memory and replayed;
* a failed STM write still moved the watermark past the message;
* the hybrid and progressive context strategies cut by message count, into
  the middle of tool loops;
* an image-only request searched memory with "";
* on the Claude Code path the request's image was gone after the first
  tool step, and a PDF went into the prompt as base64 text.
"""

from __future__ import annotations

import asyncio
import base64
import re
from typing import Any, List

from geny_executor.core.compaction import UNRECORDED_KEY, mark_recorded_upto, unrecorded_messages
from geny_executor.core.state import PipelineState
from geny_executor.llm_client import text_tool_protocol as tp
from geny_executor.memory.strategy import STM_RECORDED_KEY, ProviderDrivenStrategy
from geny_executor.stages.s02_context.artifact.default.stage import _retrieval_query
from geny_executor.stages.s02_context.artifact.default.strategies import (
    HybridStrategy,
    ProgressiveDisclosureStrategy,
)
from geny_executor.stages.s03_system.artifact.default.builders import DateTimeBlock
from geny_executor.stages.s18_memory._dehydrate import dehydrate_message

_PNG = base64.b64encode(b"\x89PNG fake").decode()
_PDF = base64.b64encode(b"%PDF-1.4 " + b"x" * 5000).decode()


def test_the_date_line_is_local_with_a_weekday():
    line = DateTimeBlock(tz="Asia/Seoul").render(PipelineState())
    assert re.match(
        r"Current date: \d{4}-\d{2}-\d{2} \((Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\), "
        r"\d{2}:\d{2} Asia/Seoul \(UTC\+09:00\)$",
        line,
    ), line
    assert "UTC" in DateTimeBlock(tz="Not/AZone").render(PipelineState())


def test_cache_markers_are_not_recorded():
    msg = {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
    stored = dehydrate_message(msg)
    assert "cache_control" not in stored["content"][0]
    assert "cache_control" in msg["content"][0]  # the live message keeps it


class _FlakyProvider:
    def __init__(self, fail_on: str):
        self.fail_on = fail_on
        self.recorded: List[Any] = []

    async def record_turn(self, turn):
        if turn.content == self.fail_on:
            self.fail_on = ""  # fails once
            raise OSError("disk full")
        self.recorded.append(turn.content)


def test_a_failed_write_is_retried_not_skipped():
    provider = _FlakyProvider(fail_on="b")
    state = PipelineState(session_id="s")
    state.messages = [{"role": "user", "content": c} for c in ("a", "b", "c")]
    strategy = ProviderDrivenStrategy(provider)
    asyncio.run(strategy.update(state))
    assert provider.recorded == ["a"]
    assert state.metadata[STM_RECORDED_KEY] == 1
    asyncio.run(strategy.update(state))
    assert provider.recorded == ["a", "b", "c"]


def test_partial_marking_spans_the_held_back_messages():
    state = PipelineState(session_id="s")
    state.metadata[UNRECORDED_KEY] = [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
    state.metadata[STM_RECORDED_KEY] = 0
    state.messages = [{"role": "user", "content": "z"}]
    mark_recorded_upto(state, 1)
    assert [m["content"] for m in unrecorded_messages(state)] == ["y", "z"]
    mark_recorded_upto(state, 2)
    assert unrecorded_messages(state) == []


def _loop_history(turns: int) -> list:
    msgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": f"request {t}"})
        for i in range(3):
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"{t}-{i}", "name": "X", "input": {}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"{t}-{i}", "content": "ok"}]})
        msgs.append({"role": "assistant", "content": f"answer {t}"})
    return msgs


def _paired(messages) -> bool:
    uses = {b["id"] for m in messages if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"}
    results = {b["tool_use_id"] for m in messages if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"}
    return results <= uses


def test_hybrid_keeps_whole_turns():
    state = PipelineState(session_id="s")
    state.messages = _loop_history(6)
    asyncio.run(HybridStrategy(max_recent_turns=2).build_context(state))
    assert state.messages[0]["content"] == "request 4"
    assert _paired(state.messages)


def test_progressive_keeps_the_task_and_says_what_it_left_out():
    state = PipelineState(session_id="s")
    state.messages = _loop_history(6)
    asyncio.run(ProgressiveDisclosureStrategy(summary_threshold=2).build_context(state))
    assert state.messages[0]["content"] == "request 0"
    assert state.messages[1]["role"] == "assistant" and "not shown" in state.messages[1]["content"]
    assert state.messages[2]["content"] == "request 4"
    assert _paired(state.messages)


def test_an_image_only_request_still_searches_memory():
    messages = [
        {"role": "user", "content": "the login page keeps failing on Safari"},
        {"role": "assistant", "content": "Send me a screenshot?"},
        {"role": "user", "content": [{"type": "image", "name": "safari.png", "source": {"type": "base64", "media_type": "image/png", "data": _PNG}}]},
    ]
    assert _retrieval_query(messages) == "the login page keeps failing on Safari safari.png"
    assert _retrieval_query([{"role": "user", "content": "plain"}]) == "plain"


def test_the_requests_media_stays_attached_through_the_tool_loop():
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what does this say?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG}},
                {"type": "document", "title": "spec.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF}},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    blocks = tp.render_transcript(history)
    kinds = [b["type"] for b in blocks]
    assert kinds.count("image") == 1 and kinds.count("document") == 1
    text = blocks[-1]["text"]
    assert _PDF[:40] not in text  # no base64 in the prompt text
    assert "[document: spec.pdf] (attached)" in text
