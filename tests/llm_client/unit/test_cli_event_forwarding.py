"""CLI agentic-loop event surfacing (2.2.0, audit §3.2 / Tier 1-1).

Two layers under test:

1. :class:`StreamJsonAccumulator` — ``user`` envelopes now surface
   CLI-executed tool results as canonical ``tool_result`` events
   (and stop inflating the unknown-shape counters: pre-2.2.0 the
   ``user`` line type was missing from ``feed``'s dispatch entirely,
   so every tool-using CLI session tripped the
   ``llm_client.unknown_wire_shape`` telemetry and hard-failed under
   ``strict_wire=True``).

2. End-to-end: the fake CLI's ``ok_stream_event_tools`` scenario
   (thinking + CLI-dispatched tool + result + text) through
   ``Pipeline.run_stream`` — the exact flow Geny/GAPT monkey-patched
   Stage 6 to see.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor import Pipeline
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient
from geny_executor.llm_client.translators._cli import StreamJsonAccumulator
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage
from geny_executor.stages.s21_yield import YieldStage

FAKE_CLAUDE = str(Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")


def _client(scenario: str, **kwargs) -> ClaudeCodeCLIClient:
    env_extras = {"FAKE_CLAUDE_SCENARIO": scenario}
    defaults = dict(
        binary_path=FAKE_CLAUDE,
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        bare_mode=True,
        timeout_s=10.0,
        env_extras=env_extras,
    )
    defaults.update(kwargs)
    return ClaudeCodeCLIClient(**defaults)


# ---------------------------------------------------------------------------
# Accumulator: user envelopes
# ---------------------------------------------------------------------------


class TestFeedUserEnvelope:
    def test_tool_result_blocks_become_canonical_events(self):
        accum = StreamJsonAccumulator(model="m")
        events = accum.feed(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "file body",
                            "is_error": False,
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": "denied",
                            "is_error": True,
                        },
                    ],
                },
            }
        )
        assert events == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "file body",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "denied",
                "is_error": True,
            },
        ]

    def test_plain_input_echo_yields_nothing(self):
        accum = StreamJsonAccumulator(model="m")
        assert accum.feed({"type": "user", "message": {"role": "user", "content": "hi"}}) == []

    def test_user_envelope_no_longer_counts_as_unknown(self):
        """The telemetry-inflation regression: a documented wire shape
        must never trip the unknown-shape counters (which feed
        llm_client.unknown_wire_shape and strict_wire failures)."""
        accum = StreamJsonAccumulator(model="m")
        accum.feed({"type": "user", "message": {"role": "user", "content": "hi"}})
        accum.feed(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}],
                },
            }
        )
        assert accum.unknown_line_count == 0
        assert accum.malformed_line_count == 0

    def test_malformed_user_envelope_is_tolerated(self):
        accum = StreamJsonAccumulator(model="m")
        assert accum.feed({"type": "user"}) == []
        assert accum.feed({"type": "user", "message": "weird"}) == []
        assert accum.unknown_line_count == 0


# ---------------------------------------------------------------------------
# Client stream: tool_result flows through create_message_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_stream_yields_full_agentic_event_sequence():
    from geny_executor.core.config import ModelConfig

    client = _client("ok_stream_event_tools")
    chunks = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="claude-sonnet-4-6"),
        messages=[{"role": "user", "content": "read /tmp/x"}],
    ):
        chunks.append(chunk)

    types = [c["type"] for c in chunks]
    assert "thinking_delta" in types
    assert "tool_use" in types
    assert "input_json_delta" in types
    assert "tool_result" in types
    assert "text_delta" in types
    assert types[-1] == "message_complete"

    tool_use = next(c for c in chunks if c["type"] == "tool_use")
    assert tool_use["id"] == "toolu_fake_1"
    assert tool_use["name"] == "Read"
    tool_result = next(c for c in chunks if c["type"] == "tool_result")
    assert tool_result["tool_use_id"] == "toolu_fake_1"
    assert tool_result["is_error"] is False

    # The terminal response must NOT carry unknown-wire telemetry —
    # user envelopes are a known shape now.
    final = chunks[-1]["response"]
    assert "unknown_line_count" not in (final.raw or {})


# ---------------------------------------------------------------------------
# End to end: fake CLI → Stage 6 → Pipeline.run_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_cli_stream_event_scenario_end_to_end_through_run_stream():
    """The monkey-patch killer, proven at the outermost surface: a
    CLI-backed pipeline streams thinking.delta + api.tool_use (+ the
    cli companion + input json + tool_result) to a run_stream consumer
    with zero host-side patching."""
    pipeline = Pipeline()
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage())
    pipeline.register_stage(YieldStage())
    pipeline.attach_runtime(llm_client=_client("ok_stream_event_tools"))

    events = []
    async for event in pipeline.run_stream("read /tmp/x please"):
        events.append(event)

    types = [e.type for e in events]
    assert types[-1] == "pipeline.complete", types

    thinking = [e for e in events if e.type == "thinking.delta"]
    assert [e.data["text"] for e in thinking] == ["Let me look. ", "Found it."]

    tool_uses = [e for e in events if e.type == "api.tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0].data["name"] == "Read"
    assert tool_uses[0].data["source"] == "cli"

    cli_calls = [e for e in events if e.type == "api.cli_tool_call"]
    assert len(cli_calls) == 1
    assert cli_calls[0].data == tool_uses[0].data

    json_deltas = [e.data["delta"] for e in events if e.type == "api.input_json_delta"]
    assert "".join(json_deltas) == '{"file_path": "/tmp/x"}'

    results = [e for e in events if e.type == "api.tool_result"]
    assert len(results) == 1
    assert results[0].data["tool_use_id"] == "toolu_fake_1"
    assert results[0].data["source"] == "cli"

    assert any(e.type == "text.delta" for e in events)
    # Correlation rides along the whole way.
    assert len({e.run_id for e in events}) == 1
