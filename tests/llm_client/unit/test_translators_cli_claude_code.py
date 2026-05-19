"""Tests for the Claude Code translation helpers (Phase B1)."""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client.translators._cli import (
    assemble_response_from_stream_json,
    build_stream_json_stdin,
    claude_code_argv,
    parse_json_output_to_response,
    stream_json_line_to_canonical_event,
    thinking_to_effort,
)
from geny_executor.llm_client.types import APIRequest


# ---------------------------------------------------------------------------
# thinking_to_effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking, expected",
    [
        (None, None),
        ({}, None),
        ({"type": "disabled"}, None),
        ({"type": "enabled", "budget_tokens": 0}, "low"),
        ({"type": "enabled", "budget_tokens": 5_000}, "low"),
        ({"type": "enabled", "budget_tokens": 10_000}, "medium"),
        ({"type": "enabled", "budget_tokens": 20_000}, "high"),
        ({"type": "enabled", "budget_tokens": 50_000}, "xhigh"),
        ({"type": "enabled", "budget_tokens": 100_000}, "max"),
    ],
)
def test_thinking_to_effort(thinking, expected) -> None:
    assert thinking_to_effort(thinking) == expected


# ---------------------------------------------------------------------------
# claude_code_argv
# ---------------------------------------------------------------------------


def _req(**kwargs) -> APIRequest:
    base = dict(model="sonnet", messages=[], system="", stream=False)
    base.update(kwargs)
    return APIRequest(**base)


def test_argv_non_stream_uses_json_output() -> None:
    argv = claude_code_argv(_req())
    assert "--print" in argv
    assert "--output-format" in argv
    idx = argv.index("--output-format")
    assert argv[idx + 1] == "json"
    assert "--bare" in argv


def test_argv_stream_uses_stream_json_io() -> None:
    argv = claude_code_argv(_req(stream=True))
    assert "--input-format" in argv
    assert "--output-format" in argv
    assert "stream-json" in argv
    assert "--include-partial-messages" in argv


def test_argv_includes_model_and_system_prompt() -> None:
    argv = claude_code_argv(_req(model="opus", system="be brief."))
    assert "--model" in argv and "opus" in argv
    assert "--system-prompt" in argv and "be brief." in argv


def test_argv_system_block_list_flattens_to_text() -> None:
    sys_blocks = [
        {"type": "text", "text": "policy A"},
        {"type": "text", "text": "policy B"},
        {"type": "image"},  # ignored
    ]
    argv = claude_code_argv(_req(system=sys_blocks))
    sp = argv[argv.index("--system-prompt") + 1]
    assert sp == "policy A\npolicy B"


def test_argv_thinking_to_effort() -> None:
    argv = claude_code_argv(
        _req(thinking={"type": "enabled", "budget_tokens": 25_000})
    )
    assert "--effort" in argv and "high" in argv


def test_argv_allow_and_deny_tools() -> None:
    argv = claude_code_argv(_req(), allow_tools=["Read", "Bash"], disallow_tools=["Write"])
    assert "--allowedTools" in argv
    assert "Read Bash" in argv
    assert "--disallowedTools" in argv
    assert "Write" in argv


def test_argv_permission_mode_default_omitted() -> None:
    argv = claude_code_argv(_req(), permission_mode="default")
    assert "--permission-mode" not in argv


def test_argv_permission_mode_non_default_emitted() -> None:
    argv = claude_code_argv(_req(), permission_mode="acceptEdits")
    assert "--permission-mode" in argv and "acceptEdits" in argv


def test_argv_max_budget_usd() -> None:
    argv = claude_code_argv(_req(), max_budget_usd=2.5)
    assert "--max-budget-usd" in argv and "2.5" in argv


def test_argv_settings_path() -> None:
    argv = claude_code_argv(_req(), settings_path="/tmp/settings.json")
    assert "--settings" in argv and "/tmp/settings.json" in argv


def test_argv_mcp_config_dict_serialized_as_json() -> None:
    cfg = {"mcpServers": {"x": {"command": "y"}}}
    argv = claude_code_argv(_req(), mcp_config=cfg)
    blob = argv[argv.index("--mcp-config") + 1]
    assert json.loads(blob) == cfg


def test_argv_mcp_config_path_passed_through() -> None:
    argv = claude_code_argv(_req(), mcp_config="/tmp/mcp.json")
    blob = argv[argv.index("--mcp-config") + 1]
    assert blob == "/tmp/mcp.json"


def test_argv_response_format_json_schema_emits_flag() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    argv = claude_code_argv(
        _req(response_format={"type": "json_schema", "json_schema": schema})
    )
    blob = argv[argv.index("--json-schema") + 1]
    assert json.loads(blob) == schema


def test_argv_response_format_other_type_ignored() -> None:
    argv = claude_code_argv(_req(response_format={"type": "json_object"}))
    assert "--json-schema" not in argv


def test_argv_session_id_without_resume() -> None:
    argv = claude_code_argv(_req(session_hint={"session_id": "abc"}))
    assert "--session-id" in argv and "abc" in argv
    assert "--resume" not in argv


def test_argv_resume_session_id() -> None:
    argv = claude_code_argv(_req(session_hint={"session_id": "abc", "resume": True}))
    assert "--resume" in argv and "abc" in argv
    assert "--session-id" not in argv


def test_argv_extra_args_appended_verbatim() -> None:
    argv = claude_code_argv(_req(), extra_args=["--verbose", "--debug", "api"])
    assert argv[-3:] == ["--verbose", "--debug", "api"]


def test_argv_dropped_fields_not_emitted() -> None:
    """Fields the CLI doesn't accept must not leak in."""
    argv = claude_code_argv(_req(temperature=0.7, top_p=0.9, top_k=10, stop_sequences=["x"]))
    for flag in ("--temperature", "--top-p", "--top-k", "--stop-sequence"):
        assert flag not in argv


# ---------------------------------------------------------------------------
# build_stream_json_stdin
# ---------------------------------------------------------------------------


def test_stdin_envelope_one_user_message() -> None:
    out = build_stream_json_stdin([{"role": "user", "content": "hi"}])
    assert out.endswith(b"\n")
    envs = [json.loads(l) for l in out.strip().split(b"\n")]
    assert envs == [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
    ]


def test_stdin_envelope_multi_turn_always_user_role() -> None:
    """Regression: every envelope's ``message.role`` MUST be ``"user"``.

    Claude Code CLI 2.x rejects ``type:user`` envelopes that carry an
    embedded ``message.role: assistant`` with::

        Error: Expected message role 'user', got 'assistant'

    The pre-fix builder forwarded canonical roles through and broke
    every multi-turn iteration of an env that pinned ``claude_code_cli``
    as the Stage 6 provider.
    """
    out = build_stream_json_stdin([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
    ])
    envs = [json.loads(l) for l in out.strip().split(b"\n")]
    # ONE synthetic envelope — multi-turn collapses to a single user
    # message; the CLI reconstructs the conversation from its content.
    assert len(envs) == 1
    assert envs[0]["type"] == "user"
    assert envs[0]["message"]["role"] == "user"


def test_stdin_envelope_multi_turn_preserves_history_in_content() -> None:
    """The collapsed envelope must carry enough fidelity that the LLM
    can reconstruct the prior conversation: text turns, tool calls
    (name + input), and tool results all show up in the flattened
    content under markdown headers."""
    out = build_stream_json_stdin([
        {"role": "user", "content": "find the README"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Read",
                    "input": {"path": "/repo/README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "# Hello"},
            ],
        },
        {"role": "user", "content": "summarize it"},
    ])
    env = json.loads(out.strip())
    text = env["message"]["content"]
    assert "## Conversation so far" in text
    assert "find the README" in text
    assert "[Tool call: Read({" in text
    assert "/repo/README.md" in text
    assert "[Tool result] # Hello" in text
    # The final user turn ("summarize it") is the "current input" and
    # appears under "## Current input" without the per-turn header.
    assert "## Current input" in text
    assert text.rstrip().endswith("summarize it")


def test_stdin_envelope_drops_thinking_and_handles_tool_errors() -> None:
    """Thinking blocks from a prior provider don't replay on the CLI
    — drop them. ``is_error: True`` tool_results render under a
    "Tool error" tag so the LLM sees the failure semantics."""
    out = build_stream_json_stdin([
        {"role": "user", "content": "do X"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret reasoning"},
                {"type": "text", "text": "trying X..."},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "command failed"},
            ],
        },
    ])
    env = json.loads(out.strip())
    text = env["message"]["content"]
    assert "secret reasoning" not in text  # thinking dropped
    assert "trying X..." in text
    assert "[Tool error] command failed" in text


def test_stdin_empty_messages_returns_empty_bytes() -> None:
    assert build_stream_json_stdin([]) == b""


# ---------------------------------------------------------------------------
# stream_json_line_to_canonical_event
# ---------------------------------------------------------------------------


def test_event_system_returns_none() -> None:
    assert stream_json_line_to_canonical_event({"type": "system"}) is None


def test_event_user_returns_none() -> None:
    assert stream_json_line_to_canonical_event({"type": "user"}) is None


def test_event_text_delta() -> None:
    out = stream_json_line_to_canonical_event(
        {"type": "assistant", "delta": {"type": "text_delta", "text": "ab"}}
    )
    assert out == {"type": "text_delta", "text": "ab"}


def test_event_thinking_delta() -> None:
    out = stream_json_line_to_canonical_event(
        {"type": "assistant", "delta": {"type": "thinking_delta", "text": "hm"}}
    )
    assert out == {"type": "thinking_delta", "text": "hm"}


def test_event_input_json_delta() -> None:
    out = stream_json_line_to_canonical_event(
        {"type": "assistant", "delta": {"type": "input_json_delta", "partial_json": "{\"a"}}
    )
    assert out == {"type": "input_json_delta", "delta": "{\"a"}


def test_event_tool_use_block_start() -> None:
    out = stream_json_line_to_canonical_event(
        {
            "type": "assistant",
            "content_block": {"type": "tool_use", "id": "id1", "name": "Read", "input": {"path": "/x"}},
        }
    )
    assert out == {"type": "tool_use", "id": "id1", "name": "Read", "input": {"path": "/x"}}


def test_event_message_stop_completes() -> None:
    out = stream_json_line_to_canonical_event({"type": "message_stop"})
    assert out == {"type": "message_complete"}


def test_event_error_propagated() -> None:
    raw = {"type": "error", "code": "oops"}
    out = stream_json_line_to_canonical_event(raw)
    assert out == {"type": "error", "raw": raw}


def test_event_malformed() -> None:
    out = stream_json_line_to_canonical_event({"__malformed__": "junk"})
    assert out == {"type": "cli_malformed", "raw": "junk"}


def test_event_unknown_type() -> None:
    raw = {"type": "future_thing", "x": 1}
    out = stream_json_line_to_canonical_event(raw)
    assert out["type"] == "cli_unknown"
    assert out["raw"] is raw


# ---------------------------------------------------------------------------
# parse_json_output_to_response
# ---------------------------------------------------------------------------


def test_parse_json_output_text_only() -> None:
    blob = json.dumps({
        "type": "result",
        "message_id": "msg_1",
        "stop_reason": "end_turn",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 12, "output_tokens": 3, "cost_usd": 0.0006},
        "duration_ms": 800,
    }).encode("utf-8")
    resp = parse_json_output_to_response(blob, model="default")
    assert resp.text == "hello"
    assert resp.message_id == "msg_1"
    assert resp.stop_reason == "end_turn"
    assert resp.model == "claude-sonnet-4-6"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 3
    assert resp.usage.cost_usd == pytest.approx(0.0006)
    assert resp.usage.duration_ms == 800


def test_parse_json_output_tool_use_round_trip() -> None:
    blob = json.dumps({
        "type": "result",
        "content": [
            {"type": "text", "text": "checking..."},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }).encode("utf-8")
    resp = parse_json_output_to_response(blob, model="m")
    assert resp.has_tool_calls is True
    tools = resp.tool_calls
    assert len(tools) == 1
    assert tools[0].tool_name == "Read"
    assert tools[0].tool_input == {"path": "/x"}


def test_parse_json_output_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_output_to_response(b"not json", model="x")


# ---------------------------------------------------------------------------
# assemble_response_from_stream_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_simple_text_stream() -> None:
    lines = [
        b'{"type": "system", "session_id": "s1", "model": "claude-sonnet-4-6"}\n',
        b'{"type": "assistant", "delta": {"type": "text_delta", "text": "Hel"}}\n',
        b'{"type": "assistant", "delta": {"type": "text_delta", "text": "lo!"}}\n',
        b'{"type": "message_stop"}\n',
        b'{"type": "result", "stop_reason": "end_turn", "usage": {"input_tokens": 4, "output_tokens": 2, "cost_usd": 0.0001}, "duration_ms": 500}\n',
    ]

    async def gen():
        for l in lines:
            yield l

    resp = await assemble_response_from_stream_json(gen(), model="default")
    assert resp.text == "Hello!"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 4
    assert resp.usage.cost_usd == pytest.approx(0.0001)
    assert resp.usage.duration_ms == 500
    assert resp.model == "claude-sonnet-4-6"
    assert resp.message_id == "s1"


@pytest.mark.asyncio
async def test_assemble_tool_use_with_partial_json() -> None:
    lines = [
        b'{"type": "system", "model": "claude-sonnet-4-6"}\n',
        b'{"type": "assistant", "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}}\n',
        b'{"type": "assistant", "delta": {"type": "input_json_delta", "partial_json": "{\\"pa"}}\n',
        b'{"type": "assistant", "delta": {"type": "input_json_delta", "partial_json": "th\\":\\"/x\\"}"}}\n',
        b'{"type": "content_block_stop"}\n',
        b'{"type": "result", "stop_reason": "tool_use", "usage": {"input_tokens": 8, "output_tokens": 4}}\n',
    ]

    async def gen():
        for l in lines:
            yield l

    resp = await assemble_response_from_stream_json(gen(), model="default")
    assert resp.has_tool_calls is True
    tu = resp.tool_calls[0]
    assert tu.tool_name == "Read"
    assert tu.tool_input == {"path": "/x"}
    assert resp.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_assemble_thinking_blocks_collected() -> None:
    lines = [
        b'{"type": "system"}\n',
        b'{"type": "assistant", "delta": {"type": "thinking_delta", "text": "let me think... "}}\n',
        b'{"type": "assistant", "delta": {"type": "thinking_delta", "text": "ok."}}\n',
        b'{"type": "assistant", "delta": {"type": "text_delta", "text": "answer"}}\n',
        b'{"type": "message_stop"}\n',
        b'{"type": "result", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}\n',
    ]

    async def gen():
        for l in lines:
            yield l

    resp = await assemble_response_from_stream_json(gen(), model="m")
    assert any(b.type == "thinking" for b in resp.content)
    assert resp.thinking_blocks[0].thinking_text == "let me think... ok."
    assert resp.text == "answer"


@pytest.mark.asyncio
async def test_assemble_raises_on_error_envelope() -> None:
    lines = [
        b'{"type": "system"}\n',
        b'{"type": "error", "message": "rate limited"}\n',
    ]

    async def gen():
        for l in lines:
            yield l

    with pytest.raises(RuntimeError, match="rate limited"):
        await assemble_response_from_stream_json(gen(), model="m")
