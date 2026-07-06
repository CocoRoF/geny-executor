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
    # ``has_api_key=True`` resolves auth_mode='auto' to the API-key path
    # where ``--bare`` is expected. (Pre-2.2.0 this was decided by
    # sniffing the spawning process's ANTHROPIC_API_KEY — deleted; see
    # claude_code_argv's docstring for the PR #868 history.)
    argv = claude_code_argv(_req(), has_api_key=True)
    assert "--print" in argv
    assert "--output-format" in argv
    idx = argv.index("--output-format")
    assert argv[idx + 1] == "json"
    assert "--bare" in argv


def test_argv_stream_uses_stream_json_io_with_verbose() -> None:
    # ``--verbose`` is required by Claude Code CLI ≥ 2.1.x whenever
    # ``--print`` is combined with ``--output-format=stream-json``;
    # the argv builder emits it automatically alongside the stream-json
    # switch so hosts don't have to thread an opt-in flag.
    argv = claude_code_argv(_req(stream=True))
    assert "--input-format" in argv
    assert "--output-format" in argv
    assert "stream-json" in argv
    assert "--include-partial-messages" in argv
    assert "--verbose" in argv


def test_argv_bare_stripped_on_oauth_path() -> None:
    """When the client holds no API key (``has_api_key=False``, the
    default), ``--bare`` is auto-stripped because the CLI's bare mode
    explicitly disables OAuth ('OAuth and keychain are never read'),
    which crashes every subscription user with 'Not logged in · Please
    run /login'."""
    argv = claude_code_argv(_req(), bare_mode=True, has_api_key=False)
    assert "--bare" not in argv


def test_argv_env_var_no_longer_consulted(monkeypatch) -> None:
    """Regression for the deleted env sniff (PR #868 history): a stray
    ANTHROPIC_API_KEY in the *parent* process env must not flip the
    auth path — the scrubbed child env never necessarily carries it,
    and the client's own credential state is the source of truth."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stray-host-key")
    argv = claude_code_argv(_req(), bare_mode=True, has_api_key=False)
    assert "--bare" not in argv

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    argv = claude_code_argv(_req(), bare_mode=True, has_api_key=True)
    assert "--bare" in argv


@pytest.mark.parametrize(
    "auth_mode, has_api_key, bare_mode, expect_bare",
    [
        # auto: resolves to api_key iff the client holds a key.
        ("auto", True, True, True),
        ("auto", False, True, False),
        # explicit api_key: the host vouches for the key → --bare even
        # when this builder wasn't told about one.
        ("api_key", False, True, True),
        ("api_key", True, True, True),
        # subscription paths never get --bare (it disables OAuth reads).
        ("oauth", True, True, False),
        ("setup_token", True, True, False),
        # bare_mode=False keeps its historical veto on every path.
        ("api_key", True, False, False),
        ("auto", True, False, False),
    ],
)
def test_argv_auth_mode_matrix(auth_mode, has_api_key, bare_mode, expect_bare) -> None:
    argv = claude_code_argv(
        _req(), bare_mode=bare_mode, auth_mode=auth_mode, has_api_key=has_api_key
    )
    assert ("--bare" in argv) is expect_bare


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


def test_argv_request_mcp_config_overrides_kwarg() -> None:
    """``APIRequest.mcp_config`` (per-request) wins over the
    constructor kwarg (per-client static). Phase I uses this to
    inject the per-session Geny tools bridge alongside any
    settings-card-configured MCP servers."""
    per_request = {"mcpServers": {"geny": {"type": "stdio", "command": "py"}}}
    per_client = {"mcpServers": {"legacy": {"command": "x"}}}
    argv = claude_code_argv(_req(mcp_config=per_request), mcp_config=per_client)
    blob = argv[argv.index("--mcp-config") + 1]
    assert json.loads(blob) == per_request  # per-request wins


def test_argv_host_mcp_emits_strict_and_keeps_builtins() -> None:
    """When the host registers MCP servers we emit
    ``--strict-mcp-config`` so the per-session bridge is the only MCP
    surface (no user-level or project-level MCP servers leak in). The
    CLI's *built-in* tool palette (``Bash`` / ``Read`` / ``Write`` /
    ``Edit`` / …) stays available alongside the MCP surface — most
    hosts (e.g. Geny's Sub-Worker) want both: file/shell built-ins for
    real work, MCP for host-delegated tools.

    Earlier executor versions auto-emitted ``--tools ""`` here to
    disable the built-in palette; 2.0.6 dropped that default. Hosts
    that want the old MCP-only behaviour can pass
    ``extra_args=("--tools", "")`` explicitly."""
    cfg = {"mcpServers": {"geny": {"type": "stdio", "command": "py"}}}
    argv = claude_code_argv(_req(mcp_config=cfg))
    assert "--tools" not in argv
    assert "--strict-mcp-config" in argv


def test_argv_host_mcp_with_explicit_allow_tools_emits_allowedtools() -> None:
    """``--allowedTools`` is the permission-pattern allowlist for CLI
    built-ins (e.g. ``Bash(git *)``). Pass it through verbatim when
    the caller supplies one."""
    cfg = {"mcpServers": {"geny": {"type": "stdio", "command": "py"}}}
    argv = claude_code_argv(_req(mcp_config=cfg), allow_tools=["Read"])
    assert "--allowedTools" in argv
    assert "--tools" not in argv


def test_argv_no_mcp_no_tools_flag() -> None:
    """Legacy callers without any MCP config keep today's behaviour:
    CLI built-ins available, no ``--tools ""`` disable, no
    ``--strict-mcp-config``."""
    argv = claude_code_argv(_req())
    assert "--tools" not in argv
    assert "--strict-mcp-config" not in argv
    assert "--mcp-config" not in argv


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
    # No prompt (empty messages) → extra_args remain the trailing tokens.
    argv = claude_code_argv(_req(), extra_args=["--verbose", "--debug", "api"])
    assert argv[-3:] == ["--verbose", "--debug", "api"]


def test_argv_non_stream_prompt_is_dash_separated() -> None:
    """The prompt travels as a ``--``-guarded trailing positional."""
    argv = claude_code_argv(_req(messages=[{"role": "user", "content": "hello there"}]))
    assert argv[-2:] == ["--", "hello there"]


def test_argv_variadic_tool_flag_does_not_swallow_prompt() -> None:
    """Regression: ``--disallowedTools`` (variadic) must not eat the prompt.

    Without the ``--`` separator the CLI parsed the prompt words as extra
    tool names ("permission deny rule '<word>' matches no known tool").
    """
    argv = claude_code_argv(
        _req(messages=[{"role": "user", "content": "add 2 and 2"}]),
        disallow_tools=["Bash", "Read", "Write"],
    )
    assert argv[-2:] == ["--", "add 2 and 2"]
    # the deny list is still present, and nothing after ``--`` is a flag
    assert "--disallowedTools" in argv
    dash = argv.index("--")
    assert all(not tok.startswith("--") for tok in argv[dash + 1 :])


def test_argv_extra_args_precede_prompt_separator() -> None:
    """extra_args (flags) must land before ``--`` so they parse as options."""
    argv = claude_code_argv(
        _req(messages=[{"role": "user", "content": "hi"}]),
        extra_args=["--tools", ""],
    )
    dash = argv.index("--")
    assert argv.index("--tools") < dash
    assert argv[-1] == "hi"


def test_argv_stream_has_no_positional_prompt() -> None:
    """Streaming delivers the prompt via stdin — no ``--``/positional in argv."""
    argv = claude_code_argv(_req(stream=True, messages=[{"role": "user", "content": "hi"}]))
    assert "--" not in argv
    assert "hi" not in argv


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
    envs = [json.loads(ln) for ln in out.strip().split(b"\n")]
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
    envs = [json.loads(ln) for ln in out.strip().split(b"\n")]
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
# stream_event wire format (Claude Code CLI 2.1.x + --include-partial-messages)
# ---------------------------------------------------------------------------


def test_event_stream_event_content_block_delta_text() -> None:
    """The wire format the CLI emits under partial-messages — must
    surface the same canonical text_delta downstream consumers
    already understand."""
    line = {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "우"},
        },
        "session_id": "s",
        "uuid": "u",
    }
    assert stream_json_line_to_canonical_event(line) == {
        "type": "text_delta",
        "text": "우",
    }


def test_event_stream_event_content_block_delta_thinking() -> None:
    line = {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "text": "hm"},
        },
    }
    assert stream_json_line_to_canonical_event(line) == {
        "type": "thinking_delta",
        "text": "hm",
    }


def test_event_stream_event_content_block_start_tool_use() -> None:
    line = {
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "id1",
                "name": "Read",
                "input": {"path": "/x"},
            },
        },
    }
    assert stream_json_line_to_canonical_event(line) == {
        "type": "tool_use",
        "id": "id1",
        "name": "Read",
        "input": {"path": "/x"},
    }


def test_event_stream_event_content_block_stop() -> None:
    line = {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}}
    assert stream_json_line_to_canonical_event(line) == {"type": "content_block_stop"}


def test_event_stream_event_message_metadata_yields_none() -> None:
    """``message_start`` / ``message_delta`` / ``message_stop`` are
    metadata-only; no UI event is emitted."""
    for etype in ("message_start", "message_delta", "message_stop"):
        line = {"type": "stream_event", "event": {"type": etype}}
        assert stream_json_line_to_canonical_event(line) is None


# ---------------------------------------------------------------------------
# StreamJsonAccumulator — stream_event flow
# ---------------------------------------------------------------------------


def test_accumulator_stream_event_yields_token_deltas() -> None:
    """End-to-end: feeding a sequence of stream_event lines that
    mirror the actual CLI output produces a list of text_delta events
    matching the per-token chunks the CLI emitted."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="claude-opus-4-7")
    chunks = ["우", "주는 시간과 공간", "한 전체이다."]
    emitted: list[dict] = []
    for chunk in chunks:
        line = {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk},
            },
        }
        emitted.extend(accum.feed(line))

    assert [e["text"] for e in emitted] == chunks, emitted


def test_accumulator_stream_event_thinking_delta_wire_key() -> None:
    """Real 2.1.149 wire carries thinking chunks under ``thinking``, not
    ``text`` (verified by the golden capture). Reading only ``text``
    silently dropped every thinking token — and the terminal envelope
    masked the loss by re-recording the full block."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    events = accum.feed({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hm, let me think"},
        },
    })
    assert events == [{"type": "thinking_delta", "text": "hm, let me think"}]
    resp = accum.finalize()
    assert resp.thinking_blocks[0].thinking_text == "hm, let me think"


def test_accumulator_rate_limit_event_is_known() -> None:
    """``rate_limit_event`` appears in every recorded golden — it must be
    bookkeeping-only, never inflating the unknown-shape counters."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    events = accum.feed({
        "type": "rate_limit_event",
        "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour"},
    })
    assert events == []
    assert accum.unknown_line_count == 0


# ---------------------------------------------------------------------------
# StreamJsonAccumulator — unknown/malformed wire telemetry (audit §2.2)
# ---------------------------------------------------------------------------


def test_accumulator_counts_unknown_lines_and_samples() -> None:
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    for i in range(5):
        events = accum.feed({"type": "future_thing", "n": i})
        # The tag is still yielded for stream consumers…
        assert events and events[0]["type"] == "cli_unknown"

    # …and now actually counted, typed, and sampled (bounded).
    assert accum.unknown_line_count == 5
    assert accum.first_unknown_type == "future_thing"
    assert len(accum.unknown_samples) == 3  # _SAMPLE_LIMIT bound holds


def test_accumulator_counts_malformed_lines() -> None:
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    assert accum.feed({"__malformed__": "this is not json"}) == []
    assert accum.malformed_line_count == 1
    assert accum.unknown_samples == ["this is not json"]


def test_accumulator_warns_once_per_instance(caplog) -> None:
    """First unknown line per accumulator → one rate-limited warning
    naming the unknown type. Token-rate log floods are the embedding
    401-spam failure mode; one signal per CLI call is the contract."""
    import logging

    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m", cli_version="9.9.9-test")
    with caplog.at_level(logging.WARNING, logger="geny_executor.llm_client.translators._cli"):
        for i in range(4):
            accum.feed({"type": "future_thing", "n": i})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "future_thing" in warnings[0].getMessage()
    assert "9.9.9-test" in warnings[0].getMessage()


def test_accumulator_finalize_merges_telemetry_into_raw() -> None:
    """Counts merge into APIResponse.raw WITHOUT clobbering the terminal
    result envelope's own fields."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    accum.feed({"type": "mystery_line"})
    accum.feed({"__malformed__": "garbage"})
    accum.feed({
        "type": "result", "stop_reason": "end_turn",
        "total_cost_usd": 0.5,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    })
    resp = accum.finalize()
    assert resp.raw["unknown_line_count"] == 1
    assert resp.raw["malformed_line_count"] == 1
    assert resp.raw["first_unknown_type"] == "mystery_line"
    assert resp.raw["unknown_samples"]
    # Envelope fields survive the merge.
    assert resp.raw["total_cost_usd"] == 0.5
    assert resp.usage.cost_usd == pytest.approx(0.5)


def test_accumulator_clean_stream_attaches_no_telemetry_keys() -> None:
    """A clean stream keeps raw byte-for-byte equal to the result
    envelope — no telemetry noise on the happy path."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="m")
    accum.feed({"type": "result", "stop_reason": "end_turn", "usage": {}})
    resp = accum.finalize()
    assert "unknown_line_count" not in resp.raw
    assert accum.unknown_line_count == 0


def test_accumulator_skips_duplicate_text_when_envelope_follows_stream() -> None:
    """The CLI sends BOTH per-token stream_event lines AND a terminal
    ``assistant`` envelope with the full text. The accumulator must
    consume the deltas (so the UI streams) and skip the envelope's
    text (so the final response isn't doubled)."""
    from geny_executor.llm_client.translators._cli import StreamJsonAccumulator

    accum = StreamJsonAccumulator(model="claude-opus-4-7")
    accum.feed(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello world"},
            },
        }
    )
    # Now the terminal envelope (would be a duplicate before the fix).
    accum.feed(
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-7",
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello world"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        }
    )

    resp = accum.finalize()
    assert resp.text == "hello world", (
        f"finalize text duplicated by envelope replay: {resp.text!r}"
    )


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


def test_parse_json_output_drops_tool_use_blocks() -> None:
    """``tool_use`` blocks in the CLI's json output are intentionally
    dropped from the assembled :class:`APIResponse` because the CLI
    already dispatched them internally. Host pipelines should see
    only the final assistant text — see ``finalize``'s docstring for
    the full rationale. The stop_reason is preserved verbatim so
    callers can still distinguish ``end_turn`` from ``tool_use`` for
    telemetry / retry decisions; ``response.tool_calls`` (the actual
    block list, which is what Stage 9 reads to populate
    ``state.pending_tool_calls``) is empty so Stage 10 no-ops."""
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
    assert resp.tool_calls == []
    assert resp.text == "checking..."
    assert resp.stop_reason == "tool_use"


def test_parse_json_output_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_output_to_response(b"not json", model="x")


def test_parse_json_output_real_envelope_result_string() -> None:
    """The REAL ``--output-format json`` envelope (audit §3.4, pinned by
    the golden fixture ``cli-2.1.149-json.json``): assistant text is a
    top-level ``result`` string and cost is top-level ``total_cost_usd``.
    The pre-2.2.0 parser expected an invented ``content[]`` shape and
    returned an empty, cost-less response for these bytes."""
    blob = json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "duration_ms": 2315, "num_turns": 1,
        "result": "우주는 거대한 전체이다.",
        "stop_reason": "end_turn",
        "session_id": "65f5c36b",
        "total_cost_usd": 0.0284,
        "usage": {"input_tokens": 6, "output_tokens": 24},
    }).encode("utf-8")
    resp = parse_json_output_to_response(blob, model="claude-opus-4-7")
    assert resp.text == "우주는 거대한 전체이다."
    assert resp.stop_reason == "end_turn"
    assert resp.usage.cost_usd == pytest.approx(0.0284)
    assert resp.usage.input_tokens == 6
    assert resp.usage.output_tokens == 24
    assert resp.usage.duration_ms == 2315
    assert resp.message_id == "65f5c36b"  # session id fallback
    assert resp.model == "claude-opus-4-7"  # request model fallback


def test_parse_json_output_content_array_still_wins() -> None:
    """Back-compat: when both shapes appear, the ``content[]`` array is
    the canonical record and the top-level ``result`` string must not
    double the text."""
    blob = json.dumps({
        "type": "result",
        "content": [{"type": "text", "text": "canonical"}],
        "result": "canonical",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8")
    resp = parse_json_output_to_response(blob, model="m")
    assert resp.text == "canonical"
    assert len([b for b in resp.content if b.type == "text"]) == 1


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
        for ln in lines:
            yield ln

    resp = await assemble_response_from_stream_json(gen(), model="default")
    assert resp.text == "Hello!"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 4
    assert resp.usage.cost_usd == pytest.approx(0.0001)
    assert resp.usage.duration_ms == 500
    assert resp.model == "claude-sonnet-4-6"
    assert resp.message_id == "s1"


@pytest.mark.asyncio
async def test_assemble_drops_tool_use_blocks() -> None:
    """Tool calls observed in the CLI's stream-json output are
    intentionally dropped from the assembled :class:`APIResponse` —
    the CLI dispatched them internally and host pipelines (e.g.
    Geny's Stage 10) must NOT re-dispatch. See ``finalize``'s
    docstring for the full rationale. The stop_reason is preserved
    so callers can still see the CLI ended in a tool turn."""
    lines = [
        b'{"type": "system", "model": "claude-sonnet-4-6"}\n',
        b'{"type": "assistant", "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}}\n',
        b'{"type": "assistant", "delta": {"type": "input_json_delta", "partial_json": "{\\"pa"}}\n',
        b'{"type": "assistant", "delta": {"type": "input_json_delta", "partial_json": "th\\":\\"/x\\"}"}}\n',
        b'{"type": "content_block_stop"}\n',
        b'{"type": "result", "stop_reason": "tool_use", "usage": {"input_tokens": 8, "output_tokens": 4}}\n',
    ]

    async def gen():
        for ln in lines:
            yield ln

    resp = await assemble_response_from_stream_json(gen(), model="default")
    assert resp.tool_calls == []
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
        for ln in lines:
            yield ln

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
        for ln in lines:
            yield ln

    with pytest.raises(RuntimeError, match="rate limited"):
        await assemble_response_from_stream_json(gen(), model="m")
