#!/usr/bin/env python3
"""Tiny fake ``claude`` CLI used by Claude Code client tests.

Scenarios driven by the ``FAKE_CLAUDE_SCENARIO`` env var:

  - ``ok_text``        — emit a single text result. JSON mode prints one
                         ``{"type": "result", ...}`` blob. Streaming mode
                         emits a 3-line stream-json sequence.
  - ``ok_tool_use``    — single tool_use response (json mode only).
  - ``ok_thinking``    — emits a thinking block + text (streaming).
  - ``ok_stream_event``— the REAL 2.1.x wire shape under
                         ``--include-partial-messages``: ``stream_event``
                         lines (message_start / content_block_start /
                         N content_block_delta / content_block_stop /
                         message_delta / message_stop) PLUS the terminal
                         ``assistant`` envelope carrying the full text,
                         then ``result``. Mirrors the recorded golden
                         transcripts in ``tests/llm_client/golden/``.
  - ``ok_stream_event_tools`` — ``ok_stream_event`` plus the agentic
                         loop: thinking deltas, a CLI-executed tool_use
                         (content_block_start + input_json_delta) and
                         its ``user``-envelope tool_result, then text.
  - ``ok_result_envelope`` — the REAL ``--output-format json`` envelope
                         (top-level ``result`` string + ``total_cost_usd``,
                         NO ``content[]`` array; audit §3.4).
  - ``stream_unknown_lines`` — like ``ok_stream_event`` but interleaves a
                         future/unknown line type and one malformed line,
                         for wire-drift telemetry tests.
  - ``stream_then_hang``— preamble + ONE text delta, then sleep forever.
                         Writes its pid to ``$FAKE_CLAUDE_PID_FILE`` so
                         cancellation tests can poll whether the child
                         was reaped after the consumer disconnected.
  - ``auth_fail``      — exit 1 with an auth-related stderr message.
  - ``permission_fail``— exit 1 with a permission-related stderr.
  - ``crash``          — exit 2 with generic stderr.
  - ``hang``           — sleep forever (callers must enforce timeout).
  - ``echo_argv``      — print argv as JSON; useful for shape assertions.

``--version`` is answered before scenario dispatch (prints a fake
version string and exits 0) because the real client performs a version
handshake on first use — without this, the handshake would capture a
scenario blob as the "version".

The script never reads its stdin in a blocking way unless asked to,
and only uses the standard library so it can run as the child binary
of a CLIProcessRunner without extra deps.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List


def _emit_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def _emit_line(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _is_streaming(argv: List[str]) -> bool:
    return "--output-format" in argv and (
        argv[argv.index("--output-format") + 1] == "stream-json"
    )


def _ok_text(argv: List[str]) -> int:
    text = os.environ.get("FAKE_CLAUDE_TEXT", "Hello from fake claude.")
    if _is_streaming(argv):
        _emit_line({"type": "system", "session_id": "fake-session-1", "model": "claude-sonnet-4-6"})
        for ch in text:
            _emit_line({"type": "assistant", "delta": {"type": "text_delta", "text": ch}})
        _emit_line({"type": "message_stop"})
        _emit_line({
            "type": "result",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": len(text), "cost_usd": 0.0002},
            "duration_ms": 50,
        })
    else:
        _emit_json({
            "type": "result",
            "message_id": "msg_fake_1",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 5, "output_tokens": len(text), "cost_usd": 0.0002},
            "duration_ms": 50,
        })
    return 0


def _ok_tool_use(argv: List[str]) -> int:
    _emit_json({
        "type": "result",
        "message_id": "msg_fake_tool",
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Let me read that file."},
            {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "/tmp/x"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 6},
        "duration_ms": 100,
    })
    return 0


def _ok_thinking(argv: List[str]) -> int:
    _emit_line({"type": "system", "session_id": "fake-think", "model": "claude-opus-4-7"})
    _emit_line({"type": "assistant", "delta": {"type": "thinking_delta", "text": "Thinking step 1. "}})
    _emit_line({"type": "assistant", "delta": {"type": "thinking_delta", "text": "Step 2."}})
    _emit_line({"type": "assistant", "delta": {"type": "text_delta", "text": "Answer."}})
    _emit_line({"type": "message_stop"})
    _emit_line({
        "type": "result",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 3},
    })
    return 0


def _auth_fail(argv: List[str]) -> int:
    sys.stderr.write("Error: not authenticated. Run `claude auth login`.\n")
    return 1


def _ok_stream_event(argv: List[str]) -> int:
    """Real Claude Code 2.1.x wire shape under ``--include-partial-messages``.

    Structurally mirrors the recorded transcripts under
    ``tests/llm_client/golden/`` (CLI 2.1.149 / 2.1.162): per-token
    ``stream_event`` lines AND a terminal ``assistant`` envelope carrying
    the same text in full, followed by the ``result`` line whose
    assistant text lives in the top-level ``result`` string. The parser
    must stream the deltas, skip the envelope duplicate, and price from
    ``total_cost_usd``.
    """
    text = os.environ.get("FAKE_CLAUDE_TEXT", "hello stream")
    sid = "fake-se-1"
    _emit_line({
        "type": "system", "subtype": "init", "session_id": sid,
        "model": "claude-sonnet-4-6", "claude_code_version": "2.1.149",
        "apiKeySource": "none", "uuid": "u-init",
    })
    _emit_line({
        "type": "rate_limit_event",
        "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour"},
        "uuid": "u-rl", "session_id": sid,
    })
    _emit_line({
        "type": "stream_event",
        "event": {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-4-6", "id": "msg_se_1",
                "type": "message", "role": "assistant", "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 6, "output_tokens": 2},
            },
        },
        "session_id": sid, "uuid": "u-ms", "ttft_ms": 100,
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_start", "index": 0,
                  "content_block": {"type": "text", "text": ""}},
        "session_id": sid, "uuid": "u-cbs",
    })
    # Chunked deltas — multiple chunks so consumers can assert real streaming.
    step = max(1, len(text) // 3)
    for i in range(0, len(text), step):
        _emit_line({
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 0,
                      "delta": {"type": "text_delta", "text": text[i:i + step]}},
            "session_id": sid, "uuid": f"u-d{i}",
        })
    _emit_line({
        "type": "assistant",
        "message": {
            "model": "claude-sonnet-4-6", "id": "msg_se_1", "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
            "usage": {"input_tokens": 6, "output_tokens": 2},
        },
        "session_id": sid, "uuid": "u-env",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_stop", "index": 0},
        "session_id": sid, "uuid": "u-cbe",
    })
    _emit_line({
        "type": "stream_event",
        "event": {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": 6, "output_tokens": len(text)},
        },
        "session_id": sid, "uuid": "u-md",
    })
    _emit_line({
        "type": "stream_event", "event": {"type": "message_stop"},
        "session_id": sid, "uuid": "u-mstop",
    })
    _emit_line({
        "type": "result", "subtype": "success", "is_error": False,
        "duration_ms": 250, "num_turns": 1,
        "result": text, "stop_reason": "end_turn", "session_id": sid,
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 6, "output_tokens": len(text)},
        "uuid": "u-res",
    })
    return 0


def _stream_unknown_lines(argv: List[str]) -> int:
    """``ok_stream_event`` plus wire drift: one future line type and one
    malformed line. Exercises the §2.2 telemetry channel end to end."""
    text = os.environ.get("FAKE_CLAUDE_TEXT", "drift")
    sid = "fake-unk-1"
    _emit_line({"type": "system", "subtype": "init", "session_id": sid,
                "model": "claude-sonnet-4-6"})
    _emit_line({"type": "telepathy_event", "payload": {"x": 1}, "uuid": "u-unk"})
    sys.stdout.write("this is not json\n")
    sys.stdout.flush()
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "text_delta", "text": text}},
        "session_id": sid, "uuid": "u-d0",
    })
    _emit_line({
        "type": "result", "subtype": "success", "is_error": False,
        "result": text, "stop_reason": "end_turn", "session_id": sid,
        "total_cost_usd": 0.001,
        "usage": {"input_tokens": 2, "output_tokens": 2},
    })
    return 0


def _ok_stream_event_tools(argv: List[str]) -> int:
    """``ok_stream_event`` extended with the agentic-loop wire shapes.

    Mirrors what the real CLI emits for a turn that thinks, runs one
    tool internally, and answers: ``stream_event`` lines carrying a
    thinking delta, a ``content_block_start`` for a ``tool_use`` block
    plus its ``input_json_delta`` fragments, the ``user`` envelope
    echoing the CLI-executed tool's ``tool_result``, then the text
    answer and the terminal ``result``. Drives the 2.2.0 full
    chunk-forwarding tests (thinking.delta / api.tool_use /
    api.cli_tool_call / api.input_json_delta / api.tool_result end to
    end through Pipeline.run_stream).
    """
    text = os.environ.get("FAKE_CLAUDE_TEXT", "tools done")
    sid = "fake-set-1"
    _emit_line({
        "type": "system", "subtype": "init", "session_id": sid,
        "model": "claude-sonnet-4-6", "claude_code_version": "2.1.149",
    })
    _emit_line({
        "type": "stream_event",
        "event": {
            "type": "message_start",
            "message": {
                "model": "claude-sonnet-4-6", "id": "msg_set_1",
                "type": "message", "role": "assistant", "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
        },
        "session_id": sid, "uuid": "u-ms",
    })
    # Thinking block.
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta", "thinking": "Let me look. "}},
        "session_id": sid, "uuid": "u-t0",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta", "thinking": "Found it."}},
        "session_id": sid, "uuid": "u-t1",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_stop", "index": 0},
        "session_id": sid, "uuid": "u-ts",
    })
    # Tool the CLI runs itself: start + streamed input json.
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_start", "index": 1,
                  "content_block": {"type": "tool_use", "id": "toolu_fake_1",
                                    "name": "Read", "input": {}}},
        "session_id": sid, "uuid": "u-tu",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 1,
                  "delta": {"type": "input_json_delta",
                            "partial_json": "{\"file_path\": \"/tmp"}},
        "session_id": sid, "uuid": "u-ij0",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 1,
                  "delta": {"type": "input_json_delta",
                            "partial_json": "/x\"}"}},
        "session_id": sid, "uuid": "u-ij1",
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_stop", "index": 1},
        "session_id": sid, "uuid": "u-tue",
    })
    # The CLI dispatched the tool internally and echoes the result as a
    # ``user`` envelope (the wire fact StreamJsonAccumulator.finalize
    # has documented since Phase I).
    _emit_line({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{
                "type": "tool_result", "tool_use_id": "toolu_fake_1",
                "content": "file contents here", "is_error": False,
            }],
        },
        "session_id": sid, "uuid": "u-tr",
    })
    # Final text answer.
    step = max(1, len(text) // 2)
    for i in range(0, len(text), step):
        _emit_line({
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 2,
                      "delta": {"type": "text_delta", "text": text[i:i + step]}},
            "session_id": sid, "uuid": f"u-d{i}",
        })
    _emit_line({
        "type": "stream_event", "event": {"type": "message_stop"},
        "session_id": sid, "uuid": "u-mstop",
    })
    _emit_line({
        "type": "result", "subtype": "success", "is_error": False,
        "duration_ms": 400, "num_turns": 2,
        "result": text, "stop_reason": "end_turn", "session_id": sid,
        "total_cost_usd": 0.02,
        "usage": {"input_tokens": 9, "output_tokens": len(text)},
        "uuid": "u-res",
    })
    return 0


def _ok_result_envelope(argv: List[str]) -> int:
    """Real ``--output-format json`` envelope (recorded shape, audit §3.4):
    assistant text in the top-level ``result`` string, cost in
    ``total_cost_usd`` — no ``content[]`` array, no ``usage.cost_usd``."""
    text = os.environ.get("FAKE_CLAUDE_TEXT", "Hi there, friend!")
    _emit_json({
        "type": "result", "subtype": "success", "is_error": False,
        "duration_ms": 4900, "duration_api_ms": 5997, "num_turns": 1,
        "result": text, "stop_reason": "end_turn",
        "session_id": "9102e3cd-1f77-42af-bb56-7bf9ac06365c",
        "total_cost_usd": 0.15079925,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 24005,
                  "cache_read_input_tokens": 0, "output_tokens": 10},
        "permission_denials": [], "uuid": "02e5af70",
    })
    return 0


def _ok_message_form(argv: List[str]) -> int:
    """Real Claude Code 2.x stream-json shape (no ``--include-partial-messages``).

    The CLI puts the entire assistant message inside one ``assistant``
    envelope's ``message.content[]`` instead of streaming deltas. The
    parser must accumulate text from this shape just as well as from
    the delta variant.
    """
    text = os.environ.get("FAKE_CLAUDE_TEXT", "hello world")
    _emit_line({
        "type": "system", "subtype": "init",
        "session_id": "fake-msg-1", "model": "claude-sonnet-4-6",
    })
    _emit_line({
        "type": "assistant",
        "message": {
            "id": "msg_fake_form_2",
            "role": "assistant",
            "type": "message",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": len(text)},
            "content": [{"type": "text", "text": text}],
        },
    })
    _emit_line({
        "type": "result", "subtype": "success", "is_error": False,
        "stop_reason": "end_turn",
        "total_cost_usd": 0.0001,
        "duration_ms": 250,
        "usage": {"input_tokens": 7, "output_tokens": len(text)},
    })
    return 0


def _message_form_auth_failed(argv: List[str]) -> int:
    """Reproduce the "Not logged in" path the user hit on prod.

    Claude Code emits the synthetic placeholder text inside a normal
    ``assistant.message.content[]`` envelope BUT annotates the outer
    line with ``error=authentication_failed``. The parser must raise
    APIError(CLI_AUTH_FAILED) instead of returning the placeholder
    text as the assistant's reply.
    """
    _emit_line({"type": "system", "subtype": "init", "session_id": "fake-na-1", "model": "claude-opus-4-7"})
    _emit_line({
        "type": "assistant",
        "error": "authentication_failed",
        "message": {
            "id": "msg_na",
            "role": "assistant",
            "type": "message",
            "stop_reason": "stop_sequence",
            "content": [{"type": "text", "text": "Not logged in · Please run /login"}],
        },
    })
    _emit_line({
        "type": "result", "subtype": "success", "is_error": True,
        "result": "Not logged in · Please run /login",
    })
    return 0


def _stream_then_hang(argv: List[str]) -> int:
    """Emit the stream-json preamble + one text delta, then block forever.

    Drives the cancellation tests (audit §3.7: CLI subprocess
    orphaning): a consumer that closes the stream generator mid-output
    leaves this child alive unless the runner kills the process group —
    the fake stays asleep so the test can poll the pid. The child's PID
    is written to ``$FAKE_CLAUDE_PID_FILE`` (when set) BEFORE the first
    line so the test can observe the process directly; writing it here
    rather than in ``main()`` keeps the ``--version`` handshake child
    from clobbering the file.
    """
    pid_file = os.environ.get("FAKE_CLAUDE_PID_FILE", "")
    if pid_file:
        with open(pid_file, "w") as fh:
            fh.write(str(os.getpid()))
    sid = "fake-hang-stream-1"
    _emit_line({
        "type": "system", "subtype": "init", "session_id": sid,
        "model": "claude-sonnet-4-6", "pid": os.getpid(),
    })
    _emit_line({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "text_delta", "text": "first chunk "}},
        "session_id": sid, "uuid": "u-d0",
    })
    time.sleep(600)
    return 0


def _permission_fail(argv: List[str]) -> int:
    sys.stderr.write("permission denied: tool Bash blocked by policy\n")
    return 1


def _crash(argv: List[str]) -> int:
    sys.stderr.write("internal error: something blew up\n")
    return 2


def _hang(argv: List[str]) -> int:
    time.sleep(60)
    return 0


def _echo_argv(argv: List[str]) -> int:
    _emit_json({
        "type": "result",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(argv)}],
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })
    return 0


SCENARIOS = {
    "ok_text": _ok_text,
    "ok_tool_use": _ok_tool_use,
    "ok_thinking": _ok_thinking,
    "ok_message_form": _ok_message_form,
    "ok_stream_event": _ok_stream_event,
    "ok_stream_event_tools": _ok_stream_event_tools,
    "ok_result_envelope": _ok_result_envelope,
    "stream_unknown_lines": _stream_unknown_lines,
    "stream_then_hang": _stream_then_hang,
    "message_form_auth_failed": _message_form_auth_failed,
    "auth_fail": _auth_fail,
    "permission_fail": _permission_fail,
    "crash": _crash,
    "hang": _hang,
    "echo_argv": _echo_argv,
}

#: Reported by ``--version`` — matches the real CLI's output shape
#: ("<semver> (Claude Code)") so version-parsing code sees a realistic
#: string, while staying obviously fake in logs/assertions.
FAKE_VERSION = "2.1.149-fake (Claude Code)"


def main() -> int:
    argv = sys.argv[1:]
    # Version handshake — answered before scenario dispatch, exactly like
    # the real binary (a hung/odd scenario must not corrupt the probe).
    if "--version" in argv:
        sys.stdout.write(FAKE_VERSION + "\n")
        return 0
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "ok_text")
    fn = SCENARIOS.get(scenario)
    if fn is None:
        sys.stderr.write(f"fake_claude: unknown scenario {scenario!r}\n")
        return 99
    return fn(argv)


if __name__ == "__main__":
    sys.exit(main())
