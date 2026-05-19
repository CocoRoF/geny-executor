#!/usr/bin/env python3
"""Tiny fake ``claude`` CLI used by Claude Code client tests.

Scenarios driven by the ``FAKE_CLAUDE_SCENARIO`` env var:

  - ``ok_text``        — emit a single text result. JSON mode prints one
                         ``{"type": "result", ...}`` blob. Streaming mode
                         emits a 3-line stream-json sequence.
  - ``ok_tool_use``    — single tool_use response (json mode only).
  - ``ok_thinking``    — emits a thinking block + text (streaming).
  - ``auth_fail``      — exit 1 with an auth-related stderr message.
  - ``permission_fail``— exit 1 with a permission-related stderr.
  - ``crash``          — exit 2 with generic stderr.
  - ``hang``           — sleep forever (callers must enforce timeout).
  - ``echo_argv``      — print argv as JSON; useful for shape assertions.

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
    "message_form_auth_failed": _message_form_auth_failed,
    "auth_fail": _auth_fail,
    "permission_fail": _permission_fail,
    "crash": _crash,
    "hang": _hang,
    "echo_argv": _echo_argv,
}


def main() -> int:
    argv = sys.argv[1:]
    scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "ok_text")
    fn = SCENARIOS.get(scenario)
    if fn is None:
        sys.stderr.write(f"fake_claude: unknown scenario {scenario!r}\n")
        return 99
    return fn(argv)


if __name__ == "__main__":
    sys.exit(main())
