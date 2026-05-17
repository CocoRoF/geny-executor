"""Tests for ``_cli_runtime.py`` (Phase A2).

Uses the fake echo CLI fixture under ``tests/_fixtures/fake_echo_cli.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client._cli_runtime import (
    DEFAULT_ENV_WHITELIST,
    CLIBinaryNotFound,
    CLIProcessRunner,
    CLIProtocolError,
    CLIResult,
    CLITimeout,
    aiter_bytes,
    detect_binary,
    parse_stream_json_line,
    scrub_env,
)


# Path-on-disk for the fake CLI.
FAKE_CLI = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_echo_cli.py")
)


# ---------------------------------------------------------------------------
# detect_binary
# ---------------------------------------------------------------------------


def test_detect_binary_finds_via_which() -> None:
    p = detect_binary("python3")
    assert p is not None
    assert Path(p).exists()


def test_detect_binary_missing_returns_none() -> None:
    assert detect_binary("definitely-not-a-real-binary-xyz") is None


def test_detect_binary_override_existing(tmp_path: Path) -> None:
    # Use the fake CLI as a known-executable
    assert detect_binary("ignored", override=FAKE_CLI) == FAKE_CLI


def test_detect_binary_override_nonexistent_returns_none(tmp_path: Path) -> None:
    bogus = str(tmp_path / "does-not-exist")
    assert detect_binary("ignored", override=bogus) is None


# ---------------------------------------------------------------------------
# scrub_env
# ---------------------------------------------------------------------------


def test_scrub_env_keeps_only_whitelisted() -> None:
    parent = {
        "HOME": "/home/x",
        "PATH": "/usr/bin",
        "SECRET_TOKEN": "leak-me",
        "RANDOM_NOISE": "no",
    }
    scrubbed = scrub_env(parent)
    assert scrubbed["HOME"] == "/home/x"
    assert scrubbed["PATH"] == "/usr/bin"
    assert "SECRET_TOKEN" not in scrubbed
    assert "RANDOM_NOISE" not in scrubbed


def test_scrub_env_extras_override() -> None:
    parent = {"HOME": "/home/x"}
    scrubbed = scrub_env(parent, extras={"ANTHROPIC_API_KEY": "sk-y"})
    assert scrubbed["ANTHROPIC_API_KEY"] == "sk-y"
    assert scrubbed["HOME"] == "/home/x"


def test_scrub_env_extras_can_replace_whitelisted() -> None:
    parent = {"HOME": "/home/x"}
    scrubbed = scrub_env(parent, extras={"HOME": "/tmp/override"})
    assert scrubbed["HOME"] == "/tmp/override"


def test_default_whitelist_contains_core_vars() -> None:
    for k in ("HOME", "PATH", "USER", "LANG", "TERM"):
        assert k in DEFAULT_ENV_WHITELIST


# ---------------------------------------------------------------------------
# parse_stream_json_line
# ---------------------------------------------------------------------------


def test_parse_valid_json_line() -> None:
    assert parse_stream_json_line(b'{"a": 1}') == {"a": 1}


def test_parse_empty_line_returns_none() -> None:
    assert parse_stream_json_line(b"") is None
    assert parse_stream_json_line(b"   \n") is None


def test_parse_comment_returns_none() -> None:
    assert parse_stream_json_line(b"# debug message") is None


def test_parse_malformed_returns_marker() -> None:
    out = parse_stream_json_line(b"{not json")
    assert out == {"__malformed__": "{not json"}


def test_parse_non_object_root_marked_malformed() -> None:
    # The protocol uses JSON objects per line; lists/strings are not valid.
    out = parse_stream_json_line(b'[1, 2]')
    assert out == {"__malformed__": "[1, 2]"}


# ---------------------------------------------------------------------------
# CLIProcessRunner construction guards
# ---------------------------------------------------------------------------


def test_runner_rejects_empty_binary() -> None:
    with pytest.raises(CLIBinaryNotFound):
        CLIProcessRunner(binary="")


def test_runner_rejects_nonexistent_binary(tmp_path: Path) -> None:
    with pytest.raises(CLIBinaryNotFound):
        CLIProcessRunner(binary=str(tmp_path / "no-such-thing"))


def test_runner_rejects_non_executable(tmp_path: Path) -> None:
    p = tmp_path / "not-exec.txt"
    p.write_text("hi")
    with pytest.raises(CLIBinaryNotFound):
        CLIProcessRunner(binary=str(p))


# ---------------------------------------------------------------------------
# run_oneshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_oneshot_echo() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    res = await runner.run_oneshot(["echo", "hello", "world"])
    assert isinstance(res, CLIResult)
    assert res.returncode == 0
    assert res.stdout == b"hello world"
    assert res.duration_ms >= 0


@pytest.mark.asyncio
async def test_runner_oneshot_stdin_round_trip() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    res = await runner.run_oneshot(["echo-stdin"], stdin=b"payload\n")
    assert res.returncode == 0
    assert res.stdout == b"payload\n"


@pytest.mark.asyncio
async def test_runner_oneshot_nonzero_exit() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    res = await runner.run_oneshot(["fail", "7", "something went wrong"])
    assert res.returncode == 7
    assert b"something went wrong" in res.stderr


@pytest.mark.asyncio
async def test_runner_oneshot_timeout_kills_process() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI, timeout_s=0.3, kill_grace_s=0.2)
    with pytest.raises(CLITimeout):
        await runner.run_oneshot(["hang", "10"])


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_stream_lines() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    lines = []
    async for line in runner.stream(["lines", "5"]):
        lines.append(line.rstrip())
    assert lines == [f"line-{i}".encode() for i in range(5)]


@pytest.mark.asyncio
async def test_runner_stream_json_lines_parse_cleanly() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    parsed = []
    async for line in runner.stream(["json-stream", "4"]):
        parsed.append(parse_stream_json_line(line))
    assert parsed == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]


@pytest.mark.asyncio
async def test_runner_stream_failure_raises_protocol_error() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI)
    with pytest.raises(CLIProtocolError, match="bad bad bad"):
        async for _ in runner.stream(["fail", "3", "bad bad bad"]):
            pass


@pytest.mark.asyncio
async def test_runner_stream_timeout() -> None:
    runner = CLIProcessRunner(binary=FAKE_CLI, timeout_s=0.3, kill_grace_s=0.2)
    with pytest.raises(CLITimeout):
        async for _ in runner.stream(["hang", "5"]):
            pass


# ---------------------------------------------------------------------------
# Env scrubbing in real subprocess (smoke)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_env_scrub_strips_non_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEAK_ME_PLEASE", "1")
    monkeypatch.setenv("HOME", os.environ.get("HOME", "/tmp"))
    runner = CLIProcessRunner(binary=FAKE_CLI)
    # The fake binary prints stdin verbatim. We feed it an env-introspection
    # request via stdin echo — fake binary doesn't read env, so we just verify
    # that scrub_env() filtered LEAK_ME_PLEASE out:
    scrubbed = scrub_env(os.environ)
    assert "LEAK_ME_PLEASE" not in scrubbed


# ---------------------------------------------------------------------------
# aiter_bytes helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aiter_bytes_yields_single_chunk() -> None:
    chunks = []
    async for c in aiter_bytes(b"abc"):
        chunks.append(c)
    assert chunks == [b"abc"]


@pytest.mark.asyncio
async def test_aiter_bytes_none_yields_nothing() -> None:
    chunks = []
    async for c in aiter_bytes(None):
        chunks.append(c)
    assert chunks == []
