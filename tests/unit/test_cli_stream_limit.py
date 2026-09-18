"""claude stdout stream limit — a huge line must not kill the turn.

Regression for the delegated-PPTX failure (2026-07-14): the claude CLI
emits one stream-json event per line, and the model's own text rides
INSIDE those lines; a 200K-char answer blew asyncio's default
StreamReader limit (64 KiB) and ``readline()`` aborted the whole turn
with "Separator is found, but chunk is longer than limit".

The runner that first learned this lesson is gone (2.68.0 — no backend
owns the agentic loop any more), so the lesson now lives on the one
client that still spawns ``claude``: :class:`ClaudeCodeTokenClient`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import List

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.claude_code_tokens import (
    ClaudeCodeTokenClient,
    _stream_limit,
)


class TestStreamLimitConfig:
    def test_default_is_32_mib(self, monkeypatch):
        monkeypatch.delenv("GENY_CLI_STREAM_LIMIT", raising=False)
        assert _stream_limit() == 32 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(2**20))
        assert _stream_limit() == 2**20

    def test_garbage_and_too_small_fall_back(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "banana")
        assert _stream_limit() == 32 * 1024 * 1024
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "1024")  # below the 64 KiB floor
        assert _stream_limit() == 32 * 1024 * 1024


def _fake_claude(tmp_path: Path, body: str) -> str:
    """A stand-in ``claude`` that prints stream-json lines we choose."""
    path = tmp_path / "claude"
    path.write_text("#!/usr/bin/env python3\nimport json, sys\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


#: one assistant line far beyond 64 KiB, then the terminal result line.
_HUGE_ANSWER = (
    "sys.stdin.read()\n"
    "big = 'x' * 1_000_000\n"
    "print(json.dumps({'type': 'assistant', 'message': "
    "{'content': [{'type': 'text', 'text': big}]}}))\n"
    "print(json.dumps({'type': 'result', 'subtype': 'success', 'result': big}))\n"
)


async def _collect(client: ClaudeCodeTokenClient) -> List[dict]:
    events = []
    async for ev in client.create_message_stream(
        model_config=ModelConfig(model="sonnet", max_tokens=64),
        messages=[{"role": "user", "content": "go"}],
    ):
        events.append(ev)
    return events


class TestLargeLines:
    @pytest.mark.asyncio
    async def test_1mb_line_survives_with_the_default_limit(self, tmp_path, monkeypatch):
        """The exact failure shape: one line far beyond 64 KiB."""
        monkeypatch.delenv("GENY_CLI_STREAM_LIMIT", raising=False)
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, _HUGE_ANSWER))
        events = await _collect(client)
        completes = [e for e in events if e.get("type") == "message_complete"]
        assert completes, "stream ended without message_complete"
        assert len(completes[-1]["response"].text) >= 1_000_000

    @pytest.mark.asyncio
    async def test_over_limit_line_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """Even when a line exceeds a (deliberately tiny) limit, the client
        logs + skips that one event and keeps reading — the turn survives."""
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(2**16))
        body = (
            "sys.stdin.read()\n"
            "print(json.dumps({'type': 'assistant', 'message': "
            "{'content': [{'type': 'text', 'text': 'x' * 300_000}]}}))\n"
            "print(json.dumps({'type': 'result', 'subtype': 'success', "
            "'result': 'ok'}))\n"
        )
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, body))
        events = await _collect(client)
        completes = [e for e in events if e.get("type") == "message_complete"]
        # The turn SURVIVES — that is the whole guarantee. Content is a
        # different matter: asyncio clears its entire buffer when the
        # separator is past the limit, so whatever shared that read (here
        # the terminal result line) goes with it. Losing a turn's text
        # beats losing the turn, which is why the default is 32 MiB and
        # this tiny limit only ever appears in this test.
        assert completes, "an oversized line killed the turn"
        assert "x" * 70_000 not in completes[-1]["response"].text


def test_env_knob_reaches_the_spawn(monkeypatch):
    """The knob is not decoration: it is what the subprocess is given."""
    monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(4 * 2**20))
    assert _stream_limit() == 4 * 2**20
    src = (
        Path(os.path.dirname(__file__)).parents[1]
        / "src/geny_executor/llm_client/claude_code_tokens.py"
    ).read_text()
    assert "limit=_stream_limit()," in src
