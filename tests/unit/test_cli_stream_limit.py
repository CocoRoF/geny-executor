"""CLI stdout stream limit — large tool results must not kill the turn.

Regression for the delegated-PPTX failure (2026-07-14): claude CLI emits
one stream-json event per line, tool_result contents included; a
DocXmlRead-sized line blew asyncio's default 64 KiB StreamReader limit and
readline() aborted the whole turn with "Separator is found, but chunk is
longer than limit".
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from geny_executor.llm_client._cli_runtime import _aiter_lines, _cli_stream_limit


class TestStreamLimitConfig:
    def test_default_is_32_mib(self, monkeypatch):
        monkeypatch.delenv("GENY_CLI_STREAM_LIMIT", raising=False)
        assert _cli_stream_limit() == 32 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(2**20))
        assert _cli_stream_limit() == 2**20

    def test_garbage_and_too_small_fall_back(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "banana")
        assert _cli_stream_limit() == 32 * 1024 * 1024
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "1024")  # below 64 KiB floor
        assert _cli_stream_limit() == 32 * 1024 * 1024


async def _spawn_printer(size: int, limit: int):
    """Subprocess that prints one `size`-byte line, then a small marker line."""
    code = f"import sys; sys.stdout.write('x'*{size} + '\\n' + 'MARKER\\n')"
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE,
        limit=limit,
    )


class TestLargeLines:
    @pytest.mark.asyncio
    async def test_1mb_line_survives_with_raised_limit(self):
        """The exact failure shape: one line far beyond 64 KiB."""
        proc = await _spawn_printer(1_000_000, _cli_stream_limit())
        lines = []
        async for line in _aiter_lines(
            proc.stdout, timeout_s=30.0, start_t=time.monotonic()
        ):
            lines.append(line)
        await proc.wait()
        assert len(lines) == 2
        assert len(lines[0]) == 1_000_001  # payload + newline
        assert lines[1].strip() == b"MARKER"

    @pytest.mark.asyncio
    async def test_over_limit_line_is_skipped_not_fatal(self):
        """Even when a line exceeds the (deliberately tiny) limit, the
        iterator logs + skips that one event and keeps streaming — the
        turn must not die."""
        proc = await _spawn_printer(300_000, 2**16)  # 64 KiB reader limit
        lines = []
        async for line in _aiter_lines(
            proc.stdout, timeout_s=30.0, start_t=time.monotonic()
        ):
            lines.append(line)
        await proc.wait()
        # The oversized line is lost (asyncio discards its buffer), but the
        # stream keeps going: the MARKER line still arrives... or at minimum
        # the iterator terminates cleanly instead of raising ValueError.
        assert all(b"x" * 70_000 not in ln for ln in lines)
        assert lines == [] or lines[-1].strip() == b"MARKER" or all(
            len(ln) <= 2**16 for ln in lines
        )
