"""A dead CLI must not be able to park a turn on its own stdout.

Production, 2026-08-10. Turns stopped answering. The stack said the
pipeline was inside ``_call_streaming``, awaiting the client's async
generator; the CLI trace said the very same invocation had exited
``rc=0`` with a complete answer, seconds in. Both were true.

A pipe reaches EOF when the LAST writer closes it, and the CLI is not
the only writer: it spawns MCP servers as children and they inherit its
stdout. One that outlives it — a bridge that misses the parent-death
signal, a server mid-request — holds the write end open, and
``readline()`` blocks on a pipe with nothing left to write to it. The
reader waited for EOF, so it waited for the full ``timeout_s``: 30
minutes of spinner over an answer that was already complete.

The fix reaps the child concurrently and, once it is gone, gives stdout
only ``exit_drain_grace_s`` to produce anything more. Buffered bytes
still arrive — draining a buffer is instant next to that grace.

``leaky_child_then_exit`` reproduces the shape exactly: full stream-json
answer, clean exit, one forked child sitting on the inherited FD.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client._cli_runtime import CLIProcessRunner


FAKE_CLAUDE = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")
)

STREAM_ARGV = ["--print", "--output-format", "stream-json"]


def _runner(scenario: str, **kwargs) -> CLIProcessRunner:
    defaults = dict(binary=FAKE_CLAUDE, timeout_s=30.0, kill_grace_s=0.5)
    defaults.update(kwargs)
    return CLIProcessRunner(
        env_extras={"FAKE_CLAUDE_SCENARIO": scenario}, **defaults
    )


async def _collect(runner: CLIProcessRunner) -> list[bytes]:
    return [line async for line in runner.stream(STREAM_ARGV)]


@pytest.mark.asyncio
async def test_exited_child_with_leaked_stdout_ends_the_stream() -> None:
    """The stream ends on the drain grace, not on ``timeout_s``.

    ``timeout_s`` is 30s and the grace 1s; the leaked child sleeps 600s.
    Before the fix this test hung for the full 30s and then raised
    ``CLITimeout``, losing a complete answer.
    """
    runner = _runner("leaky_child_then_exit", timeout_s=30.0,
                     exit_drain_grace_s=1.0)
    t0 = time.monotonic()
    lines = await asyncio.wait_for(_collect(runner), timeout=20.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 10.0, f"stream waited {elapsed:.1f}s on a dead child"
    # The answer survived: ending early must not truncate delivered output.
    assert lines, "no output collected"
    joined = b"".join(lines)
    assert b"content_block_delta" in joined or b"assistant" in joined


@pytest.mark.asyncio
async def test_drain_grace_is_measured_from_child_exit() -> None:
    """A longer grace costs exactly that much more, and no more.

    Pins the shape of the wait: the cost of a leaked FD is the grace,
    not the timeout — which is the whole point of the change.
    """
    runner = _runner("leaky_child_then_exit", timeout_s=30.0,
                     exit_drain_grace_s=3.0)
    t0 = time.monotonic()
    await asyncio.wait_for(_collect(runner), timeout=20.0)
    elapsed = time.monotonic() - t0
    assert 2.0 <= elapsed < 12.0, f"grace not honoured (took {elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_healthy_stream_is_unaffected() -> None:
    """No grandchild, no grace: a clean CLI still ends on real EOF.

    The guard must not shorten a normal stream — the common path has to
    stay exactly as fast as it was.
    """
    runner = _runner("ok_stream_event", timeout_s=30.0,
                     exit_drain_grace_s=5.0)
    t0 = time.monotonic()
    lines = await asyncio.wait_for(_collect(runner), timeout=20.0)
    elapsed = time.monotonic() - t0
    assert lines
    assert elapsed < 5.0, (
        f"clean stream paid the drain grace ({elapsed:.1f}s) — it should "
        "end on EOF immediately"
    )


@pytest.mark.asyncio
async def test_slow_but_alive_child_is_not_cut_off() -> None:
    """The grace applies only AFTER exit — a slow live child keeps its
    full budget.

    ``stream_then_hang`` emits one delta and sleeps 600s while alive, so
    the only thing that may stop it is ``timeout_s``. If the drain grace
    leaked into the alive path, this would end in ~1s with a CLITimeout
    raised far too early.
    """
    from geny_executor.llm_client._cli_runtime import CLITimeout

    runner = _runner("stream_then_hang", timeout_s=4.0,
                     exit_drain_grace_s=1.0)
    t0 = time.monotonic()
    with pytest.raises(CLITimeout):
        await asyncio.wait_for(_collect(runner), timeout=25.0)
    elapsed = time.monotonic() - t0
    assert elapsed >= 3.0, (
        f"alive child was cut off after {elapsed:.1f}s — the drain grace "
        "must not apply before exit"
    )
