"""CLI subprocess cancellation / orphaning suite (2.2.0 wave 3).

Audit 2026-06-09 §3.7: "cancellation 테스트 0 (run_stream 소비자 이탈,
CLI subprocess 고아화)". Both hosts are SSE servers — a client
disconnect closes the async generator chain mid-stream, and whatever
the CLI runner does (or fails to do) at that moment decides whether a
real ``claude`` process is left running on the box.

Pinned here, against the fake binary in ``tests/_fixtures/fake_claude.py``
(scenario ``stream_then_hang`` emits one delta then sleeps, writing its
pid to ``$FAKE_CLAUDE_PID_FILE`` so tests can poll the process):

  * ``_kill_tree`` semantics — SIGTERM→grace→SIGKILL on the process
    group actually reaps a live child, and is a no-op on an exited one.
  * The CLITimeout path through ``stream()`` kills the child (the one
    cancellation path that works today).
  * THE PREDICTED BUG (strict xfail): closing the ``stream()``
    generator mid-output — exactly what a consumer disconnect does —
    does NOT kill the child and does NOT reap the stdin-drain task.
    ``stream()``'s ``finally`` block only cancels the stderr collector;
    the fix belongs there (kill_tree + drain-task cancel alongside
    ``stderr_task.cancel()``), at which point these xfails flip.
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

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client._cli_runtime import (
    CLIProcessRunner,
    CLITimeout,
    aiter_bytes,
)
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient


FAKE_CLAUDE = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")
)

#: argv shape mirrors a real streaming invocation; the fake's scenario
#: dispatch ignores it (everything but ``--version`` is env-driven).
STREAM_ARGV = ["--print", "--output-format", "stream-json"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner(scenario: str, pid_file: Path | None = None, **kwargs) -> CLIProcessRunner:
    env_extras = {"FAKE_CLAUDE_SCENARIO": scenario}
    if pid_file is not None:
        env_extras["FAKE_CLAUDE_PID_FILE"] = str(pid_file)
    defaults = dict(binary=FAKE_CLAUDE, timeout_s=10.0, kill_grace_s=0.5)
    defaults.update(kwargs)
    return CLIProcessRunner(env_extras=env_extras, **defaults)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — alive but not ours
        return True
    return True


async def _read_pid(pid_file: Path, timeout: float = 5.0) -> int:
    """Poll until the child wrote its pid file."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid_file.exists():
            content = pid_file.read_text().strip()
            if content:
                return int(content)
        await asyncio.sleep(0.01)
    raise AssertionError(f"child never wrote {pid_file}")


async def _wait_pid_dead(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        await asyncio.sleep(0.02)
    return not _pid_alive(pid)


def _force_kill(pid: int) -> None:
    """Test cleanup — never leave a 600s-sleeping fake child behind."""
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        except Exception:  # noqa: BLE001 — cleanup must not raise
            return


def _pending_drain_tasks() -> set:
    return {
        t
        for t in asyncio.all_tasks()
        if not t.done() and t.get_coro().__qualname__.startswith("_drain_stdin")
    }


# ---------------------------------------------------------------------------
# _kill_tree semantics (the documented SIGTERM → grace → SIGKILL ladder)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_tree_reaps_a_live_child() -> None:
    runner = _runner("hang")
    proc, _t0 = await runner._spawn(STREAM_ARGV)
    assert _pid_alive(proc.pid)
    try:
        await runner._kill_tree(proc)

        assert proc.returncode is not None
        assert await _wait_pid_dead(proc.pid, 2.0)
    finally:
        _force_kill(proc.pid)


@pytest.mark.asyncio
async def test_kill_tree_is_noop_on_exited_process() -> None:
    runner = _runner("ok_text")
    result = await runner.run_oneshot(["--print", "hi"])
    assert result.returncode == 0
    # run_oneshot already reaped the child; calling the kill ladder on a
    # dead process must neither raise nor hang.
    proc, _t0 = await runner._spawn(["--version"])
    await proc.wait()
    await runner._kill_tree(proc)


@pytest.mark.asyncio
async def test_stream_timeout_kills_child(tmp_path: Path) -> None:
    """The CLITimeout path is the one consumer-side abort that reaps the
    child today — pin it so the xfail below stays a one-bug story."""
    pid_file = tmp_path / "pid"
    runner = _runner("stream_then_hang", pid_file, timeout_s=0.5)

    lines = []
    with pytest.raises(CLITimeout):
        async for line in runner.stream(STREAM_ARGV):
            lines.append(line)

    assert lines, "the child emitted its preamble before hanging"
    pid = await _read_pid(pid_file)
    try:
        assert await _wait_pid_dead(pid, 3.0), (
            f"child {pid} survived the CLITimeout kill ladder"
        )
    finally:
        _force_kill(pid)


# ---------------------------------------------------------------------------
# Consumer disconnect — generator close mid-output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "audit §3.7 predicted bug: closing CLIProcessRunner.stream() "
        "mid-output (the SSE consumer-disconnect path) does not kill the "
        "child — stream()'s finally block only cancels the stderr task. "
        "Fix belongs in _cli_runtime.stream()'s finally: _kill_tree(proc) "
        "when the generator exits before the child has been reaped."
    ),
    strict=True,
)
async def test_runner_stream_generator_close_kills_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    runner = _runner("stream_then_hang", pid_file)

    agen = runner.stream(STREAM_ARGV)
    pid = -1
    try:
        first = await asyncio.wait_for(agen.__anext__(), 5.0)
        assert first  # the preamble line arrived — child is mid-stream
        pid = await _read_pid(pid_file)
        assert _pid_alive(pid)

        await agen.aclose()  # consumer disconnect

        assert await _wait_pid_dead(pid, 1.5), (
            f"child {pid} orphaned after generator close — "
            "stream() never ran the kill ladder"
        )
    finally:
        if pid > 0:
            _force_kill(pid)


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "audit §3.7 companion leak: stream() spawns the _drain_stdin task "
        "fire-and-forget; on generator close nothing cancels it, so a "
        "still-producing stdin_iter pins the task (and the child's stdin "
        "pipe) forever. Fix belongs in the same _cli_runtime.stream() "
        "finally block: keep the task handle and cancel it next to "
        "stderr_task.cancel()."
    ),
    strict=True,
)
async def test_runner_stream_generator_close_reaps_stdin_drain_task(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pid"
    runner = _runner("stream_then_hang", pid_file)

    unblock = asyncio.Event()

    async def _blocked_stdin():
        yield b'{"type": "user"}\n'
        await unblock.wait()  # a host feeding turns as they arrive

    assert _pending_drain_tasks() == set()
    agen = runner.stream(STREAM_ARGV, stdin_iter=_blocked_stdin())
    pid = -1
    try:
        await asyncio.wait_for(agen.__anext__(), 5.0)
        pid = await _read_pid(pid_file)
        assert _pending_drain_tasks(), "drain task should be alive mid-stream"

        await agen.aclose()  # consumer disconnect
        await asyncio.sleep(0.05)

        assert _pending_drain_tasks() == set(), (
            "stdin-drain task leaked after generator close"
        )
    finally:
        unblock.set()  # let the leaked task unwind so it can't poison later tests
        await asyncio.sleep(0)
        if pid > 0:
            _force_kill(pid)


@pytest.mark.asyncio
async def test_stdin_drain_task_finishes_on_normal_completion(tmp_path: Path) -> None:
    """Baseline for the leak xfail above: when the stream is consumed to
    the end, the drain task (finite stdin) finishes on its own."""
    runner = _runner("ok_stream_event")

    lines = [
        line
        async for line in runner.stream(STREAM_ARGV, stdin_iter=aiter_bytes(b"{}\n"))
    ]
    assert lines
    await asyncio.sleep(0.05)
    assert _pending_drain_tasks() == set()


# ---------------------------------------------------------------------------
# Client-level: ClaudeCodeCLIClient.create_message_stream close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "audit §3.7 predicted bug, client surface: aclosing "
        "ClaudeCodeCLIClient.create_message_stream mid-answer orphans the "
        "CLI child. Same root cause as the runner-level xfail — the fix "
        "is _cli_runtime.stream()'s finally block killing the tree; this "
        "test pins that the fix actually reaches the surface hosts call."
    ),
    strict=True,
)
async def test_client_stream_close_kills_child_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    client = ClaudeCodeCLIClient(
        binary_path=FAKE_CLAUDE,
        workspace_dir=str(tmp_path),
        api_key="sk-fake",
        bare_mode=True,
        timeout_s=10.0,
        env_extras={
            "FAKE_CLAUDE_SCENARIO": "stream_then_hang",
            "FAKE_CLAUDE_PID_FILE": str(pid_file),
        },
    )

    agen = client.create_message_stream(
        model_config=ModelConfig(model="claude-sonnet-4-6", max_tokens=64),
        messages=[{"role": "user", "content": "hi"}],
    )
    pid = -1
    try:
        # Drive until the first streamed delta — the moment a browser
        # tab would close on a half-rendered answer.
        async for event in agen:
            if event.get("type") == "text_delta":
                break
        pid = await _read_pid(pid_file)
        assert _pid_alive(pid)

        await agen.aclose()
        # The inner runner.stream generator is finalized asynchronously;
        # give the loop a few ticks before judging.
        for _ in range(5):
            await asyncio.sleep(0.02)

        assert await _wait_pid_dead(pid, 1.5), (
            f"CLI child {pid} orphaned after client stream close"
        )
    finally:
        if pid > 0:
            _force_kill(pid)
