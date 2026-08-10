"""The hot spare must be alive, not merely un-reaped.

The prewarm exists for time-to-first-token: after a turn, a `claude` process
is started so the next turn skips Node boot, auth and MCP startup. The turn
then hands its prompt to that process over stdin.

`Process.returncode` is bookkeeping — it stays None until the event loop
reaps the child — so a process that already exited reads as a healthy spare
forever. In production the CLI started, listed its tools, exited, and every
following turn stalled until a watchdog abandoned it at 320 s. The same
session answered in 11 s with the prewarm disabled.
"""

from __future__ import annotations

import os

import pytest

from geny_executor.llm_client.claude_code import (
    ClaudeCodeCLIClient,
    _process_alive,
)


class _Proc:
    def __init__(self, pid, returncode=None):
        self.pid = pid
        self.returncode = returncode


def _client() -> ClaudeCodeCLIClient:
    return ClaudeCodeCLIClient.__new__(ClaudeCodeCLIClient)


def _with_spare(argv, proc):
    c = _client()
    c._spare = {"proc": proc, "argv": list(argv), "runner": None}
    c._discarded = []
    c._discard_spare_proc = lambda spare: c._discarded.append(spare)
    return c


# ── the liveness probe itself ───────────────────────────────────────

def test_a_running_process_is_alive():
    assert _process_alive(_Proc(os.getpid())) is True


def test_a_reaped_process_is_not():
    assert _process_alive(_Proc(os.getpid(), returncode=0)) is False


def test_a_vanished_process_is_not_alive_despite_no_returncode():
    """THE bug. returncode is None because nobody reaped it; the process is
    gone all the same."""
    pid = os.fork()
    if pid == 0:                      # child
        os._exit(0)
    os.waitpid(pid, 0)                # reap outside asyncio's view

    assert _process_alive(_Proc(pid)) is False


def test_no_pid_is_not_alive():
    assert _process_alive(_Proc(None)) is False


# ── what _take_spare does with it ───────────────────────────────────

def test_a_live_matching_spare_is_handed_over():
    proc = _Proc(os.getpid())
    c = _with_spare(["claude", "--model", "opus"], proc)

    assert c._take_spare(["claude", "--model", "opus"]) is proc
    assert c._spare is None, "the spare was handed over AND kept"


def test_a_dead_spare_is_discarded_not_reused():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    c = _with_spare(["claude"], _Proc(pid))

    assert c._take_spare(["claude"]) is None
    assert len(c._discarded) == 1


def test_a_spare_booted_with_different_argv_is_discarded():
    """Model, MCP config, resume flags — any drift means stale config."""
    c = _with_spare(["claude", "--model", "opus"], _Proc(os.getpid()))

    assert c._take_spare(["claude", "--model", "haiku"]) is None
    assert len(c._discarded) == 1


def test_no_spare_is_not_an_error():
    c = _client()
    c._spare = None
    assert c._take_spare(["claude"]) is None
