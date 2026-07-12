"""Hot-spare CLI prewarm (TTFT program 2.50.0, finding C1).

The claude_code backend pays a full Node boot + auth + MCP startup per
turn because every call spawns a fresh ``claude --print`` process. The
hot spare boots the NEXT turn's process right after a streamed turn
completes; the following call claims it when the argv matches exactly
and feeds it stdin like any fresh spawn — identical semantics (full
history over stdin), boot cost prepaid.

Uses the ``wait_stdin_stream`` fake scenario, which blocks on stdin
before emitting — the same idle behavior as the real CLI's stream-json
input mode, so a booted spare stays alive until claimed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client._cli_runtime import CLIProcessRunner
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient

FAKE_CLAUDE = str((Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py"))


def _client(**kwargs: Any) -> ClaudeCodeCLIClient:
    env_extras = dict(kwargs.pop("env_extras", None) or {})
    env_extras.setdefault("FAKE_CLAUDE_SCENARIO", "wait_stdin_stream")
    defaults = dict(
        binary_path=FAKE_CLAUDE,
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        bare_mode=True,
        timeout_s=10.0,
        env_extras=env_extras,
    )
    defaults.update(kwargs)
    return ClaudeCodeCLIClient(**defaults)


async def _run_turn(client: ClaudeCodeCLIClient) -> str:
    chunks = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="sonnet", max_tokens=64),
        messages=[{"role": "user", "content": "hello"}],
    ):
        chunks.append(chunk)
    assert chunks[-1]["type"] == "message_complete"
    return chunks[-1]["response"].text


async def _wait_for_spare(client: ClaudeCodeCLIClient, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while client._spare is None:
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("spare was never scheduled")
        await asyncio.sleep(0.02)


async def _cleanup_spare(client: ClaudeCodeCLIClient) -> None:
    spare = client._spare
    client._spare = None
    if spare is not None:
        spare["expire"].cancel()
        proc = spare["proc"]
        if proc.returncode is None:
            await spare["runner"]._kill_tree(proc)


@pytest.mark.asyncio
async def test_spare_scheduled_after_streamed_turn():
    client = _client()
    try:
        text = await _run_turn(client)
        assert text  # the fake emitted real content
        await _wait_for_spare(client)
        spare = client._spare
        assert spare is not None
        assert spare["proc"].returncode is None  # alive, idle on stdin
    finally:
        await _cleanup_spare(client)


@pytest.mark.asyncio
async def test_second_turn_claims_the_spare(monkeypatch: pytest.MonkeyPatch):
    seen: list = []
    orig_stream = CLIProcessRunner.stream

    def spy(self, argv, *, stdin_iter=None, prespawned=None):
        seen.append(prespawned)
        return orig_stream(self, argv, stdin_iter=stdin_iter, prespawned=prespawned)

    monkeypatch.setattr(CLIProcessRunner, "stream", spy)

    client = _client()
    try:
        await _run_turn(client)
        await _wait_for_spare(client)
        spare_proc = client._spare["proc"]

        text = await _run_turn(client)
        assert text
        # Turn 1 had no spare; turn 2 was served by the prewarmed process.
        assert seen[0] is None
        assert seen[1] is spare_proc
    finally:
        await _cleanup_spare(client)


@pytest.mark.asyncio
async def test_config_drift_discards_the_spare():
    client = _client()
    try:
        await _run_turn(client)
        await _wait_for_spare(client)
        stale_proc = client._spare["proc"]

        # Different model → different argv → the spare must NOT serve it.
        claimed = client._take_spare(["--model", "opus", "--something-else"])
        assert claimed is None
        assert client._spare is None
        # The stale process is being reaped in the background.
        await asyncio.sleep(0.3)
        assert stale_proc.returncode is not None
    finally:
        await _cleanup_spare(client)


@pytest.mark.asyncio
async def test_dead_spare_is_discarded_at_claim_time():
    client = _client()
    try:
        await _run_turn(client)
        await _wait_for_spare(client)
        spare = client._spare
        spare["proc"].kill()
        await spare["proc"].wait()

        claimed = client._take_spare(list(spare["argv"]))
        assert claimed is None  # dead spare never serves a turn
    finally:
        await _cleanup_spare(client)


@pytest.mark.asyncio
async def test_prewarm_can_be_disabled():
    client = _client(prewarm_spawn=False)
    try:
        await _run_turn(client)
        await asyncio.sleep(0.2)
        assert client._spare is None
    finally:
        await _cleanup_spare(client)


def test_env_override_disables_prewarm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GENY_CLI_PREWARM", "0")
    client = _client()
    assert client._prewarm_spawn is False
    monkeypatch.setenv("GENY_CLI_PREWARM", "1")
    assert _client()._prewarm_spawn is True
