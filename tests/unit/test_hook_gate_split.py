"""2.2.0 hooks gate split tests (audit 2026-06-09 §1-5).

The ``GENY_ALLOW_HOOKS`` env opt-in gates ONLY subprocess spawning.
In-process handlers fire on ``HookConfig.enabled`` alone — the audit
found GAPT forging the env var just to run its in-process policy
engine, which is the conflation these tests pin against regressing.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import Any, List

import pytest

from geny_executor.hooks.config import HookConfig, HookConfigEntry
from geny_executor.hooks.events import HookEvent, HookEventPayload, HookOutcome
from geny_executor.hooks.runner import HookRunner


def _payload() -> HookEventPayload:
    return HookEventPayload(
        event=HookEvent.PRE_TOOL_USE,
        session_id="s1",
        timestamp="2026-06-10T00:00:00Z",
        tool_name="test",
    )


def _write_script(tmp_path: Path, name: str, body_lines: List[str]) -> Path:
    path = tmp_path / name
    path.write_text("#!{}\n".format(sys.executable) + "\n".join(body_lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _sentinel_config(tmp_path: Path, sentinel: Path, *, enabled: bool = True) -> HookConfig:
    """Config with one PRE_TOOL_USE subprocess hook that touches a sentinel."""
    script = _write_script(
        tmp_path,
        "hook.py",
        [
            "import json, sys",
            f"open({str(sentinel)!r}, 'w').write('hook ran')",
            "json.dump({}, sys.stdout)",
        ],
    )
    return HookConfig(
        enabled=enabled,
        entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
    )


# ── In-process handlers fire WITHOUT the env opt-in ──────────────────


class TestInProcessWithoutEnvVar:
    @pytest.mark.asyncio
    async def test_handler_fires_without_env_opt_in(self):
        # env={} → GENY_ALLOW_HOOKS unset. config.enabled=True suffices.
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        called: List[Any] = []

        async def handler(payload):
            called.append(payload)
            return None

        runner.register_in_process(HookEvent.PRE_TOOL_USE, handler)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert len(called) == 1
        assert outcome.blocked is False

    @pytest.mark.asyncio
    async def test_blocking_handler_blocks_without_env_opt_in(self):
        # The GAPT scenario: an in-process policy engine must be able
        # to deny without forging the subprocess security env var.
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        runner.register_in_process(
            HookEvent.PRE_TOOL_USE,
            lambda p: HookOutcome.block("policy says no"),
        )
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is True
        assert outcome.stop_reason == "policy says no"

    @pytest.mark.asyncio
    async def test_config_disabled_still_skips_in_process(self):
        # config.enabled=False kills BOTH layers — unchanged posture.
        runner = HookRunner(HookConfig(enabled=False, entries={}), env={"GENY_ALLOW_HOOKS": "1"})
        called: List[Any] = []
        runner.register_in_process(HookEvent.PRE_TOOL_USE, lambda p: called.append(1) or None)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert called == []
        assert outcome.blocked is False


# ── Subprocess hooks STILL require the env opt-in ────────────────────


class TestSubprocessStillEnvGated:
    @pytest.mark.asyncio
    async def test_subprocess_does_not_spawn_without_env_opt_in(self, tmp_path):
        sentinel = tmp_path / "sentinel.txt"
        cfg = _sentinel_config(tmp_path, sentinel)
        runner = HookRunner(cfg, env={})  # no GENY_ALLOW_HOOKS

        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is False
        assert not sentinel.exists()

    @pytest.mark.asyncio
    async def test_subprocess_spawns_with_env_opt_in(self, tmp_path):
        sentinel = tmp_path / "sentinel.txt"
        cfg = _sentinel_config(tmp_path, sentinel)
        runner = HookRunner(cfg, env={"GENY_ALLOW_HOOKS": "1"})

        await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert sentinel.exists()

    @pytest.mark.asyncio
    async def test_in_process_fires_while_subprocess_locked_out(self, tmp_path):
        # Mixed config: in-process handler runs, subprocess entry for
        # the same event does not — and the handler's (non-blocking)
        # outcome survives.
        sentinel = tmp_path / "sentinel.txt"
        cfg = _sentinel_config(tmp_path, sentinel)
        runner = HookRunner(cfg, env={})
        called: List[Any] = []

        async def handler(payload):
            called.append(payload)
            return HookOutcome(suppress_output=True)

        runner.register_in_process(HookEvent.PRE_TOOL_USE, handler)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert len(called) == 1
        assert not sentinel.exists()
        assert outcome.suppress_output is True


# ── Property surface ─────────────────────────────────────────────────


class TestGateProperties:
    def test_enabled_means_subprocess_fully_enabled(self):
        cfg = HookConfig(enabled=True, entries={})
        assert HookRunner(cfg, env={}).enabled is False
        assert HookRunner(cfg, env={"GENY_ALLOW_HOOKS": "1"}).enabled is True

    def test_in_process_enabled_tracks_config_alone(self):
        on = HookConfig(enabled=True, entries={})
        off = HookConfig(enabled=False, entries={})
        assert HookRunner(on, env={}).in_process_enabled is True
        assert HookRunner(on, env={"GENY_ALLOW_HOOKS": "1"}).in_process_enabled is True
        assert HookRunner(off, env={"GENY_ALLOW_HOOKS": "1"}).in_process_enabled is False
