"""Phase 5 Week 9 — HookRunner + HookConfig tests.

Spawning real subprocesses keeps the test honest — the runner's whole
job is to manage subprocess lifecycle. Each test writes a small Python
hook script to ``tmp_path`` and invokes it. CI runs Python 3.11/3.12/
3.13 so ``sys.executable`` is reliably available.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from geny_executor.hooks import (
    DEFAULT_TIMEOUT_MS,
    HookConfig,
    HookConfigEntry,
    HookEvent,
    HookEventPayload,
    HookOutcome,
    HookRunner,
    hooks_opt_in_from_env,
    load_hooks_config,
    parse_hook_config,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _write_hook_script(
    tmp_path: Path,
    name: str,
    body_lines: List[str],
) -> Path:
    """Write an executable Python script and return its path."""
    path = tmp_path / name
    path.write_text("#!{}\n".format(sys.executable) + "\n".join(body_lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _payload(
    event: HookEvent = HookEvent.PRE_TOOL_USE,
    *,
    tool_name: str = "Bash",
    session_id: str = "sess-123",
) -> HookEventPayload:
    return HookEventPayload(
        event=event,
        session_id=session_id,
        timestamp="2026-04-24T12:00:00Z",
        tool_name=tool_name,
        tool_input={"command": "echo hi"},
    )


def _runner(
    config: HookConfig,
    *,
    extra_env: Dict[str, str] | None = None,
    opt_in: bool = True,
) -> HookRunner:
    env = dict(os.environ)
    if opt_in:
        env["GENY_ALLOW_HOOKS"] = "1"
    else:
        env.pop("GENY_ALLOW_HOOKS", None)
    if extra_env:
        env.update(extra_env)
    return HookRunner(config, env=env)


# ─────────────────────────────────────────────────────────────────
# Config parsing + opt-in detection
# ─────────────────────────────────────────────────────────────────


class TestConfigParsing:
    def test_disabled_default(self):
        cfg = HookConfig.disabled()
        assert cfg.enabled is False
        assert cfg.entries == {}

    def test_parse_minimal(self):
        cfg = parse_hook_config({"enabled": True, "hooks": {}})
        assert cfg.enabled is True
        assert cfg.entries == {}

    def test_parse_full(self):
        raw = {
            "enabled": True,
            "audit_log_path": "/tmp/x.log",
            "hooks": {
                "pre_tool_use": [
                    {
                        "command": "/bin/true",
                        "args": ["--x"],
                        "timeout_ms": 1000,
                        "match": {"tool": "Bash"},
                        "env": {"K": "V"},
                        "working_dir": "/tmp",
                    }
                ],
                "post_tool_use": [{"command": "/bin/echo"}],
            },
        }
        cfg = parse_hook_config(raw)
        assert cfg.enabled is True
        assert cfg.audit_log_path == "/tmp/x.log"
        pre = cfg.entries_for(HookEvent.PRE_TOOL_USE)
        assert len(pre) == 1
        entry = pre[0]
        assert entry.command == "/bin/true"
        assert entry.args == ["--x"]
        assert entry.timeout_ms == 1000
        assert entry.match == {"tool": "Bash"}
        assert entry.env == {"K": "V"}
        assert entry.working_dir == "/tmp"
        post = cfg.entries_for(HookEvent.POST_TOOL_USE)
        assert len(post) == 1 and post[0].command == "/bin/echo"

    def test_unknown_event_warns_and_skips(self, caplog):
        caplog.set_level("WARNING")
        cfg = parse_hook_config(
            {"enabled": True, "hooks": {"future_event": [{"command": "/bin/true"}]}}
        )
        assert cfg.entries == {}
        assert any("future_event" in r.message for r in caplog.records)

    def test_missing_command_raises(self):
        with pytest.raises(ValueError, match="command"):
            parse_hook_config(
                {"enabled": True, "hooks": {"pre_tool_use": [{"args": []}]}}
            )

    def test_non_list_args_raises(self):
        with pytest.raises(ValueError, match="args"):
            parse_hook_config(
                {
                    "enabled": True,
                    "hooks": {"pre_tool_use": [{"command": "/bin/x", "args": "oops"}]},
                }
            )

    def test_non_positive_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout_ms"):
            parse_hook_config(
                {
                    "enabled": True,
                    "hooks": {
                        "pre_tool_use": [{"command": "/bin/x", "timeout_ms": 0}]
                    },
                }
            )

    def test_non_dict_root_raises(self):
        with pytest.raises(ValueError, match="root"):
            parse_hook_config([1, 2, 3])

    def test_load_hooks_config_missing_file_returns_disabled(self, tmp_path):
        cfg = load_hooks_config(tmp_path / "missing.yaml")
        assert cfg.enabled is False

    def test_load_hooks_config_empty_returns_disabled(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_hooks_config(path)
        assert cfg.enabled is False

    def test_load_hooks_config_round_trip(self, tmp_path):
        path = tmp_path / "h.yaml"
        path.write_text(
            "enabled: true\n"
            "hooks:\n"
            "  pre_tool_use:\n"
            "    - command: /bin/true\n"
            "      timeout_ms: 100\n"
        )
        cfg = load_hooks_config(path)
        assert cfg.enabled is True
        entries = cfg.entries_for(HookEvent.PRE_TOOL_USE)
        assert len(entries) == 1
        assert entries[0].timeout_ms == 100


class TestOptInDetection:
    def test_unset_is_false(self):
        assert hooks_opt_in_from_env({}) is False

    def test_truthy_values(self):
        for raw in ("1", "true", "True", "YES", "on"):
            assert hooks_opt_in_from_env({"GENY_ALLOW_HOOKS": raw}) is True

    def test_falsy_values(self):
        for raw in ("0", "false", "no", "off", ""):
            assert hooks_opt_in_from_env({"GENY_ALLOW_HOOKS": raw}) is False


# ─────────────────────────────────────────────────────────────────
# Match expressions
# ─────────────────────────────────────────────────────────────────


class TestEntryMatching:
    def test_empty_match_matches_all(self):
        e = HookConfigEntry(command="/bin/true", match={})
        assert e.matches(HookEvent.PRE_TOOL_USE, "Bash") is True
        assert e.matches(HookEvent.PRE_TOOL_USE, None) is True

    def test_tool_filter_exact_match(self):
        e = HookConfigEntry(command="/bin/true", match={"tool": "Bash"})
        assert e.matches(HookEvent.PRE_TOOL_USE, "Bash") is True
        assert e.matches(HookEvent.PRE_TOOL_USE, "Read") is False

    def test_tool_filter_with_no_tool_name_fails(self):
        e = HookConfigEntry(command="/bin/true", match={"tool": "Bash"})
        assert e.matches(HookEvent.PRE_TOOL_USE, None) is False


# ─────────────────────────────────────────────────────────────────
# Runner — opt-in gating
# ─────────────────────────────────────────────────────────────────


class TestRunnerGating:
    @pytest.mark.asyncio
    async def test_disabled_config_passthrough(self, tmp_path):
        runner = _runner(HookConfig.disabled())
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert outcome.decision is None

    @pytest.mark.asyncio
    async def test_no_env_opt_in_passthrough(self, tmp_path):
        cfg = HookConfig(enabled=True)
        runner = _runner(cfg, opt_in=False)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True

    @pytest.mark.asyncio
    async def test_no_entries_passthrough(self):
        cfg = HookConfig(enabled=True, entries={})
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True

    @pytest.mark.asyncio
    async def test_no_matching_entries_passthrough(self, tmp_path):
        # Hook registered for tool=Read; payload has tool=Bash → no match.
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(command="/bin/true", match={"tool": "Read"}),
                ]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload(tool_name="Bash"))
        assert outcome.continue_ is True


# ─────────────────────────────────────────────────────────────────
# Runner — happy-path execution
# ─────────────────────────────────────────────────────────────────


class TestRunnerHappyPath:
    @pytest.mark.asyncio
    async def test_passthrough_outcome_when_stdout_empty(self, tmp_path):
        # /bin/true exits 0 with no output → passthrough
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command="/bin/true")]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert outcome.decision is None

    @pytest.mark.asyncio
    async def test_block_outcome(self, tmp_path):
        script = _write_hook_script(
            tmp_path,
            "block.py",
            [
                "import json, sys",
                "json.dump({'continue': False, 'decision': 'block', 'stop_reason': 'nope'}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is True
        assert outcome.decision == "block"
        assert outcome.stop_reason == "nope"

    @pytest.mark.asyncio
    async def test_modify_input_outcome(self, tmp_path):
        script = _write_hook_script(
            tmp_path,
            "modify.py",
            [
                "import json, sys",
                "json.dump({'modified_input': {'command': 'echo SAFE'}}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.modified_input == {"command": "echo SAFE"}

    @pytest.mark.asyncio
    async def test_payload_reaches_hook_via_stdin(self, tmp_path):
        # Hook echoes back its received session_id in stop_reason for assertion.
        script = _write_hook_script(
            tmp_path,
            "echo.py",
            [
                "import json, sys",
                "data = json.loads(sys.stdin.read())",
                "json.dump({'continue': False, 'stop_reason': data['session_id']}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(
            HookEvent.PRE_TOOL_USE, _payload(session_id="abc-xyz")
        )
        assert outcome.stop_reason == "abc-xyz"

    @pytest.mark.asyncio
    async def test_env_vars_forwarded(self, tmp_path):
        script = _write_hook_script(
            tmp_path,
            "env.py",
            [
                "import json, os, sys",
                "json.dump({'stop_reason': os.environ.get('CUSTOM_VAR', 'missing')}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(
                        command=str(script), env={"CUSTOM_VAR": "value-here"}
                    )
                ]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.stop_reason == "value-here"


# ─────────────────────────────────────────────────────────────────
# Runner — failure isolation
# ─────────────────────────────────────────────────────────────────


class TestRunnerFailureModes:
    @pytest.mark.asyncio
    async def test_command_not_found_fails_open(self, tmp_path, caplog):
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(command="/this/does/not/exist")
                ]
            },
        )
        runner = _runner(cfg)
        caplog.set_level("WARNING")
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert any("not found" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_zero_exit_fails_open(self, tmp_path, caplog):
        script = _write_hook_script(
            tmp_path,
            "fail.py",
            [
                "import sys",
                "print('explosion', file=sys.stderr)",
                "sys.exit(1)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)
        caplog.set_level("WARNING")
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert any("exited 1" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_json_stdout_fails_open(self, tmp_path, caplog):
        script = _write_hook_script(
            tmp_path,
            "garbage.py",
            ["print('not valid json at all')"],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)
        caplog.set_level("WARNING")
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert any("non-JSON" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_timeout_kills_and_fails_open(self, tmp_path, caplog):
        script = _write_hook_script(
            tmp_path,
            "sleep.py",
            ["import time", "time.sleep(3)"],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(command=str(script), timeout_ms=200)
                ]
            },
        )
        runner = _runner(cfg)
        caplog.set_level("WARNING")
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.continue_ is True
        assert any("timed out" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────
# Multiple hooks per event
# ─────────────────────────────────────────────────────────────────


class TestMultipleHooks:
    @pytest.mark.asyncio
    async def test_outcomes_combine_block_wins(self, tmp_path):
        approve = _write_hook_script(
            tmp_path,
            "approve.py",
            ["import json, sys", "json.dump({'decision': 'approve'}, sys.stdout)"],
        )
        block = _write_hook_script(
            tmp_path,
            "block.py",
            ["import json, sys", "json.dump({'continue': False, 'decision': 'block', 'stop_reason': 'nope'}, sys.stdout)"],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(command=str(approve)),
                    HookConfigEntry(command=str(block)),
                ]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is True
        assert outcome.stop_reason == "nope"

    @pytest.mark.asyncio
    async def test_block_short_circuits_remaining_hooks(self, tmp_path):
        # Second hook would set a unique stop_reason marker; if it's
        # absent, the runner skipped it after the first blocked.
        block = _write_hook_script(
            tmp_path,
            "block.py",
            ["import json, sys", "json.dump({'continue': False, 'stop_reason': 'first'}, sys.stdout)"],
        )
        marker = _write_hook_script(
            tmp_path,
            "marker.py",
            ["import json, sys", "json.dump({'continue': True, 'stop_reason': 'SECOND_RAN'}, sys.stdout)"],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [
                    HookConfigEntry(command=str(block)),
                    HookConfigEntry(command=str(marker)),
                ]
            },
        )
        runner = _runner(cfg)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.stop_reason == "first"
        assert "SECOND_RAN" not in (outcome.stop_reason or "")


# ─────────────────────────────────────────────────────────────────
# Audit logging
# ─────────────────────────────────────────────────────────────────


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_audit_log_written(self, tmp_path):
        script = _write_hook_script(
            tmp_path,
            "ok.py",
            ["import json, sys", "json.dump({}, sys.stdout)"],
        )
        log_path = tmp_path / "audit" / "hooks.jsonl"
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
            audit_log_path=str(log_path),
        )
        runner = _runner(cfg)
        await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert log_path.is_file()
        line = log_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["event"] == "pre_tool_use"
        assert record["command"] == str(script)
        assert record["exit_code"] == 0
        assert record["outcome"]["continue"] is True

    @pytest.mark.asyncio
    async def test_audit_callback_invoked(self, tmp_path):
        script = _write_hook_script(
            tmp_path,
            "ok.py",
            ["import json, sys", "json.dump({}, sys.stdout)"],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner(cfg)

        records: List[Dict[str, Any]] = []

        async def _capture(record: Dict[str, Any]) -> None:
            records.append(record)

        runner.set_audit_callback(_capture)
        await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert len(records) == 1
        assert records[0]["command"] == str(script)


# ─────────────────────────────────────────────────────────────────
# Defaults sanity
# ─────────────────────────────────────────────────────────────────


def test_default_timeout_is_5_seconds():
    assert DEFAULT_TIMEOUT_MS == 5000
