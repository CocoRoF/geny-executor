"""Phase 5 Week 9 — Stage 10 subprocess hook wiring tests.

Confirms ``RegistryRouter`` fires ``PRE_TOOL_USE`` / ``POST_TOOL_USE`` /
``POST_TOOL_FAILURE`` through ``ToolContext.hook_runner`` when one is
bound, honours block / modify_input outcomes, and stays a no-op when
no runner is attached.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from geny_executor.hooks import (
    HookConfig,
    HookConfigEntry,
    HookEvent,
    HookEventPayload,
    HookOutcome,
    HookRunner,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _write_script(tmp_path: Path, name: str, body_lines: List[str]) -> Path:
    path = tmp_path / name
    path.write_text("#!{}\n".format(sys.executable) + "\n".join(body_lines) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _runner_with(config: HookConfig) -> HookRunner:
    env = dict(os.environ)
    env["GENY_ALLOW_HOOKS"] = "1"
    return HookRunner(config, env=env)


class _RecordingTool(Tool):
    def __init__(self, name: str = "rec"):
        self._name = name
        self.received_inputs: List[Dict[str, Any]] = []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "records inputs"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def execute(self, input, context):
        self.received_inputs.append(dict(input))
        return ToolResult(content="ok")


class _ErrorReturnTool(_RecordingTool):
    """Tool that returns a soft-error result without raising."""

    async def execute(self, input, context):
        return ToolResult(content="oops", is_error=True)


class _CrashingTool(_RecordingTool):
    async def execute(self, input, context):
        raise RuntimeError("boom")


class _StrictSchemaTool(Tool):
    @property
    def name(self):
        return "strict"

    @property
    def description(self):
        return "n required"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        }

    async def execute(self, input, context):
        return ToolResult(content=f"got {input.get('n')}")


def _registry_with(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ─────────────────────────────────────────────────────────────────
# No runner attached — nothing changes
# ─────────────────────────────────────────────────────────────────


class TestRouterWithoutHookRunner:
    @pytest.mark.asyncio
    async def test_dispatch_unchanged_when_runner_missing(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s")
        assert ctx.hook_runner is None
        result = await router.route("rec", {"x": 1}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]


# ─────────────────────────────────────────────────────────────────
# PRE_TOOL_USE — block
# ─────────────────────────────────────────────────────────────────


class TestPreHookBlock:
    @pytest.mark.asyncio
    async def test_block_outcome_short_circuits_execute(self, tmp_path):
        script = _write_script(
            tmp_path,
            "block.py",
            [
                "import json, sys",
                "json.dump({'continue': False, 'decision': 'block', 'stop_reason': 'no go'}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        result = await router.route("rec", {"x": 1}, ctx)

        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        assert "no go" in str(result.content["error"]["message"])
        # execute() must NOT have run
        assert tool.received_inputs == []


# ─────────────────────────────────────────────────────────────────
# PRE_TOOL_USE — modify_input
# ─────────────────────────────────────────────────────────────────


class TestPreHookModifyInput:
    @pytest.mark.asyncio
    async def test_modified_input_replaces_payload(self, tmp_path):
        script = _write_script(
            tmp_path,
            "rewrite.py",
            [
                "import json, sys",
                "json.dump({'modified_input': {'x': 99, 'y': 'hi'}}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        await router.route("rec", {"x": 1}, ctx)

        # Tool saw the rewritten input, not the original
        assert tool.received_inputs == [{"x": 99, "y": "hi"}]

    @pytest.mark.asyncio
    async def test_rewrite_revalidates_against_input_schema(self, tmp_path):
        """A misbehaving hook can't bypass the tool's input schema —
        the rewrite must still match before execute() runs."""
        # Hook rewrites to a payload that's missing required ``n``.
        script = _write_script(
            tmp_path,
            "bad_rewrite.py",
            [
                "import json, sys",
                "json.dump({'modified_input': {'wrong_key': 1}}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_StrictSchemaTool()))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        result = await router.route("strict", {"n": 1}, ctx)

        assert result.is_error
        assert result.content["error"]["code"] == "invalid_input"


# ─────────────────────────────────────────────────────────────────
# POST_TOOL_USE / POST_TOOL_FAILURE
# ─────────────────────────────────────────────────────────────────


class TestPostHooks:
    @pytest.mark.asyncio
    async def test_post_tool_use_fires_on_success(self, tmp_path):
        sentinel = tmp_path / "sentinel.txt"
        script = _write_script(
            tmp_path,
            "post_ok.py",
            [
                "import json, sys",
                f"open({str(sentinel)!r}, 'w').write('post_tool_use ran')",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.POST_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_RecordingTool()))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        result = await router.route("rec", {}, ctx)

        assert not result.is_error
        assert sentinel.read_text() == "post_tool_use ran"

    @pytest.mark.asyncio
    async def test_post_tool_failure_fires_on_soft_error(self, tmp_path):
        sentinel = tmp_path / "fail_sentinel.txt"
        script = _write_script(
            tmp_path,
            "post_fail.py",
            [
                "import json, sys",
                f"open({str(sentinel)!r}, 'w').write('failure ran')",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.POST_TOOL_FAILURE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_ErrorReturnTool("rec")))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        await router.route("rec", {}, ctx)

        assert sentinel.read_text() == "failure ran"

    @pytest.mark.asyncio
    async def test_post_tool_failure_fires_on_unexpected_exception(self, tmp_path):
        sentinel = tmp_path / "crash_sentinel.txt"
        script = _write_script(
            tmp_path,
            "post_crash.py",
            [
                "import json, sys",
                f"open({str(sentinel)!r}, 'w').write('crash ran')",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.POST_TOOL_FAILURE: [HookConfigEntry(command=str(script))]
            },
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_CrashingTool("rec")))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        result = await router.route("rec", {}, ctx)

        assert result.is_error
        assert result.content["error"]["code"] == "tool_crashed"
        assert sentinel.read_text() == "crash ran"

    @pytest.mark.asyncio
    async def test_post_tool_use_does_not_fire_for_failure(self, tmp_path):
        """Soft errors should NOT trigger POST_TOOL_USE — only
        POST_TOOL_FAILURE."""
        sentinel = tmp_path / "should_not_exist.txt"
        script = _write_script(
            tmp_path,
            "should_not_run.py",
            [
                "import json, sys",
                f"open({str(sentinel)!r}, 'w').write('!!')",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.POST_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_ErrorReturnTool("rec")))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        await router.route("rec", {}, ctx)

        assert not sentinel.exists()


# ─────────────────────────────────────────────────────────────────
# Payload completeness — what hooks see on stdin
# ─────────────────────────────────────────────────────────────────


class TestHookPayloadShape:
    @pytest.mark.asyncio
    async def test_payload_includes_tool_name_and_input(self, tmp_path):
        capture = tmp_path / "captured.json"
        script = _write_script(
            tmp_path,
            "capture.py",
            [
                "import json, sys",
                "data = json.loads(sys.stdin.read())",
                f"open({str(capture)!r}, 'w').write(json.dumps(data))",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_RecordingTool()))
        ctx = ToolContext(
            session_id="sess-xyz",
            hook_runner=runner,
            permission_mode="auto",
            stage_order=10,
            stage_name="tool",
        )

        await router.route("rec", {"command": "echo hi"}, ctx)

        payload = __import__("json").loads(capture.read_text())
        assert payload["event"] == "pre_tool_use"
        assert payload["session_id"] == "sess-xyz"
        assert payload["tool_name"] == "rec"
        assert payload["tool_input"] == {"command": "echo hi"}
        assert payload["permission_mode"] == "auto"
        assert payload["stage_order"] == 10
        assert payload["stage_name"] == "tool"

    @pytest.mark.asyncio
    async def test_post_payload_includes_output_preview(self, tmp_path):
        capture = tmp_path / "post_payload.json"
        script = _write_script(
            tmp_path,
            "capture_post.py",
            [
                "import json, sys",
                "data = json.loads(sys.stdin.read())",
                f"open({str(capture)!r}, 'w').write(json.dumps(data))",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.POST_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)
        router = RegistryRouter(_registry_with(_RecordingTool()))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        await router.route("rec", {}, ctx)

        payload = __import__("json").loads(capture.read_text())
        assert payload["event"] == "post_tool_use"
        # _RecordingTool returns ToolResult(content="ok")
        assert payload["tool_output"] == "ok"


# ─────────────────────────────────────────────────────────────────
# Integration with HookRunner.passthrough at runtime
# ─────────────────────────────────────────────────────────────────


class TestRunnerPassthroughBehaviour:
    @pytest.mark.asyncio
    async def test_disabled_runner_is_a_no_op_for_dispatch(self):
        """A runner with ``enabled=False`` returns passthrough on every
        fire — the router behaves as if no runner were attached."""
        runner = HookRunner(HookConfig.disabled(), env={"GENY_ALLOW_HOOKS": "1"})
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        result = await router.route("rec", {"x": 1}, ctx)

        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_runner_without_env_opt_in_is_a_no_op(self):
        cfg = HookConfig(
            enabled=True,
            entries={
                HookEvent.PRE_TOOL_USE: [HookConfigEntry(command="/bin/false")]
            },
        )
        runner = HookRunner(cfg, env={})  # no GENY_ALLOW_HOOKS
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s", hook_runner=runner)

        # /bin/false returns 1 — would normally just log fail-open passthrough.
        # But since runner is gated off by env, no subprocess is spawned at all.
        result = await router.route("rec", {"x": 1}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]
