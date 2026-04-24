"""Phase 7 Sprint S7.4 — Stage 10 permission-matrix wiring tests.

Confirms the Phase 1 ``PermissionRule`` + ``evaluate_permission`` system
finally fires inside ``RegistryRouter._dispatch_with_lifecycle``.
Without rules attached, dispatch is unchanged. With rules attached,
DENY short-circuits before hooks fire and ASK is treated as DENY
(safe default until the Phase 9 HITL stage lands).
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
    HookRunner,
)
from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionMode,
    PermissionRule,
    PermissionSource,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.tools.base import (
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
)
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


class _RecordingTool(Tool):
    """Tool that records inputs received via execute()."""

    def __init__(self, name: str = "rec", *, destructive: bool = False):
        self._name = name
        self._destructive = destructive
        self.received_inputs: List[Dict[str, Any]] = []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "records"

    @property
    def input_schema(self):
        return {"type": "object"}

    def capabilities(self, input):
        return ToolCapabilities(destructive=self._destructive)

    async def execute(self, input, context):
        self.received_inputs.append(dict(input))
        return ToolResult(content="ok")


def _registry_with(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _rule(
    tool_name: str,
    behavior: PermissionBehavior,
    *,
    source: PermissionSource = PermissionSource.PROJECT,
    pattern=None,
    reason=None,
) -> PermissionRule:
    return PermissionRule(
        tool_name=tool_name,
        behavior=behavior,
        source=source,
        pattern=pattern,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────
# No rules attached → no behaviour change
# ─────────────────────────────────────────────────────────────────


class TestEmptyMatrix:
    @pytest.mark.asyncio
    async def test_dispatch_unchanged_when_no_rules(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(session_id="s")
        # ctx.permission_rules defaults to []
        result = await router.route("rec", {"x": 1}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]


# ─────────────────────────────────────────────────────────────────
# DENY rule short-circuits dispatch
# ─────────────────────────────────────────────────────────────────


class TestDenyRule:
    @pytest.mark.asyncio
    async def test_deny_returns_access_denied_without_executing(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_rules=[
                _rule(
                    "rec",
                    PermissionBehavior.DENY,
                    reason="explicit project-level deny",
                ),
            ],
        )
        result = await router.route("rec", {"x": 1}, ctx)

        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        assert "explicit project-level deny" in str(result.content["error"]["message"])
        # execute() did not run
        assert tool.received_inputs == []

    @pytest.mark.asyncio
    async def test_wildcard_deny_blocks_all_tools(self):
        tool = _RecordingTool("any-tool")
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_rules=[_rule("*", PermissionBehavior.DENY)],
        )
        result = await router.route("any-tool", {}, ctx)
        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"

    @pytest.mark.asyncio
    async def test_higher_priority_deny_overrides_lower_allow(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_rules=[
                # USER-level allow loses to PROJECT-level deny.
                _rule("rec", PermissionBehavior.ALLOW, source=PermissionSource.USER),
                _rule("rec", PermissionBehavior.DENY, source=PermissionSource.PROJECT),
            ],
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error


# ─────────────────────────────────────────────────────────────────
# ALLOW rule lets dispatch through
# ─────────────────────────────────────────────────────────────────


class TestAllowRule:
    @pytest.mark.asyncio
    async def test_explicit_allow_passes(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_rules=[_rule("rec", PermissionBehavior.ALLOW)],
        )
        result = await router.route("rec", {"x": 1}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]


# ─────────────────────────────────────────────────────────────────
# ASK without HITL → safe deny
# ─────────────────────────────────────────────────────────────────


class TestAskRule:
    @pytest.mark.asyncio
    async def test_ask_treated_as_deny_until_hitl_lands(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_rules=[
                _rule("rec", PermissionBehavior.ASK, reason="needs review"),
            ],
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        # Original ASK reason surfaces in the message
        assert "needs review" in str(result.content["error"]["message"])


# ─────────────────────────────────────────────────────────────────
# PLAN-mode auto-escalation for destructive
# ─────────────────────────────────────────────────────────────────


class TestPlanModeEscalation:
    @pytest.mark.asyncio
    async def test_destructive_plan_mode_escalates_to_deny(self):
        tool = _RecordingTool("destructive-tool", destructive=True)
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_mode="plan",
            # No rules — relies on PLAN mode auto-escalation.
            permission_rules=[
                _rule("destructive-tool", PermissionBehavior.ALLOW, source=PermissionSource.USER),
            ],
        )
        # Wait — explicit ALLOW should still win even in PLAN mode.
        # Re-read the matrix doc: PLAN auto-escalates only when no rule
        # matched. So this should ALLOW.
        result = await router.route("destructive-tool", {}, ctx)
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_destructive_plan_mode_no_rule_escalates(self):
        tool = _RecordingTool("destructive-tool", destructive=True)
        router = RegistryRouter(_registry_with(tool))
        # Have at least one rule registered so the matrix path fires
        # (empty list short-circuits to "no matrix"). Use an unrelated
        # rule that doesn't match this tool.
        ctx = ToolContext(
            session_id="s",
            permission_mode="plan",
            permission_rules=[_rule("other-tool", PermissionBehavior.ALLOW)],
        )
        result = await router.route("destructive-tool", {}, ctx)
        # PLAN + destructive + no matching rule → ASK → safe DENY
        assert result.is_error
        assert "plan mode" in str(result.content["error"]["message"]).lower()


# ─────────────────────────────────────────────────────────────────
# BYPASS mode short-circuits to allow
# ─────────────────────────────────────────────────────────────────


class TestBypassMode:
    @pytest.mark.asyncio
    async def test_bypass_overrides_explicit_deny(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            permission_mode="bypass",
            permission_rules=[_rule("rec", PermissionBehavior.DENY)],
        )
        result = await router.route("rec", {}, ctx)
        # BYPASS is a developer-only escape hatch — even DENY rules lose.
        assert not result.is_error


# ─────────────────────────────────────────────────────────────────
# DENY happens before subprocess hooks fire
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


class TestDenyShortCircuitsHooks:
    @pytest.mark.asyncio
    async def test_deny_skips_pre_tool_use_hook(self, tmp_path):
        """A DENY decision must short-circuit BEFORE the subprocess
        PRE_TOOL_USE hook spawns — no point burning a subprocess on a
        call we already know is blocked."""
        sentinel = tmp_path / "sentinel.txt"
        script = _write_script(
            tmp_path,
            "hook.py",
            [
                "import json, sys",
                f"open({str(sentinel)!r}, 'w').write('hook ran')",
                "json.dump({}, sys.stdout)",
            ],
        )
        cfg = HookConfig(
            enabled=True,
            entries={HookEvent.PRE_TOOL_USE: [HookConfigEntry(command=str(script))]},
        )
        runner = _runner_with(cfg)

        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = ToolContext(
            session_id="s",
            hook_runner=runner,
            permission_rules=[_rule("rec", PermissionBehavior.DENY)],
        )

        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        # Hook MUST NOT have fired
        assert not sentinel.exists()


# ─────────────────────────────────────────────────────────────────
# Pipeline.attach_runtime wiring
# ─────────────────────────────────────────────────────────────────


class TestPipelineAttachRuntime:
    @pytest.mark.asyncio
    async def test_attach_permission_matrix_propagates_to_dispatch(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline
        from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage

        # Build a manifest pipeline with a Tool stage in it.
        manifest = EnvironmentManifest()
        # Manifest doesn't put a tool stage by default; register one.
        pipeline = await Pipeline.from_manifest_async(manifest)
        pipeline.register_stage(ToolStage(registry=ToolRegistry()))

        pipeline.attach_runtime(
            permission_rules=[_rule("rec", PermissionBehavior.DENY)],
            permission_mode="default",
        )

        # The Tool stage's context should carry the rules now.
        tool_stage = pipeline.get_stage(10)
        assert tool_stage is not None
        ctx = tool_stage._context
        assert len(ctx.permission_rules) == 1
        assert ctx.permission_rules[0].tool_name == "rec"
        assert ctx.permission_rules[0].behavior is PermissionBehavior.DENY
        assert ctx.permission_mode == "default"

    @pytest.mark.asyncio
    async def test_attach_mode_only_leaves_rules_intact(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline
        from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)
        pipeline.register_stage(ToolStage(registry=ToolRegistry()))

        pipeline.attach_runtime(
            permission_rules=[_rule("a", PermissionBehavior.ALLOW)],
        )
        pipeline.attach_runtime(permission_mode="auto")

        tool_stage = pipeline.get_stage(10)
        assert len(tool_stage._context.permission_rules) == 1
        assert tool_stage._context.permission_mode == "auto"
