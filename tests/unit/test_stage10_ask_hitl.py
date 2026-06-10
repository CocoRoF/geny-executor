"""2.2.0 — Stage 10 ASK→HITL routing + posture-at-dispatch tests (audit §1-5).

Before this release the permission matrix could *return* ASK but
nothing ever *produced* an approval request — ASK was a hard deny in
disguise. These tests pin the new plumbing:

* ASK + bound requester (Stage 15 ``Requester`` contract) → request
  emitted, verdict honoured (APPROVE proceeds, REJECT/CANCEL/None deny).
* ASK + ``PERMISSION_REQUEST`` in-process hook → machine policy can
  answer without a human and without ``GENY_ALLOW_HOOKS`` (the GAPT
  scenario — gate split + ASK routing working together).
* ASK with neither → deny (an ASK rule is an explicit demand for
  judgement; the ambient allow posture must not auto-approve it).
* ``permission_default_posture`` honoured at dispatch, including with
  ZERO rules bound (deny posture + no rules = allowlist-only).
* ``PERMISSION_DENIED`` / ``PERMISSION_REQUEST`` hook events fire from
  the dispatch path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geny_executor.hooks import HookConfig, HookEvent, HookOutcome, HookRunner
from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionPosture,
    PermissionRule,
    PermissionSource,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.stages.s15_hitl.interface import (
    HITL_HISTORY_KEY,
    HITL_LAST_DECISION_KEY,
)
from geny_executor.stages.s15_hitl.types import HITLDecision, HITLRequest
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


class _RecordingTool(Tool):
    def __init__(self, name: str = "rec"):
        self._name = name
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
        return ToolCapabilities()

    async def execute(self, input, context):
        self.received_inputs.append(dict(input))
        return ToolResult(content="ok")


class _StubRequester:
    """Duck-typed Stage 15 Requester — records requests, returns a verdict."""

    name = "stub"

    def __init__(self, verdict: Optional[Any] = HITLDecision.APPROVE, *, raise_exc: bool = False):
        self.verdict = verdict
        self.raise_exc = raise_exc
        self.requests: List[HITLRequest] = []

    async def request(self, request: HITLRequest, state: Any) -> Optional[HITLDecision]:
        self.requests.append(request)
        if self.raise_exc:
            raise RuntimeError("requester exploded")
        return self.verdict


class _FakeState:
    """Minimal PipelineState stand-in: shared dict + event recorder."""

    def __init__(self):
        self.shared: Dict[str, Any] = {}
        self.events: List[Any] = []

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append((event_type, data))


def _registry_with(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _ask_rule(tool: str = "rec", reason: str = "needs review") -> PermissionRule:
    return PermissionRule(
        tool_name=tool,
        behavior=PermissionBehavior.ASK,
        source=PermissionSource.PROJECT,
        reason=reason,
    )


def _ctx(**kwargs) -> ToolContext:
    """ToolContext with posture / requester set as dynamic attributes,
    mirroring how Stage 10 propagates them from the stage context."""
    requester = kwargs.pop("hitl_requester", None)
    posture = kwargs.pop("permission_default_posture", None)
    ctx = ToolContext(session_id="s", **kwargs)
    if requester is not None:
        ctx.hitl_requester = requester
    if posture is not None:
        ctx.permission_default_posture = posture
    return ctx


# ─────────────────────────────────────────────────────────────────
# ASK + requester
# ─────────────────────────────────────────────────────────────────


class TestAskWithRequester:
    @pytest.mark.asyncio
    async def test_approve_executes_tool_and_emits_request(self):
        tool = _RecordingTool()
        requester = _StubRequester(HITLDecision.APPROVE)
        state = _FakeState()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[_ask_rule()],
            hitl_requester=requester,
            state_view=state,
        )
        result = await router.route("rec", {"x": 1}, ctx)

        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]
        # Request was emitted with the permission context attached.
        assert len(requester.requests) == 1
        req = requester.requests[0]
        assert req.payload["source"] == "permission_matrix"
        assert req.payload["tool_name"] == "rec"
        assert "needs review" in req.reason
        # Stage 15 audit contract mirrored.
        assert state.shared[HITL_LAST_DECISION_KEY] == "approve"
        assert len(state.shared[HITL_HISTORY_KEY]) == 1
        event_types = [t for t, _ in state.events]
        assert "hitl.request" in event_types
        assert "hitl.decision" in event_types

    @pytest.mark.asyncio
    async def test_reject_denies_without_executing(self):
        tool = _RecordingTool()
        requester = _StubRequester(HITLDecision.REJECT)
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hitl_requester=requester)
        result = await router.route("rec", {}, ctx)

        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        assert tool.received_inputs == []
        assert len(requester.requests) == 1

    @pytest.mark.asyncio
    async def test_cancel_denies(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[_ask_rule()],
            hitl_requester=_StubRequester(HITLDecision.CANCEL),
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert tool.received_inputs == []

    @pytest.mark.asyncio
    async def test_no_decision_denies(self):
        # None from a requester means "no verdict" — the only safe
        # reading at the dispatch site is no.
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hitl_requester=_StubRequester(None))
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert "no decision" in str(result.content["error"]["message"])

    @pytest.mark.asyncio
    async def test_requester_exception_denies_not_crashes(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[_ask_rule()],
            hitl_requester=_StubRequester(raise_exc=True),
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert tool.received_inputs == []

    @pytest.mark.asyncio
    async def test_string_verdict_coerced(self):
        # Hosts wiring quick callbacks may return plain strings; the
        # router runs them through coerce_decision.
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hitl_requester=_StubRequester("approve"))
        result = await router.route("rec", {}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{}]


# ─────────────────────────────────────────────────────────────────
# ASK without requester → safe deny (documented fallback)
# ─────────────────────────────────────────────────────────────────


class TestAskWithoutRequester:
    @pytest.mark.asyncio
    async def test_ask_without_requester_denies(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()])
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        assert "needs review" in str(result.content["error"]["message"])
        assert tool.received_inputs == []

    @pytest.mark.asyncio
    async def test_ask_without_requester_denies_even_under_allow_posture(self):
        # An ASK rule is an explicit demand for judgement — the ambient
        # allow posture must NOT auto-approve it.
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[_ask_rule()],
            permission_default_posture=PermissionPosture.ALLOW,
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error


# ─────────────────────────────────────────────────────────────────
# PERMISSION_REQUEST hook can answer the ASK (machine policy)
# ─────────────────────────────────────────────────────────────────


class TestPermissionRequestHook:
    @pytest.mark.asyncio
    async def test_in_process_approve_without_env_var(self):
        # The GAPT scenario end-to-end: in-process policy engine answers
        # the ASK; no GENY_ALLOW_HOOKS anywhere.
        tool = _RecordingTool()
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        seen: List[Any] = []

        async def policy(payload):
            seen.append(payload)
            return HookOutcome.approve("machine policy clearance")

        runner.register_in_process(HookEvent.PERMISSION_REQUEST, policy)
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hook_runner=runner)
        result = await router.route("rec", {"x": 2}, ctx)

        assert not result.is_error
        assert tool.received_inputs == [{"x": 2}]
        assert len(seen) == 1
        assert seen[0].event is HookEvent.PERMISSION_REQUEST
        assert seen[0].details["reason"]

    @pytest.mark.asyncio
    async def test_in_process_block_denies(self):
        tool = _RecordingTool()
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        runner.register_in_process(
            HookEvent.PERMISSION_REQUEST,
            lambda p: HookOutcome.block("policy says no"),
        )
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hook_runner=runner)
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert "policy says no" in str(result.content["error"]["message"])

    @pytest.mark.asyncio
    async def test_hook_passthrough_falls_through_to_requester(self):
        tool = _RecordingTool()
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        runner.register_in_process(HookEvent.PERMISSION_REQUEST, lambda p: None)
        requester = _StubRequester(HITLDecision.APPROVE)
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[_ask_rule()],
            hook_runner=runner,
            hitl_requester=requester,
        )
        result = await router.route("rec", {}, ctx)
        assert not result.is_error
        assert len(requester.requests) == 1


# ─────────────────────────────────────────────────────────────────
# PERMISSION_DENIED observability
# ─────────────────────────────────────────────────────────────────


class TestPermissionDeniedEvent:
    @pytest.mark.asyncio
    async def test_deny_rule_fires_permission_denied(self):
        tool = _RecordingTool()
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        seen: List[Any] = []
        runner.register_in_process(
            HookEvent.PERMISSION_DENIED, lambda p: seen.append(p) or None
        )
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[
                PermissionRule(
                    tool_name="rec",
                    behavior=PermissionBehavior.DENY,
                    source=PermissionSource.PROJECT,
                    reason="explicit deny",
                )
            ],
            hook_runner=runner,
        )
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert len(seen) == 1
        assert "explicit deny" in seen[0].details["reason"]

    @pytest.mark.asyncio
    async def test_ask_fallback_deny_fires_permission_denied(self):
        tool = _RecordingTool()
        runner = HookRunner(HookConfig(enabled=True, entries={}), env={})
        seen: List[Any] = []
        runner.register_in_process(
            HookEvent.PERMISSION_DENIED, lambda p: seen.append(p) or None
        )
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_rules=[_ask_rule()], hook_runner=runner)
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert len(seen) == 1


# ─────────────────────────────────────────────────────────────────
# Posture at the dispatch site
# ─────────────────────────────────────────────────────────────────


class TestDispatchPosture:
    @pytest.mark.asyncio
    async def test_deny_posture_with_zero_rules_denies(self):
        # Historically an empty rule list skipped the matrix entirely —
        # which would have made the deny posture a decoy setting.
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_default_posture=PermissionPosture.DENY)
        result = await router.route("rec", {}, ctx)
        assert result.is_error
        assert result.content["error"]["code"] == "access_denied"
        assert tool.received_inputs == []

    @pytest.mark.asyncio
    async def test_deny_posture_string_value_accepted(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(permission_default_posture="deny")
        result = await router.route("rec", {}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_deny_posture_with_explicit_allow_rule_executes(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        ctx = _ctx(
            permission_rules=[
                PermissionRule(
                    tool_name="rec",
                    behavior=PermissionBehavior.ALLOW,
                    source=PermissionSource.PROJECT,
                )
            ],
            permission_default_posture=PermissionPosture.DENY,
        )
        result = await router.route("rec", {"x": 1}, ctx)
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_no_posture_no_rules_unchanged_back_compat(self):
        tool = _RecordingTool()
        router = RegistryRouter(_registry_with(tool))
        result = await router.route("rec", {"x": 1}, ToolContext(session_id="s"))
        assert not result.is_error
        assert tool.received_inputs == [{"x": 1}]


# ─────────────────────────────────────────────────────────────────
# Stage-level propagation (dynamic attributes survive ctx rebuild)
# ─────────────────────────────────────────────────────────────────


class _CapturingExecutor:
    """Duck-typed ToolExecutor that captures the dispatch context."""

    name = "capturing"
    description = "captures ctx"

    def __init__(self):
        self.contexts: List[ToolContext] = []

    async def execute_all(self, tool_calls, router, context, *, on_event=None):
        self.contexts.append(context)
        return [
            {"tool_use_id": tc.get("tool_use_id", ""), "is_error": False}
            for tc in tool_calls
        ]


class TestStagePropagation:
    @pytest.mark.asyncio
    async def test_stage_propagates_posture_and_requester(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage

        requester = _StubRequester()
        stage_ctx = ToolContext(session_id="s")
        stage_ctx.permission_default_posture = PermissionPosture.DENY
        stage_ctx.hitl_requester = requester

        executor = _CapturingExecutor()
        stage = ToolStage(
            registry=_registry_with(_RecordingTool()),
            executor=executor,
            context=stage_ctx,
        )
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "rec", "tool_input": {}}
        ]
        await stage.execute(None, state)

        assert len(executor.contexts) == 1
        dispatched = executor.contexts[0]
        assert dispatched.permission_default_posture is PermissionPosture.DENY
        assert dispatched.hitl_requester is requester

    @pytest.mark.asyncio
    async def test_stage_omits_attrs_when_unset(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage

        executor = _CapturingExecutor()
        stage = ToolStage(registry=_registry_with(_RecordingTool()), executor=executor)
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "rec", "tool_input": {}}
        ]
        await stage.execute(None, state)

        dispatched = executor.contexts[0]
        assert getattr(dispatched, "permission_default_posture", None) is None
        assert getattr(dispatched, "hitl_requester", None) is None
