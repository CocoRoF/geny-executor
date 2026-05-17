"""Tests for SubagentTypeDescriptor + SubAgentBuildContext (Phase D1)."""

from __future__ import annotations

import sys
import os
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.llm_client.credentials import CredentialBundle, ProviderCredentials
from geny_executor.stages.s12_agent.subagent_type import (
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)


# ---------------------------------------------------------------------------
# Descriptor shape — D1 new fields
# ---------------------------------------------------------------------------


def test_descriptor_has_new_fields() -> None:
    d = SubagentTypeDescriptor(agent_type="x", factory=lambda ctx: None)
    assert d.provider is None
    assert d.provider_credentials_extras == {}
    assert d.parallel is False
    assert d.max_concurrent == 1


def test_descriptor_provider_field() -> None:
    d = SubagentTypeDescriptor(
        agent_type="researcher",
        factory=lambda ctx: None,
        provider="openai",
        provider_credentials_extras={"max_budget_usd": 2.0},
        model_override="gpt-4o-mini",
        parallel=True,
        max_concurrent=4,
    )
    assert d.provider == "openai"
    assert d.provider_credentials_extras == {"max_budget_usd": 2.0}
    assert d.model_override == "gpt-4o-mini"
    assert d.parallel is True
    assert d.max_concurrent == 4


def test_descriptor_is_frozen() -> None:
    d = SubagentTypeDescriptor(agent_type="x", factory=lambda ctx: None)
    with pytest.raises(FrozenInstanceError):
        d.provider = "openai"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SubAgentBuildContext
# ---------------------------------------------------------------------------


def test_context_carries_credentials_and_descriptor() -> None:
    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-x"),
    })
    desc = SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: None, provider="anthropic")
    ctx = SubAgentBuildContext(
        parent_session_id="parent-1",
        sub_session_id="parent-1-sub-1",
        credentials=bundle,
        descriptor=desc,
    )
    assert ctx.parent_session_id == "parent-1"
    assert ctx.sub_session_id == "parent-1-sub-1"
    assert ctx.credentials is bundle
    assert ctx.descriptor is desc
    assert ctx.workspace_snapshot is None
    assert ctx.parent_state_shared == {}


def test_context_with_workspace_snapshot() -> None:
    desc = SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: None)
    ctx = SubAgentBuildContext(
        parent_session_id="p",
        sub_session_id="s",
        credentials=None,
        descriptor=desc,
        workspace_snapshot={"cwd": "/tmp/wd", "branch": "main"},
        parent_state_shared={"key": "value"},
    )
    assert ctx.workspace_snapshot == {"cwd": "/tmp/wd", "branch": "main"}
    assert ctx.parent_state_shared == {"key": "value"}


def test_context_is_frozen() -> None:
    desc = SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: None)
    ctx = SubAgentBuildContext(
        parent_session_id="p",
        sub_session_id="s",
        credentials=None,
        descriptor=desc,
    )
    with pytest.raises(FrozenInstanceError):
        ctx.parent_session_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory receives the context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_receives_build_context() -> None:
    """A factory accepting ``ctx`` actually sees the descriptor + parent
    session details."""
    received: dict = {}

    class _FakePipeline:
        async def run(self, task, sub_state):
            return type("R", (), {"success": True, "text": f"ok:{task}", "error": None})()

    def factory(ctx: SubAgentBuildContext):
        received["ctx"] = ctx
        return _FakePipeline()

    bundle = CredentialBundle(by_provider={"openai": ProviderCredentials(api_key="o")})
    desc = SubagentTypeDescriptor(
        agent_type="reviewer", factory=factory, provider="openai",
    )
    reg = SubagentTypeRegistry().register(desc)
    orch = SubagentTypeOrchestrator(reg)

    from geny_executor.core.state import PipelineState

    state = PipelineState(session_id="sess-7")
    state.credentials = bundle
    state.shared["workspace_snapshot"] = {"cwd": "/tmp/x"}
    state.delegate_requests = [{"agent_type": "reviewer", "task": "review"}]

    result = await orch.orchestrate(state)
    assert result.delegated is True

    ctx: SubAgentBuildContext = received["ctx"]
    assert ctx.parent_session_id == "sess-7"
    assert ctx.sub_session_id.startswith("sess-7-reviewer-")
    assert ctx.descriptor.provider == "openai"
    assert ctx.credentials is bundle
    assert ctx.workspace_snapshot == {"cwd": "/tmp/x"}


@pytest.mark.asyncio
async def test_legacy_zero_arg_factory_still_works() -> None:
    """Pre-D1 factories (no ctx parameter) keep working via the
    TypeError fallback in _resolve_pipeline."""

    class _FakePipeline:
        async def run(self, task, sub_state):
            return type("R", (), {"success": True, "text": "legacy", "error": None})()

    def factory():
        return _FakePipeline()

    desc = SubagentTypeDescriptor(agent_type="legacy", factory=factory)
    reg = SubagentTypeRegistry().register(desc)
    orch = SubagentTypeOrchestrator(reg)

    from geny_executor.core.state import PipelineState

    state = PipelineState(session_id="sess-l")
    state.delegate_requests = [{"agent_type": "legacy", "task": "noop"}]
    result = await orch.orchestrate(state)
    assert result.delegated is True
    assert result.sub_results[0]["success"] is True
    assert result.sub_results[0]["text"] == "legacy"


# ---------------------------------------------------------------------------
# Metadata exposes new fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_metadata_carries_provider_and_parallel_flags() -> None:
    class _FakePipeline:
        async def run(self, task, sub_state):
            return type("R", (), {"success": True, "text": "x", "error": None})()

    desc = SubagentTypeDescriptor(
        agent_type="r", factory=lambda ctx: _FakePipeline(),
        provider="claude_code_cli", parallel=True, max_concurrent=3,
        model_override="opus",
    )
    reg = SubagentTypeRegistry().register(desc)
    orch = SubagentTypeOrchestrator(reg)

    from geny_executor.core.state import PipelineState
    state = PipelineState(session_id="s")
    state.delegate_requests = [{"agent_type": "r", "task": "go"}]
    result = await orch.orchestrate(state)
    meta = result.sub_results[0]["subagent_metadata"]
    assert meta["provider"] == "claude_code_cli"
    assert meta["model_override"] == "opus"
    assert meta["parallel"] is True
    assert meta["max_concurrent"] == 3
