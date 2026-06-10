"""Tests for credential + subagent_registry propagation through the
pipeline into sub-agent factories (Phase D3)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.core.state import PipelineState
from geny_executor.llm_client.credentials import (
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.stages.s12_agent.subagent_type import (
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_manifest_with_api_stage(provider: str = "anthropic") -> EnvironmentManifest:
    """Build a manifest with just enough stages to exercise pipeline
    bring-up + Stage 12 wiring."""
    m = EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_d3", name="d3-test"),
        model={},
        pipeline={},
        stages=[],
        tools=ToolsSnapshot(),
    )
    m.set_stage_entries([
        StageManifestEntry(order=1, name="input"),
        StageManifestEntry(
            order=6, name="api",
            config={"provider": provider},
            strategies={"retry": "exponential_backoff", "router": "passthrough"},
        ),
        StageManifestEntry(order=9, name="parse"),
        StageManifestEntry(order=21, name="yield"),
    ])
    return m


# ---------------------------------------------------------------------------
# Pipeline slots
# ---------------------------------------------------------------------------


def test_pipeline_has_subagent_registry_slot() -> None:
    p = Pipeline()
    assert p._subagent_registry is None


def test_attach_runtime_accepts_subagent_registry() -> None:
    p = Pipeline()
    reg = SubagentTypeRegistry()
    p.attach_runtime(subagent_registry=reg)
    assert p._subagent_registry is reg


def test_from_manifest_accepts_subagent_registry() -> None:
    m = _make_minimal_manifest_with_api_stage()
    reg = SubagentTypeRegistry()
    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-x"),
    })
    p = Pipeline.from_manifest(m, credentials=bundle, subagent_registry=reg, strict=False)
    assert p._subagent_registry is reg


# ---------------------------------------------------------------------------
# State propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_state_propagates_credentials_and_subagent_registry() -> None:
    m = _make_minimal_manifest_with_api_stage()
    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-1"),
        "openai":    ProviderCredentials(api_key="sk-2"),
    })
    reg = SubagentTypeRegistry()
    p = Pipeline.from_manifest(m, credentials=bundle, subagent_registry=reg, strict=False)

    state = p._init_state(None)
    assert state.credentials is bundle
    assert state.subagent_registry is reg


# ---------------------------------------------------------------------------
# Sub-agent factory sees parent's credentials through SubAgentBuildContext
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_factory_receives_parent_credentials() -> None:
    """The sub-agent factory must see the parent's CredentialBundle so
    the sub-pipeline can authenticate against any provider — this is the
    backbone of multi-provider sub-agents."""

    received: dict = {}

    class _FakeSubPipeline:
        async def run(self, task, sub_state):
            return type("R", (), {"success": True, "text": f"sub:{task}", "error": None})()

    def factory(ctx: SubAgentBuildContext):
        received["ctx"] = ctx
        return _FakeSubPipeline()

    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="parent-anth"),
        "openai":    ProviderCredentials(api_key="parent-oai"),
    })
    reg = SubagentTypeRegistry().register(SubagentTypeDescriptor(
        agent_type="researcher",
        factory=factory,
        provider="openai",   # sub uses a different provider than parent stage 6
    ))

    orch = SubagentTypeOrchestrator(reg)

    state = PipelineState(session_id="parent-sess")
    state.credentials = bundle
    state.subagent_registry = reg
    state.delegate_requests = [{"agent_type": "researcher", "task": "look up X"}]

    result = await orch.orchestrate(state)
    assert result.delegated is True

    ctx = received["ctx"]
    # Same bundle reference — no copy.
    assert ctx.credentials is bundle
    # The factory can pick up openai creds via the descriptor's provider hint.
    assert ctx.descriptor.provider == "openai"
    assert ctx.credentials.require("openai").api_key == "parent-oai"
    assert ctx.credentials.require("anthropic").api_key == "parent-anth"


@pytest.mark.asyncio
async def test_subagent_can_build_sub_pipeline_from_manifest_with_inherited_credentials() -> None:
    """End-to-end: a factory uses the inherited bundle to build a real
    sub-pipeline via Pipeline.from_manifest. The sub-pipeline's
    state.credentials matches the parent's."""

    captured: dict = {}

    async def factory(ctx: SubAgentBuildContext):
        # Build a real sub-pipeline using the inherited bundle.
        sub_manifest = _make_minimal_manifest_with_api_stage(
            provider=ctx.descriptor.provider or "anthropic",
        )
        sub_pipeline = Pipeline.from_manifest(
            sub_manifest, credentials=ctx.credentials, strict=False,
        )
        captured["sub_pipeline"] = sub_pipeline
        # Run init_state synchronously to populate sub state.credentials
        # without driving the full run loop.
        sub_state = sub_pipeline._init_state(None)
        captured["sub_state"] = sub_state

        class _Stub:
            async def run(self, task, sub_state_arg):
                return type("R", (), {"success": True, "text": "ok", "error": None})()

        return _Stub()

    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="A"),
        "openai":    ProviderCredentials(api_key="O"),
    })
    reg = SubagentTypeRegistry().register(SubagentTypeDescriptor(
        agent_type="r", factory=factory, provider="openai",
    ))
    orch = SubagentTypeOrchestrator(reg)

    state = PipelineState(session_id="p")
    state.credentials = bundle
    state.delegate_requests = [{"agent_type": "r", "task": "go"}]
    await orch.orchestrate(state)

    sub_state = captured["sub_state"]
    # Parent and sub share the same bundle reference.
    assert sub_state.credentials is bundle
    # OpenAI key is reachable in the sub-pipeline.
    assert sub_state.credentials.require("openai").api_key == "O"


# ---------------------------------------------------------------------------
# attach_runtime rewires the agent stage's orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_runtime_wires_subagent_orchestrator_when_agent_stage_present() -> None:
    """If the manifest registered the agent stage but the subagent_registry
    arrived late (via attach_runtime), the agent stage's orchestrator
    should be rebuilt to consume the registry."""
    from geny_executor.stages.s12_agent.artifact.default.stage import AgentStage
    from geny_executor.stages.s12_agent.subagent_type import SubagentTypeOrchestrator

    p = Pipeline()
    p.register_stage(AgentStage())
    # Initially the orchestrator is a SingleAgentOrchestrator (default).
    agent = next(s for s in p.stages if s.name == "agent")
    assert agent._orchestrator.name == "single_agent"

    reg = SubagentTypeRegistry()
    p.attach_runtime(subagent_registry=reg)
    agent = next(s for s in p.stages if s.name == "agent")
    assert isinstance(agent._orchestrator, SubagentTypeOrchestrator)
    assert agent._orchestrator.registry is reg


@pytest.mark.asyncio
async def test_attach_runtime_no_op_when_no_agent_stage() -> None:
    """If the pipeline has no agent stage, attach_runtime(subagent_registry=)
    still stores the registry on the pipeline but does not raise."""
    p = Pipeline()
    reg = SubagentTypeRegistry()
    p.attach_runtime(subagent_registry=reg)
    assert p._subagent_registry is reg  # stored
    # No agent stage → no crash
