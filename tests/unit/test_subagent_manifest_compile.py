"""2.2.0 Wave 3 — manifest ``subagents`` → SubagentTypeDescriptor compilation.

Covers the library half of "sub-agent environments are first-class"
(audit §1-1): ``compile_subagent_descriptors`` + the default
``ManifestSubagentPipelineFactory``, the single-home provider
resolution order (``resolve_subagent_provider``, audit §2.8), the
``Pipeline.from_manifest`` merge semantics (explicit registry wins),
the typed ``SubAgentBuildContext.parent_provider`` field, and the
unified ``run_subagent`` delegation surface.
"""

from __future__ import annotations

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.state import PipelineState
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.stages.s12_agent.subagent_type import (
    ManifestSubagentPipelineFactory,
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
    compile_subagent_descriptors,
    resolve_subagent_provider,
)

from tests._fixtures.manifest_entries import required_stage_entries


# ── Helpers ──────────────────────────────────────────────────


def _manifest(*, subagents=None, provider: str = "anthropic") -> EnvironmentManifest:
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_sc", name="subagent-compile"),
        stages=required_stage_entries(provider),
        tools=ToolsSnapshot(),
        subagents=list(subagents or []),
    )


def _bundle(*providers: str) -> CredentialBundle:
    return CredentialBundle(
        by_provider={p: ProviderCredentials(api_key=f"sk-{p}") for p in providers}
    )


def _ctx(descriptor, *, credentials=None, shared=None, parent_provider=None):
    return SubAgentBuildContext(
        parent_session_id="parent",
        sub_session_id="parent-sub-1",
        credentials=credentials,
        descriptor=descriptor,
        parent_state_shared=dict(shared or {}),
        parent_provider=parent_provider,
    )


def _stage6_provider(pipeline) -> str:
    api = next(s for s in pipeline.stages if s.name == "api")
    return api.get_config()["provider"]


class _FakeResult:
    def __init__(self, *, text="sub-ok", success=True, error=None):
        self.text = text
        self.success = success
        self.error = error


class _FakeSubPipeline:
    def __init__(self, result=None):
        self._result = result or _FakeResult()
        self.runs = []

    async def run(self, task, state):
        self.runs.append((task, state))
        return self._result


# ── resolve_subagent_provider — the single resolution home ──


class TestResolveProvider:
    def test_descriptor_provider_wins(self):
        d = SubagentTypeDescriptor(agent_type="t", factory=lambda c: None, provider="openai")
        ctx = _ctx(d, parent_provider="anthropic", credentials=_bundle("google"))
        assert resolve_subagent_provider(ctx) == "openai"

    def test_parent_provider_field_beats_bundle(self):
        d = SubagentTypeDescriptor(agent_type="t", factory=lambda c: None)
        ctx = _ctx(d, parent_provider="anthropic", credentials=_bundle("openai"))
        assert resolve_subagent_provider(ctx) == "anthropic"

    def test_legacy_shared_key_honoured_without_typed_field(self):
        d = SubagentTypeDescriptor(agent_type="t", factory=lambda c: None)
        ctx = _ctx(d, shared={SharedKeys.PRIMARY_PROVIDER: "google"})
        assert resolve_subagent_provider(ctx) == "google"

    def test_bundle_preferred_provider_is_last_resort(self):
        d = SubagentTypeDescriptor(agent_type="t", factory=lambda c: None)
        ctx = _ctx(d, credentials=_bundle("openai"))
        assert resolve_subagent_provider(ctx) == "openai"

    def test_nothing_resolves_returns_none(self):
        d = SubagentTypeDescriptor(agent_type="t", factory=lambda c: None)
        assert resolve_subagent_provider(_ctx(d)) is None


# ── compile_subagent_descriptors ─────────────────────────────


class TestCompile:
    def test_entry_fields_land_on_descriptor(self):
        entry = {
            "agent_type": "researcher",
            "description": "Looks things up",
            "provider": "openai",
            "model_override": "gpt-5",
            "allowed_tools": ["Read", "WebSearch"],
            "env_id": "env_stored",
        }
        (d,) = compile_subagent_descriptors([entry])
        assert d.agent_type == "researcher"
        assert d.description == "Looks things up"
        assert d.provider == "openai"
        assert d.model_override == "gpt-5"
        assert d.allowed_tools == ("Read", "WebSearch")
        assert d.env_id == "env_stored"
        assert isinstance(d.factory, ManifestSubagentPipelineFactory)
        assert d.factory.env_id == "env_stored"

    def test_compiled_descriptors_register(self):
        entries = [{"agent_type": "a"}, {"agent_type": "b"}]
        registry = SubagentTypeRegistry()
        for d in compile_subagent_descriptors(entries):
            registry.register(d)
        assert registry.list_types() == ["a", "b"]

    def test_malformed_entries_skipped_leniently(self):
        entries = ["not-a-dict", {"description": "nameless"}, {"agent_type": "ok"}]
        compiled = compile_subagent_descriptors(entries)
        assert [d.agent_type for d in compiled] == ["ok"]


# ── Default factory builds ───────────────────────────────────


class TestDefaultFactoryBuilds:
    @pytest.mark.asyncio
    async def test_no_manifest_inherits_parent_provider(self):
        """Audit §2.8 end-to-end: descriptor.provider=None must NOT fall
        through to a host-global heuristic — the parent's resolved
        provider (PRIMARY_PROVIDER producer) drives the sub-build."""
        (d,) = compile_subagent_descriptors([{"agent_type": "worker"}])
        ctx = _ctx(d, credentials=_bundle("anthropic"), parent_provider="anthropic")
        sub = await d.factory(ctx)
        try:
            assert _stage6_provider(sub) == "anthropic"
            assert sub._manifest_provider == "anthropic"
        finally:
            await sub.aclose()

    @pytest.mark.asyncio
    async def test_no_manifest_threads_model_and_allowed_tools(self):
        (d,) = compile_subagent_descriptors(
            [
                {
                    "agent_type": "coder",
                    "provider": "anthropic",
                    "model_override": "claude-opus-4-7",
                    "allowed_tools": ["Read", "Grep"],
                }
            ]
        )
        ctx = _ctx(d, credentials=_bundle("anthropic"))
        sub = await d.factory(ctx)
        try:
            assert _stage6_provider(sub) == "anthropic"
            report = sub.tool_resolution_report
            assert set(report.resolved) == {"Read", "Grep"}
            # model_override landed in the sub-manifest's model block →
            # the sub-pipeline's PipelineConfig.
            assert sub._config.model.model == "claude-opus-4-7"
        finally:
            await sub.aclose()

    @pytest.mark.asyncio
    async def test_no_provider_resolvable_raises_actionable_error(self):
        (d,) = compile_subagent_descriptors([{"agent_type": "worker"}])
        with pytest.raises(ConfigError, match="no provider could be resolved"):
            await d.factory(_ctx(d, credentials=CredentialBundle()))

    @pytest.mark.asyncio
    async def test_inline_manifest_sub_build(self):
        inline = _manifest(provider="openai").to_dict()
        (d,) = compile_subagent_descriptors(
            [{"agent_type": "persona", "manifest": inline}]
        )
        ctx = _ctx(d, credentials=_bundle("openai"))
        sub = await d.factory(ctx)
        try:
            assert _stage6_provider(sub) == "openai"
        finally:
            await sub.aclose()

    @pytest.mark.asyncio
    async def test_env_id_without_resolver_raises_actionable_error(self):
        (d,) = compile_subagent_descriptors(
            [{"agent_type": "stored", "env_id": "env_abc"}]
        )
        with pytest.raises(ConfigError, match="subagent_env_resolver"):
            await d.factory(_ctx(d, credentials=_bundle("anthropic")))

    @pytest.mark.asyncio
    async def test_env_id_with_resolver_builds(self):
        seen = {}

        def resolver(env_id: str):
            seen["env_id"] = env_id
            return _manifest(provider="anthropic").to_dict()

        (d,) = compile_subagent_descriptors(
            [{"agent_type": "stored", "env_id": "env_abc"}], env_resolver=resolver
        )
        sub = await d.factory(_ctx(d, credentials=_bundle("anthropic")))
        try:
            assert seen["env_id"] == "env_abc"
            assert _stage6_provider(sub) == "anthropic"
        finally:
            await sub.aclose()

    @pytest.mark.asyncio
    async def test_async_resolver_supported(self):
        async def resolver(env_id: str):
            return _manifest(provider="anthropic")

        (d,) = compile_subagent_descriptors(
            [{"agent_type": "stored", "env_id": "env_async"}], env_resolver=resolver
        )
        sub = await d.factory(_ctx(d, credentials=_bundle("anthropic")))
        try:
            assert _stage6_provider(sub) == "anthropic"
        finally:
            await sub.aclose()


# ── Pipeline.from_manifest: compile + merge semantics ────────


class TestFromManifestMerge:
    def test_manifest_only_builds_registry(self):
        m = _manifest(subagents=[{"agent_type": "a"}, {"agent_type": "b"}])
        p = Pipeline.from_manifest(m, credentials=_bundle("anthropic"))
        assert p._subagent_registry is not None
        assert p._subagent_registry.list_types() == ["a", "b"]

    def test_explicit_registry_wins_on_collision(self, caplog):
        explicit_descriptor = SubagentTypeDescriptor(
            agent_type="a", factory=lambda ctx: _FakeSubPipeline(), description="host"
        )
        explicit = SubagentTypeRegistry().register(explicit_descriptor)
        m = _manifest(
            subagents=[{"agent_type": "a", "description": "manifest"}, {"agent_type": "b"}]
        )
        import logging

        with caplog.at_level(logging.INFO, logger="geny_executor.core.pipeline"):
            p = Pipeline.from_manifest(
                m, credentials=_bundle("anthropic"), subagent_registry=explicit
            )
        assert p._subagent_registry is explicit
        assert p._subagent_registry.get("a") is explicit_descriptor
        assert "b" in p._subagent_registry  # manifest entry merged in
        assert "explicit" in caplog.text

    def test_no_subagents_keeps_legacy_behaviour(self):
        p = Pipeline.from_manifest(_manifest(), credentials=_bundle("anthropic"))
        assert p._subagent_registry is None

    def test_resolver_kwarg_threads_to_factories(self):
        resolver = lambda env_id: _manifest()  # noqa: E731
        m = _manifest(subagents=[{"agent_type": "stored", "env_id": "env_1"}])
        p = Pipeline.from_manifest(
            m, credentials=_bundle("anthropic"), subagent_env_resolver=resolver
        )
        factory = p._subagent_registry.get("stored").factory
        assert isinstance(factory, ManifestSubagentPipelineFactory)
        assert factory._env_resolver is resolver


# ── Orchestrator: parent_provider wiring + run_subagent ──────


class TestOrchestratorSurface:
    @pytest.mark.asyncio
    async def test_dispatch_populates_typed_parent_provider(self):
        """host_ergonomics #7 short-term fix: factories read a typed
        field instead of digging the bare key out of shared."""
        received = {}

        def factory(ctx):
            received["ctx"] = ctx
            return _FakeSubPipeline()

        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(agent_type="t", factory=factory)
        )
        orch = SubagentTypeOrchestrator(registry)
        state = PipelineState(session_id="parent")
        state.shared[SharedKeys.PRIMARY_PROVIDER] = "anthropic"
        state.delegate_requests = [{"agent_type": "t", "task": "go"}]

        await orch.orchestrate(state)
        assert received["ctx"].parent_provider == "anthropic"

    @pytest.mark.asyncio
    async def test_dispatch_without_primary_provider_is_none(self):
        received = {}

        def factory(ctx):
            received["ctx"] = ctx
            return _FakeSubPipeline()

        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(agent_type="t", factory=factory)
        )
        state = PipelineState(session_id="parent")
        state.delegate_requests = [{"agent_type": "t", "task": "go"}]
        await SubagentTypeOrchestrator(registry).orchestrate(state)
        assert received["ctx"].parent_provider is None

    @pytest.mark.asyncio
    async def test_run_subagent_returns_record(self):
        """The AgentTool / LocalAgentExecutor call shape now exists on
        the orchestrator (audit: 'two incompatible delegation
        interfaces')."""
        sub = _FakeSubPipeline(_FakeResult(text="done"))
        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: sub)
        )
        orch = SubagentTypeOrchestrator(registry)
        record = await orch.run_subagent("t", "do the thing")
        assert record["success"] is True
        assert record["text"] == "done"
        assert sub.runs[0][0] == "do the thing"

    @pytest.mark.asyncio
    async def test_run_subagent_model_override_is_per_call(self):
        seen = {}

        def factory(ctx):
            seen["model"] = ctx.descriptor.model_override
            return _FakeSubPipeline()

        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(
                agent_type="t", factory=factory, model_override="claude-sonnet-4-6"
            )
        )
        orch = SubagentTypeOrchestrator(registry)
        await orch.run_subagent("t", "go", model="claude-opus-4-7")
        assert seen["model"] == "claude-opus-4-7"
        # The registry descriptor itself is untouched (one-shot replace).
        assert registry.get("t").model_override == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_run_subagent_unknown_type_raises_keyerror(self):
        orch = SubagentTypeOrchestrator(SubagentTypeRegistry())
        with pytest.raises(KeyError):
            await orch.run_subagent("ghost", "go")

    @pytest.mark.asyncio
    async def test_run_subagent_failure_raises_runtimeerror(self):
        def factory(ctx):
            raise RuntimeError("factory exploded")

        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(agent_type="t", factory=factory)
        )
        with pytest.raises(RuntimeError, match="factory_error"):
            await SubagentTypeOrchestrator(registry).run_subagent("t", "go")

    @pytest.mark.asyncio
    async def test_run_subagent_inherits_parent_state(self):
        received = {}

        def factory(ctx):
            received["ctx"] = ctx
            return _FakeSubPipeline()

        registry = SubagentTypeRegistry().register(
            SubagentTypeDescriptor(agent_type="t", factory=factory)
        )
        bundle = _bundle("anthropic")
        state = PipelineState(session_id="parent")
        state.credentials = bundle
        state.shared[SharedKeys.PRIMARY_PROVIDER] = "anthropic"

        await SubagentTypeOrchestrator(registry).run_subagent("t", "go", state=state)
        assert received["ctx"].credentials is bundle
        assert received["ctx"].parent_provider == "anthropic"
        assert received["ctx"].parent_session_id == "parent"
