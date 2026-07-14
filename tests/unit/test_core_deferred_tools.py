"""Core vs deferred tool exposure (2.42.0).

Covers the token-optimization contract:

- :class:`ToolRegistry` core flags, runtime activation, and
  ``to_api_format(exposed_only=True)``.
- ``ToolsSnapshot.core_overrides`` round-trip + legacy manifests.
- ``_resolve_core_flag`` exact / wildcard / default resolution.
- Manifest build policy: framework built-ins register core, external
  provider tools register deferred, overrides flip either way, and
  ``ToolSearch`` auto-registers whenever deferred tools exist.
- Stage 3 ships only exposed schemas and picks up activations on the
  registry-version bump.
- ``ToolSearch`` searches the full catalogue and activates deferred
  matches.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.environment import EnvironmentManifest, ToolsSnapshot
from geny_executor.core.pipeline import Pipeline, _resolve_core_flag
from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.artifact.default.stage import SystemStage
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in.tool_search_tool import ToolSearchTool
from geny_executor.tools.registry import ToolRegistry
from tests._fixtures.manifest_entries import required_stage_entries


class _NamedTool(Tool):
    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description or f"{name} tool"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._name)


class _DictProvider:
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def _manifest(
    *,
    built_in: List[str] = (),
    external: List[str] = (),
    core_overrides: Optional[Dict[str, bool]] = None,
) -> EnvironmentManifest:
    return EnvironmentManifest(
        stages=required_stage_entries(),
        tools=ToolsSnapshot(
            built_in=list(built_in),
            external=list(external),
            core_overrides=dict(core_overrides or {}),
        ),
    )


# ── ToolRegistry mechanics ─────────────────────────────────


class TestRegistryCoreFlags:
    def test_register_defaults_to_core(self):
        reg = ToolRegistry().register(_NamedTool("a"))
        assert reg.is_core("a")
        assert reg.is_exposed("a")
        assert reg.list_deferred() == []

    def test_register_deferred(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        assert not reg.is_core("a")
        assert not reg.is_exposed("a")
        assert [t.name for t in reg.list_deferred()] == ["a"]
        assert reg.list_exposed() == []

    def test_activate_promotes_and_bumps_version(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        v = reg.version
        assert reg.activate("a") is True
        assert reg.is_exposed("a")
        assert not reg.is_core("a")  # activation is not a core promotion
        assert reg.version == v + 1
        # Idempotent — no second bump.
        assert reg.activate("a") is True
        assert reg.version == v + 1

    def test_activate_core_tool_is_noop(self):
        reg = ToolRegistry().register(_NamedTool("a"))
        v = reg.version
        assert reg.activate("a") is True
        assert reg.version == v

    def test_activate_unknown_returns_false(self):
        assert ToolRegistry().activate("ghost") is False

    def test_deactivate_demotes(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        reg.activate("a")
        assert reg.deactivate("a") is True
        assert not reg.is_exposed("a")

    def test_set_core_flips_and_bumps_version(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        v = reg.version
        assert reg.set_core("a", True) is True
        assert reg.is_core("a") and reg.is_exposed("a")
        assert reg.version == v + 1
        # No-op change → no bump.
        assert reg.set_core("a", True) is True
        assert reg.version == v + 1
        assert reg.set_core("ghost", True) is False

    def test_unregister_clears_flags(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        reg.activate("a")
        reg.unregister("a")
        reg.register(_NamedTool("a"), core=False)
        # Fresh registration must not inherit the old activation.
        assert not reg.is_exposed("a")

    def test_reregistration_resets_activation(self):
        reg = ToolRegistry().register(_NamedTool("a"), core=False)
        reg.activate("a")
        reg.register(_NamedTool("a"), core=False)
        assert not reg.is_exposed("a")

    def test_to_api_format_exposed_only(self):
        reg = (
            ToolRegistry()
            .register(_NamedTool("core_tool"))
            .register(_NamedTool("hidden_tool"), core=False)
        )
        full = {d["name"] for d in reg.to_api_format()}
        exposed = {d["name"] for d in reg.to_api_format(exposed_only=True)}
        assert full == {"core_tool", "hidden_tool"}
        assert exposed == {"core_tool"}
        reg.activate("hidden_tool")
        exposed_after = {d["name"] for d in reg.to_api_format(exposed_only=True)}
        assert exposed_after == {"core_tool", "hidden_tool"}


# ── ToolsSnapshot.core_overrides ───────────────────────────


class TestCoreOverridesSnapshot:
    def test_round_trip(self):
        snap = ToolsSnapshot(built_in=["Read"], core_overrides={"Read": False, "x": True})
        restored = ToolsSnapshot.from_dict(snap.to_dict())
        assert restored.core_overrides == {"Read": False, "x": True}

    def test_legacy_manifest_without_field(self):
        restored = ToolsSnapshot.from_dict({"built_in": ["Read"]})
        assert restored.core_overrides == {}

    def test_manifest_round_trip(self):
        manifest = _manifest(built_in=["Read"], core_overrides={"Read": False})
        restored = EnvironmentManifest.from_dict(manifest.to_dict())
        assert restored.tools.core_overrides == {"Read": False}


# ── _resolve_core_flag ─────────────────────────────────────


class TestResolveCoreFlag:
    def test_default_when_no_override(self):
        assert _resolve_core_flag("Read", {}, True) is True
        assert _resolve_core_flag("news", {}, False) is False

    def test_exact_match_wins(self):
        assert _resolve_core_flag("Read", {"Read": False}, True) is False
        assert _resolve_core_flag("news", {"news": True}, False) is True

    def test_wildcard_prefix(self):
        overrides = {"mcp__github__*": True}
        assert _resolve_core_flag("mcp__github__create_issue", overrides, False) is True
        assert _resolve_core_flag("mcp__slack__post", overrides, False) is False

    def test_exact_beats_wildcard(self):
        overrides = {"mcp__github__*": True, "mcp__github__delete_repo": False}
        assert _resolve_core_flag("mcp__github__delete_repo", overrides, False) is False
        assert _resolve_core_flag("mcp__github__create_issue", overrides, False) is True

    def test_longest_wildcard_wins(self):
        overrides = {"mcp__*": False, "mcp__github__*": True}
        assert _resolve_core_flag("mcp__github__x", overrides, False) is True
        assert _resolve_core_flag("mcp__slack__x", overrides, True) is False


# ── Manifest build policy ──────────────────────────────────


class TestManifestCorePolicy:
    def test_built_ins_register_core(self):
        pipeline = Pipeline.from_manifest(_manifest(built_in=["Read", "Grep"]))
        reg = pipeline.tool_registry
        assert reg.is_core("Read") and reg.is_core("Grep")
        # No deferred tools → no auto ToolSearch.
        assert reg.get("ToolSearch") is None

    def test_built_in_override_to_deferred(self):
        pipeline = Pipeline.from_manifest(
            _manifest(built_in=["Read", "Grep"], core_overrides={"Grep": False})
        )
        reg = pipeline.tool_registry
        assert reg.is_core("Read")
        assert not reg.is_exposed("Grep")
        # Deferred tool exists → ToolSearch auto-registered as core.
        assert reg.is_core("ToolSearch")

    def test_external_defaults_deferred(self):
        provider = _DictProvider({"news_search": _NamedTool("news_search")})
        pipeline = Pipeline.from_manifest(
            _manifest(external=["news_search"]), adhoc_providers=[provider]
        )
        reg = pipeline.tool_registry
        assert not reg.is_exposed("news_search")
        assert reg.is_core("ToolSearch")

    def test_external_override_to_core(self):
        provider = _DictProvider({"news_search": _NamedTool("news_search")})
        pipeline = Pipeline.from_manifest(
            _manifest(external=["news_search"], core_overrides={"news_search": True}),
            adhoc_providers=[provider],
        )
        reg = pipeline.tool_registry
        assert reg.is_core("news_search")
        assert reg.get("ToolSearch") is None  # nothing deferred remains

    def test_external_shadowing_built_in_keeps_core(self):
        """A host hardening a framework built-in via external must not
        silently pull the tool out of the model's list."""
        provider = _DictProvider({"Read": _NamedTool("Read")})
        pipeline = Pipeline.from_manifest(
            _manifest(built_in=["Read"], external=["Read"]), adhoc_providers=[provider]
        )
        assert pipeline.tool_registry.is_core("Read")

    def test_tool_search_forced_core_when_demoted(self):
        provider = _DictProvider({"news_search": _NamedTool("news_search")})
        pipeline = Pipeline.from_manifest(
            _manifest(
                built_in=["ToolSearch"],
                external=["news_search"],
                core_overrides={"ToolSearch": False},
            ),
            adhoc_providers=[provider],
        )
        # Demoting ToolSearch while deferred tools exist would strand
        # them — the build forces it back to core.
        assert pipeline.tool_registry.is_core("ToolSearch")


# ── Stage 3 exposure ───────────────────────────────────────


class TestStage3Exposure:
    @pytest.mark.asyncio
    async def test_state_tools_only_exposed(self):
        reg = (
            ToolRegistry()
            .register(_NamedTool("core_tool"))
            .register(_NamedTool("hidden_tool"), core=False)
        )
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        assert {d["name"] for d in state.tools} == {"core_tool"}

    @pytest.mark.asyncio
    async def test_activation_reaches_next_iteration(self):
        reg = (
            ToolRegistry()
            .register(_NamedTool("core_tool"))
            .register(_NamedTool("hidden_tool"), core=False)
        )
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        assert {d["name"] for d in state.tools} == {"core_tool"}
        # ToolSearch hit mid-turn: activate bumps the registry version,
        # so the next Stage 3 pass rebuilds state.tools.
        reg.activate("hidden_tool")
        await stage.execute(None, state)
        assert {d["name"] for d in state.tools} == {"core_tool", "hidden_tool"}


# ── ToolSearch discovery + activation ──────────────────────


class TestToolSearchActivation:
    @pytest.mark.asyncio
    async def test_deferred_match_is_activated(self):
        reg = (
            ToolRegistry()
            .register(_NamedTool("Read", "Read a file"))
            .register(_NamedTool("xlsx_edit", "Edit spreadsheet workbooks"), core=False)
        )
        ctx = ToolContext(tool_registry=reg)
        result = await ToolSearchTool().execute({"query": "spreadsheet"}, ctx)
        assert not result.is_error
        assert result.metadata["activated"] == ["xlsx_edit"]
        assert reg.is_exposed("xlsx_edit")
        assert "[activated]" in result.content
        assert "next step" in result.content

    @pytest.mark.asyncio
    async def test_core_match_reported_available(self):
        reg = ToolRegistry().register(_NamedTool("Read", "Read a file"))
        ctx = ToolContext(tool_registry=reg)
        result = await ToolSearchTool().execute({"query": "read"}, ctx)
        assert result.metadata["activated"] == []
        assert "[available]" in result.content

    @pytest.mark.asyncio
    async def test_deferred_tools_are_searchable(self):
        """The whole point: a tool absent from state.tools must still
        be discoverable through the registry-backed search."""
        reg = ToolRegistry().register(
            _NamedTool("secret_tool", "Frobnicate the widget"), core=False
        )
        ctx = ToolContext(tool_registry=reg)
        result = await ToolSearchTool().execute({"query": "frobnicate"}, ctx)
        assert result.metadata["results_count"] == 1
        assert reg.is_exposed("secret_tool")

    @pytest.mark.asyncio
    async def test_no_registry_falls_back_to_state_view(self):
        class _View:
            tools = [{"name": "alpha", "description": "alpha tool", "input_schema": {}}]

        ctx = ToolContext(state_view=_View())
        result = await ToolSearchTool().execute({"query": "alpha"}, ctx)
        assert result.metadata["results_count"] == 1
        assert result.metadata["activated"] == []

    @pytest.mark.asyncio
    async def test_limit_caps_activation(self):
        reg = ToolRegistry()
        for i in range(5):
            reg.register(_NamedTool(f"widget_{i}", "widget maker"), core=False)
        ctx = ToolContext(tool_registry=reg)
        result = await ToolSearchTool().execute({"query": "widget", "limit": 2}, ctx)
        assert result.metadata["results_count"] == 2
        assert len(result.metadata["activated"]) == 2
        assert len([n for n in reg.activated_names()]) == 2


# ── End-to-end agent loop ──────────────────────────────────


class TestAgentLoopEndToEnd:
    @pytest.mark.asyncio
    async def test_discovery_within_one_turn(self):
        """The full token contract in one turn: request 1 carries only
        core schemas; the LLM calls ToolSearch; request 2 (same turn,
        next loop iteration) carries the activated deferred schema and
        the LLM calls the discovered tool successfully."""
        from geny_executor import Pipeline, PipelineConfig
        from geny_executor.core.state import TokenUsage
        from geny_executor.stages.s01_input import InputStage
        from geny_executor.stages.s04_guard import GuardStage, IterationGuard
        from geny_executor.stages.s06_api import APIStage, APIResponse, MockProvider
        from geny_executor.stages.s06_api.retry import NoRetry
        from geny_executor.stages.s06_api.types import ContentBlock
        from geny_executor.stages.s07_token import TokenStage
        from geny_executor.stages.s09_parse import ParseStage
        from geny_executor.stages.s10_tool import ToolStage
        from geny_executor.stages.s16_loop import LoopStage, StandardLoopController
        from geny_executor.stages.s21_yield import YieldStage

        def _tool_use(name: str, tool_input: dict, tool_id: str) -> APIResponse:
            return APIResponse(
                content=[
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tool_id,
                        tool_name=name,
                        tool_input=tool_input,
                        raw={
                            "type": "tool_use",
                            "id": tool_id,
                            "name": name,
                            "input": tool_input,
                        },
                    )
                ],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                model="test",
            )

        registry = (
            ToolRegistry()
            .register(_NamedTool("visible_tool", "Always-on helper"))
            .register(_NamedTool("secret_tool", "Frobnicate the widget"), core=False)
            .register(ToolSearchTool())
        )

        provider = MockProvider()
        provider.add_response(_tool_use("ToolSearch", {"query": "frobnicate"}, "tu_1"))
        provider.add_response(_tool_use("secret_tool", {}, "tu_2"))
        provider.add_response(
            APIResponse(
                content=[ContentBlock(type="text", text="done")],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                model="test",
            )
        )

        pipeline = Pipeline(PipelineConfig(name="core-deferred-e2e"))
        pipeline.register_stage(InputStage())
        pipeline.register_stage(SystemStage(prompt="sys", tool_registry=registry))
        pipeline.register_stage(GuardStage([IterationGuard(max_iterations=10)]))
        pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
        pipeline.register_stage(TokenStage())
        pipeline.register_stage(ParseStage())
        pipeline.register_stage(ToolStage(registry=registry))
        pipeline.register_stage(LoopStage(StandardLoopController()))
        pipeline.register_stage(YieldStage())

        result = await pipeline.run("go")
        assert result.success

        history = provider.call_history
        assert len(history) == 3
        first = {d["name"] for d in (history[0].tools or [])}
        second = {d["name"] for d in (history[1].tools or [])}
        # Request 1: deferred schema withheld.
        assert first == {"visible_tool", "ToolSearch"}
        # Request 2 (post-ToolSearch, same turn): schema activated.
        assert second == {"visible_tool", "ToolSearch", "secret_tool"}
        assert registry.is_exposed("secret_tool")
        # The discovered tool actually executed (its ToolResult echoes
        # its name into the tool_result block of request 3's messages).
        assert "secret_tool" in str(history[2].messages)


class TestDeferredCatalogInSystemPrompt:
    """Progressive disclosure tier 0: the model must be able to SEE that
    hidden tools exist (you can't search for what you don't know about)."""

    def _reg(self):
        return (
            ToolRegistry()
            .register(_NamedTool("core_tool"))
            .register(_NamedTool("hidden_alpha"), core=False)
            .register(_NamedTool("hidden_beta"), core=False)
        )

    @pytest.mark.asyncio
    async def test_catalog_appended_to_system(self):
        stage = SystemStage(prompt="sys", tool_registry=self._reg())
        state = PipelineState()
        await stage.execute(None, state)
        assert "sys" in state.system
        assert "Additional tools (hidden" in state.system
        assert "hidden_alpha" in state.system
        assert "hidden_beta" in state.system
        assert "core_tool —" not in state.system.split("Additional tools")[1]
        assert "ToolSearch" in state.system  # the compact usage rule

    @pytest.mark.asyncio
    async def test_catalog_is_cache_stable_across_activation(self):
        """Activating a tool must NOT change the system text — the catalog
        derives from the core flag, keeping the prompt-cache prefix warm."""
        reg = self._reg()
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        before = state.system
        reg.activate("hidden_alpha")
        await stage.execute(None, state)
        assert state.system == before
        # ...while the tools export DID pick up the activation.
        assert "hidden_alpha" in {d["name"] for d in state.tools}

    @pytest.mark.asyncio
    async def test_no_deferred_tools_no_catalog(self):
        reg = ToolRegistry().register(_NamedTool("core_tool"))
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        assert "Additional tools" not in state.system

    @pytest.mark.asyncio
    async def test_new_registration_refreshes_catalog(self):
        reg = self._reg()
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        reg.register(_NamedTool("hidden_gamma"), core=False)
        await stage.execute(None, state)
        assert "hidden_gamma" in state.system

    @pytest.mark.asyncio
    async def test_oversized_catalog_degrades_to_names_only(self):
        reg = ToolRegistry().register(_NamedTool("core_tool"))
        for i in range(120):
            reg.register(_NamedTool(f"hidden_tool_number_{i:03d}"), core=False)
        stage = SystemStage(prompt="sys", tool_registry=reg)
        state = PipelineState()
        await stage.execute(None, state)
        block = state.system.split("Additional tools")[1]
        assert "hidden_tool_number_000" in block
        assert len(block) < 6_000  # capped, not 120 full lines
