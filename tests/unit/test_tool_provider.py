"""Phase 3 Week 7 — ToolProvider Protocol tests.

Covers the :class:`ToolProvider` ABC, the shipped
:class:`BuiltInToolProvider` implementation, and the
:func:`register_providers` / :func:`shutdown_providers` helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.provider import (
    BuiltInToolProvider,
    ToolProvider,
    register_providers,
    shutdown_providers,
)
from geny_executor.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────
# Test tool fixture
# ─────────────────────────────────────────────────────────────────


class _StubTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return f"stub {self._name}"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def execute(self, input, context):
        return ToolResult(content=f"ran {self._name}")


class _RecordingProvider(ToolProvider):
    """ToolProvider subclass that records lifecycle calls in order."""

    def __init__(
        self,
        name: str,
        tool_names: List[str],
        *,
        startup_fails: bool = False,
        shutdown_fails: bool = False,
        trace: List[str] | None = None,
    ):
        self._name = name
        self._tool_names = tool_names
        self._trace = trace if trace is not None else []
        self._startup_fails = startup_fails
        self._shutdown_fails = shutdown_fails

    @property
    def name(self):
        return self._name

    def list_tools(self):
        return [_StubTool(n) for n in self._tool_names]

    async def startup(self):
        self._trace.append(f"startup:{self._name}")
        if self._startup_fails:
            raise RuntimeError(f"{self._name} startup boom")

    async def shutdown(self):
        self._trace.append(f"shutdown:{self._name}")
        if self._shutdown_fails:
            raise RuntimeError(f"{self._name} shutdown boom")


# ─────────────────────────────────────────────────────────────────
# ABC contract
# ─────────────────────────────────────────────────────────────────


class TestABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ToolProvider()  # type: ignore[abstract]

    def test_requires_name(self):
        class _Missing(ToolProvider):
            def list_tools(self):
                return []

        with pytest.raises(TypeError):
            _Missing()  # name is abstract

    def test_requires_list_tools(self):
        class _Missing(ToolProvider):
            @property
            def name(self):
                return "x"

        with pytest.raises(TypeError):
            _Missing()

    def test_lifecycle_defaults_noop(self):
        p = _RecordingProvider("p", [])
        # Default description is empty
        assert isinstance(p.description, str)


# ─────────────────────────────────────────────────────────────────
# BuiltInToolProvider
# ─────────────────────────────────────────────────────────────────


class TestBuiltInProvider:
    def test_default_returns_every_builtin(self):
        from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES

        p = BuiltInToolProvider()
        names = [t.name for t in p.list_tools()]
        assert set(names) == set(BUILT_IN_TOOL_CLASSES.keys())
        assert p.name == "builtin"
        assert "tools" in p.description

    def test_features_narrow_selection(self):
        p = BuiltInToolProvider(features=["filesystem"])
        names = [t.name for t in p.list_tools()]
        assert set(names) == {"Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"}

    def test_names_narrow_selection(self):
        p = BuiltInToolProvider(names=["Read", "Grep"])
        names = [t.name for t in p.list_tools()]
        assert set(names) == {"Read", "Grep"}

    def test_unknown_feature_raises_at_construction(self):
        with pytest.raises(KeyError):
            BuiltInToolProvider(features=["notathing"])

    def test_returns_fresh_instances_per_call(self):
        p = BuiltInToolProvider(names=["Read"])
        a = p.list_tools()
        b = p.list_tools()
        # Fresh instances — mutation on a won't leak
        assert a is not b
        assert a[0] is not b[0]


# ─────────────────────────────────────────────────────────────────
# register_providers
# ─────────────────────────────────────────────────────────────────


class TestRegisterProviders:
    @pytest.mark.asyncio
    async def test_registers_all_tools(self):
        registry = ToolRegistry()
        trace: List[str] = []
        providers = [
            _RecordingProvider("alpha", ["a1", "a2"], trace=trace),
            _RecordingProvider("beta", ["b1"], trace=trace),
        ]
        started = await register_providers(providers, registry)
        assert [p.name for p in started] == ["alpha", "beta"]
        assert trace == ["startup:alpha", "startup:beta"]
        assert {t.name for t in registry.list_all()} == {"a1", "a2", "b1"}

    @pytest.mark.asyncio
    async def test_duplicate_provider_names_rejected(self):
        registry = ToolRegistry()
        providers = [
            _RecordingProvider("dup", ["a"]),
            _RecordingProvider("dup", ["b"]),
        ]
        with pytest.raises(ValueError, match="duplicate tool provider name"):
            await register_providers(providers, registry)

    @pytest.mark.asyncio
    async def test_tool_name_collision_logs_and_skips(self, caplog):
        registry = ToolRegistry()
        # First provider wins a name; second provider's tool is skipped.
        providers = [
            _RecordingProvider("first", ["shared"]),
            _RecordingProvider("second", ["shared", "unique"]),
        ]
        caplog.set_level("WARNING")
        await register_providers(providers, registry)
        names = {t.name for t in registry.list_all()}
        # Both first.shared and second.unique registered; second.shared skipped
        assert names == {"shared", "unique"}
        # Warning mentions the collision
        assert any(
            "already has a tool" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_startup_failure_shuts_down_previously_started(self):
        registry = ToolRegistry()
        trace: List[str] = []
        providers = [
            _RecordingProvider("ok", ["a"], trace=trace),
            _RecordingProvider("broken", ["b"], startup_fails=True, trace=trace),
            _RecordingProvider("never", ["c"], trace=trace),
        ]
        with pytest.raises(RuntimeError, match="broken startup boom"):
            await register_providers(providers, registry)
        # 'ok' should have been shut down; 'never' should not have started
        assert "startup:ok" in trace
        assert "startup:broken" in trace
        assert "startup:never" not in trace
        assert "shutdown:ok" in trace


class TestShutdownProviders:
    @pytest.mark.asyncio
    async def test_shutdown_reverse_order(self):
        trace: List[str] = []
        providers = [
            _RecordingProvider("a", [], trace=trace),
            _RecordingProvider("b", [], trace=trace),
            _RecordingProvider("c", [], trace=trace),
        ]
        await shutdown_providers(providers)
        assert trace == ["shutdown:c", "shutdown:b", "shutdown:a"]

    @pytest.mark.asyncio
    async def test_shutdown_failure_does_not_block_others(self, caplog):
        trace: List[str] = []
        providers = [
            _RecordingProvider("a", [], trace=trace),
            _RecordingProvider("b", [], shutdown_fails=True, trace=trace),
            _RecordingProvider("c", [], trace=trace),
        ]
        caplog.set_level("WARNING")
        await shutdown_providers(providers)
        # All three attempted; 'b' was logged
        assert trace == ["shutdown:c", "shutdown:b", "shutdown:a"]
        assert any("b" in r.message and "shutdown" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────
# Pipeline.from_manifest_async integration
# ─────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_providers_register_through_manifest(self):
        """End-to-end: tool_providers kwarg populates pipeline.tool_registry."""
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(
            manifest,
            tool_providers=[
                BuiltInToolProvider(features=["filesystem"]),
            ],
        )

        assert pipeline.tool_registry is not None
        names = {t.name for t in pipeline.tool_registry.list_all()}
        # BuiltInToolProvider(features=['filesystem']) → 6 tools
        assert {"Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"} == names
        # Pipeline exposes started providers for introspection
        assert len(pipeline.tool_providers) == 1
        assert pipeline.tool_providers[0].name == "builtin"

    @pytest.mark.asyncio
    async def test_shutdown_tool_providers_clears_list(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        manifest = EnvironmentManifest()
        trace: List[str] = []
        pipeline = await Pipeline.from_manifest_async(
            manifest,
            tool_providers=[_RecordingProvider("t1", ["x"], trace=trace)],
        )
        assert pipeline.tool_providers != []
        await pipeline.shutdown_tool_providers()
        assert pipeline.tool_providers == []
        assert "shutdown:t1" in trace

    @pytest.mark.asyncio
    async def test_no_providers_keeps_pipeline_empty(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)
        assert pipeline.tool_providers == []
