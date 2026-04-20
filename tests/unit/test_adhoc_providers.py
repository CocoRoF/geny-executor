"""AdhocToolProvider protocol + external field wiring (Phase C / PR4).

Covers:
- :class:`AdhocToolProvider` Protocol runtime-checkability and shape.
- :class:`ToolsSnapshot.external` round-trip through ``to_dict`` /
  ``from_dict`` (including legacy manifests without the field).
- :meth:`Pipeline.from_manifest` registers only names listed in
  ``manifest.tools.external`` and respects provider precedence.
- :meth:`Pipeline.from_manifest_async` funnels external + MCP tools
  into a single shared registry.
- Caller-supplied registry is reused rather than discarded.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.environment import EnvironmentManifest, ToolsSnapshot
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.mcp.errors import MCPConnectionError
from geny_executor.tools.mcp.manager import (
    MCPManager,
    MCPServerConnection,
)
from geny_executor.tools.providers import AdhocToolProvider
from geny_executor.tools.registry import ToolRegistry


# ── Helpers ────────────────────────────────────────────────


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._name)


class _DictProvider:
    """Minimal provider backed by a name→Tool dict."""

    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def _manifest_with(*, external: List[str] = (), mcp: List[Dict[str, Any]] = ()) -> EnvironmentManifest:
    return EnvironmentManifest(
        stages=[],
        tools=ToolsSnapshot(
            external=list(external),
            mcp_servers=list(mcp),
        ),
    )


# ══════════════════════════════════════════════════════════
# Protocol smoke
# ══════════════════════════════════════════════════════════


class TestAdhocToolProviderProtocol:
    def test_dict_provider_satisfies_protocol(self):
        provider = _DictProvider({"foo": _NamedTool("foo")})
        assert isinstance(provider, AdhocToolProvider)

    def test_missing_methods_fail_isinstance(self):
        class _Incomplete:
            def list_names(self) -> List[str]:
                return []
            # no .get

        assert not isinstance(_Incomplete(), AdhocToolProvider)

    def test_provider_returns_none_for_unknown(self):
        provider = _DictProvider({"foo": _NamedTool("foo")})
        assert provider.get("nope") is None
        assert provider.list_names() == ["foo"]


# ══════════════════════════════════════════════════════════
# ToolsSnapshot.external round-trip
# ══════════════════════════════════════════════════════════


class TestExternalFieldRoundTrip:
    def test_to_dict_includes_external(self):
        snap = ToolsSnapshot(external=["news_search", "search_engine"])
        data = snap.to_dict()
        assert data["external"] == ["news_search", "search_engine"]

    def test_from_dict_reads_external(self):
        snap = ToolsSnapshot.from_dict(
            {"built_in": [], "external": ["x"], "scope": {}}
        )
        assert snap.external == ["x"]

    def test_from_dict_missing_external_defaults_empty(self):
        """Legacy manifests written before v0.22.0 lack the field — the
        load path must not break on them."""
        snap = ToolsSnapshot.from_dict({"built_in": ["Read"]})
        assert snap.external == []
        assert snap.built_in == ["Read"]

    def test_manifest_full_round_trip(self):
        manifest = _manifest_with(external=["alpha", "beta"])
        data = manifest.to_dict()
        restored = EnvironmentManifest.from_dict(data)
        assert restored.tools.external == ["alpha", "beta"]


# ══════════════════════════════════════════════════════════
# Pipeline.from_manifest external-provider wiring
# ══════════════════════════════════════════════════════════


class TestFromManifestExternalProviders:
    def test_external_name_in_manifest_registers_provider_tool(self):
        manifest = _manifest_with(external=["news_search"])
        provider = _DictProvider({"news_search": _NamedTool("news_search")})
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[provider]
        )
        assert pipeline.tool_registry.get("news_search") is not None

    def test_provider_tool_not_in_external_is_ignored(self):
        """Manifest is authoritative — a provider may *offer* more than
        the manifest activates, but the pipeline must only register
        names the manifest names."""
        manifest = _manifest_with(external=["news_search"])
        provider = _DictProvider(
            {
                "news_search": _NamedTool("news_search"),
                "extra_tool": _NamedTool("extra_tool"),
            }
        )
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[provider]
        )
        assert pipeline.tool_registry.list_names() == ["news_search"]

    def test_missing_provider_for_external_name_is_skipped(self, caplog):
        manifest = _manifest_with(external=["not_supplied"])
        provider = _DictProvider({"something_else": _NamedTool("something_else")})
        with caplog.at_level("WARNING"):
            pipeline = Pipeline.from_manifest(
                manifest, adhoc_providers=[provider]
            )
        assert pipeline.tool_registry.list_names() == []
        assert any("not_supplied" in rec.message for rec in caplog.records)

    def test_external_declared_without_providers_warns(self, caplog):
        manifest = _manifest_with(external=["news_search"])
        with caplog.at_level("WARNING"):
            pipeline = Pipeline.from_manifest(manifest)
        assert pipeline.tool_registry.list_names() == []
        assert any("no AdhocToolProvider" in rec.message for rec in caplog.records)

    def test_first_matching_provider_wins(self):
        """Precedence: providers are queried left-to-right. Once one
        claims a name, later providers are not consulted for that
        name."""
        winner = _NamedTool("alpha")
        loser = _NamedTool("alpha")
        prov_a = _DictProvider({"alpha": winner})
        prov_b = _DictProvider({"alpha": loser})
        manifest = _manifest_with(external=["alpha"])
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[prov_a, prov_b]
        )
        assert pipeline.tool_registry.get("alpha") is winner

    def test_second_provider_fills_gap_when_first_returns_none(self):
        """Fallback: when provider A does not supply a name, provider B
        gets a chance."""
        prov_a = _DictProvider({})  # supplies nothing
        prov_b = _DictProvider({"beta": _NamedTool("beta")})
        manifest = _manifest_with(external=["beta"])
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[prov_a, prov_b]
        )
        assert pipeline.tool_registry.get("beta") is not None

    def test_empty_external_skips_providers(self):
        manifest = _manifest_with(external=[])
        provider = _DictProvider({"unused": _NamedTool("unused")})
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[provider]
        )
        assert pipeline.tool_registry.list_names() == []

    def test_caller_supplied_registry_is_populated(self):
        manifest = _manifest_with(external=["alpha"])
        provider = _DictProvider({"alpha": _NamedTool("alpha")})
        registry = ToolRegistry()
        pipeline = Pipeline.from_manifest(
            manifest, adhoc_providers=[provider], tool_registry=registry
        )
        assert pipeline.tool_registry is registry
        assert registry.get("alpha") is not None

    def test_preserves_preexisting_tools_in_caller_registry(self):
        preexisting = _NamedTool("builtin")
        registry = ToolRegistry().register(preexisting)
        manifest = _manifest_with(external=["alpha"])
        provider = _DictProvider({"alpha": _NamedTool("alpha")})
        Pipeline.from_manifest(
            manifest, adhoc_providers=[provider], tool_registry=registry
        )
        assert set(registry.list_names()) == {"builtin", "alpha"}


# ══════════════════════════════════════════════════════════
# Pipeline.from_manifest_async — external + MCP coexist
# ══════════════════════════════════════════════════════════


class TestFromManifestAsyncExternalAndMcp:
    @pytest.mark.asyncio
    async def test_external_only_registers_provider_tools(self):
        manifest = _manifest_with(external=["alpha"])
        provider = _DictProvider({"alpha": _NamedTool("alpha")})
        pipeline = await Pipeline.from_manifest_async(
            manifest, adhoc_providers=[provider]
        )
        assert pipeline.tool_registry.list_names() == ["alpha"]
        assert pipeline.mcp_manager.list_servers() == []

    @pytest.mark.asyncio
    async def test_external_and_mcp_share_registry(self, monkeypatch):
        async def fake_connect_all(self, configs):
            for name, cfg in configs.items():
                conn = MCPServerConnection(cfg)
                conn._connected = True
                conn._tools = [
                    {"name": "ping", "description": "", "input_schema": {}}
                ]
                self._servers[name] = conn
                self._configs[name] = cfg

        monkeypatch.setattr(MCPManager, "connect_all", fake_connect_all)

        manifest = _manifest_with(
            external=["alpha"],
            mcp=[{"name": "srv", "command": "noop"}],
        )
        provider = _DictProvider({"alpha": _NamedTool("alpha")})
        pipeline = await Pipeline.from_manifest_async(
            manifest, adhoc_providers=[provider]
        )
        assert set(pipeline.tool_registry.list_names()) == {
            "alpha",
            "mcp__srv__ping",
        }

    @pytest.mark.asyncio
    async def test_mcp_failure_does_not_hide_external_wiring_flow(
        self, monkeypatch
    ):
        """Even if external tools were registered before MCP blew up,
        the caller should see :class:`MCPConnectionError` surface —
        the failure path takes precedence over partial success."""

        async def boom(self, configs):
            raise MCPConnectionError("srv", "connect")

        async def noop_disconnect(self):
            return None

        monkeypatch.setattr(MCPManager, "connect_all", boom)
        monkeypatch.setattr(MCPManager, "disconnect_all", noop_disconnect)

        manifest = _manifest_with(
            external=["alpha"],
            mcp=[{"name": "srv", "command": "noop"}],
        )
        provider = _DictProvider({"alpha": _NamedTool("alpha")})
        with pytest.raises(MCPConnectionError):
            await Pipeline.from_manifest_async(
                manifest, adhoc_providers=[provider]
            )
