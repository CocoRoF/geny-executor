"""Phase 6 Week 11-12 — MCP annotation → ToolCapabilities + attach_runtime tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from geny_executor.tools.base import ToolCapabilities, ToolContext
from geny_executor.tools.mcp.adapter import MCPToolAdapter, _annotations_to_capabilities
from geny_executor.tools.mcp.manager import (
    MCPServerConfig,
    MCPServerConnection,
    _serialise_mcp_tool,
)
from geny_executor.tools.mcp.state import MCPConnectionState


# ─────────────────────────────────────────────────────────────────
# annotations → ToolCapabilities mapping
# ─────────────────────────────────────────────────────────────────


class TestAnnotationsToCapabilities:
    def test_empty_falls_back_to_default(self):
        caps = _annotations_to_capabilities({})
        # Default is fail-closed: not concurrency-safe.
        assert caps.concurrency_safe is False
        assert caps.read_only is False
        assert caps.destructive is False
        assert caps.idempotent is False
        assert caps.network_egress is False

    def test_read_only_implies_concurrency_safe(self):
        caps = _annotations_to_capabilities({"readOnlyHint": True})
        assert caps.read_only is True
        assert caps.concurrency_safe is True

    def test_destructive_overrides_read_only(self):
        """If a server inconsistently sets BOTH readOnly and destructive,
        we must err on the side of caution and serialise."""
        caps = _annotations_to_capabilities(
            {"readOnlyHint": True, "destructiveHint": True}
        )
        assert caps.destructive is True
        assert caps.concurrency_safe is False

    def test_idempotent_propagates(self):
        caps = _annotations_to_capabilities({"idempotentHint": True})
        assert caps.idempotent is True

    def test_open_world_means_network_egress(self):
        caps = _annotations_to_capabilities({"openWorldHint": True})
        assert caps.network_egress is True

    def test_full_combination(self):
        caps = _annotations_to_capabilities(
            {
                "readOnlyHint": True,
                "idempotentHint": True,
                "openWorldHint": True,
            }
        )
        assert caps.read_only is True
        assert caps.concurrency_safe is True
        assert caps.idempotent is True
        assert caps.network_egress is True
        assert caps.destructive is False


# ─────────────────────────────────────────────────────────────────
# _serialise_mcp_tool
# ─────────────────────────────────────────────────────────────────


def _fake_mcp_tool(name: str, *, annotations=None, description: str = "x"):
    """Build a duck-typed object mirroring the mcp SDK's Tool shape."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object"},
        annotations=annotations,
    )


class TestSerialiseMcpTool:
    def test_no_annotations_yields_empty_dict(self):
        d = _serialise_mcp_tool(_fake_mcp_tool("t"))
        assert d["name"] == "t"
        assert d["annotations"] == {}

    def test_object_annotations(self):
        anno = SimpleNamespace(
            readOnlyHint=True,
            destructiveHint=None,
            idempotentHint=True,
            openWorldHint=None,
            title="Read T",
        )
        d = _serialise_mcp_tool(_fake_mcp_tool("t", annotations=anno))
        # None-valued attrs are dropped, present ones preserved
        assert d["annotations"]["readOnlyHint"] is True
        assert d["annotations"]["idempotentHint"] is True
        assert d["annotations"]["title"] == "Read T"
        assert "destructiveHint" not in d["annotations"]
        assert "openWorldHint" not in d["annotations"]

    def test_dict_annotations_also_supported(self):
        # Some SDKs / mocks pass annotations as a plain dict.
        d = _serialise_mcp_tool(
            _fake_mcp_tool("t", annotations={"readOnlyHint": True})
        )
        assert d["annotations"]["readOnlyHint"] is True


# ─────────────────────────────────────────────────────────────────
# MCPToolAdapter.capabilities
# ─────────────────────────────────────────────────────────────────


def _adapter_for(definition: Dict[str, Any]) -> MCPToolAdapter:
    cfg = MCPServerConfig(name="srv")
    conn = MCPServerConnection(cfg)
    return MCPToolAdapter(server=conn, definition=definition)


class TestAdapterCapabilities:
    def test_no_annotations_default_caps(self):
        adapter = _adapter_for({"name": "tool", "description": "x"})
        caps = adapter.capabilities({})
        assert caps == ToolCapabilities()

    def test_read_only_annotation_propagates(self):
        adapter = _adapter_for(
            {
                "name": "search",
                "description": "x",
                "annotations": {"readOnlyHint": True, "openWorldHint": True},
            }
        )
        caps = adapter.capabilities({})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.network_egress is True


# ─────────────────────────────────────────────────────────────────
# attach_runtime(mcp_manager=...)
# ─────────────────────────────────────────────────────────────────


def _connected_manager(name: str, *, tools_definitions):
    """Build a manager-with-fake-server that doesn't need a real MCP SDK."""
    from geny_executor.tools.mcp.manager import MCPManager

    mgr = MCPManager()
    cfg = MCPServerConfig(name=name)
    conn = MCPServerConnection(cfg)
    conn._state = MCPConnectionState.CONNECTED
    conn._tools = tools_definitions
    mgr._servers[name] = conn
    mgr._configs[name] = cfg
    return mgr


class TestPipelineAttachManager:
    @pytest.mark.asyncio
    async def test_attach_replaces_manager_and_seeds_registry(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        # Build a pipeline first via the manifest path so it has an
        # empty registry attached.
        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)
        assert pipeline.tool_registry is not None
        before_names = {t.name for t in pipeline.tool_registry.list_all()}

        # Construct a manager with two fake tools and attach it.
        mgr = _connected_manager(
            "search-svc",
            tools_definitions=[
                {
                    "name": "search",
                    "description": "x",
                    "input_schema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                },
                {
                    "name": "delete",
                    "description": "y",
                    "input_schema": {"type": "object"},
                    "annotations": {"destructiveHint": True},
                },
            ],
        )
        pipeline.attach_runtime(mcp_manager=mgr)

        # Manager handle was swapped
        assert pipeline.mcp_manager is mgr
        # Registry was seeded with the prefixed names
        names_after = {t.name for t in pipeline.tool_registry.list_all()}
        added = names_after - before_names
        assert {"mcp__search-svc__search", "mcp__search-svc__delete"} == added

        # Capabilities reflect annotations
        search_tool = pipeline.tool_registry.get("mcp__search-svc__search")
        delete_tool = pipeline.tool_registry.get("mcp__search-svc__delete")
        assert search_tool is not None and delete_tool is not None
        assert search_tool.capabilities({}).concurrency_safe is True
        assert delete_tool.capabilities({}).destructive is True
        assert delete_tool.capabilities({}).concurrency_safe is False

    @pytest.mark.asyncio
    async def test_attach_skips_disconnected_servers(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)

        mgr = _connected_manager(
            "live",
            tools_definitions=[
                {
                    "name": "go",
                    "description": "ok",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        # Add a second server in DISABLED state — should not contribute tools.
        cfg2 = MCPServerConfig(name="dead")
        dead_conn = MCPServerConnection(cfg2)
        dead_conn.mark_disabled()
        dead_conn._tools = [
            {"name": "ghost", "description": "x", "input_schema": {}}
        ]
        mgr._servers["dead"] = dead_conn
        mgr._configs["dead"] = cfg2

        pipeline.attach_runtime(mcp_manager=mgr)

        names = {t.name for t in pipeline.tool_registry.list_all()}
        assert "mcp__live__go" in names
        assert "mcp__dead__ghost" not in names

    @pytest.mark.asyncio
    async def test_attach_does_not_clobber_existing_names(self):
        """If the registry already has a tool with the same prefixed
        name (e.g. an adhoc bridge), the attach pass keeps it."""
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline
        from geny_executor.tools.base import Tool, ToolResult

        class _Sentinel(Tool):
            @property
            def name(self):
                return "mcp__live__go"

            @property
            def description(self):
                return "sentinel"

            @property
            def input_schema(self):
                return {"type": "object"}

            async def execute(self, input, context):
                return ToolResult(content="sentinel")

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)
        pipeline.tool_registry.register(_Sentinel())

        mgr = _connected_manager(
            "live",
            tools_definitions=[
                {"name": "go", "description": "real", "input_schema": {}}
            ],
        )
        pipeline.attach_runtime(mcp_manager=mgr)

        # Sentinel survives
        survivor = pipeline.tool_registry.get("mcp__live__go")
        assert survivor.description == "sentinel"

    @pytest.mark.asyncio
    async def test_attach_after_run_started_raises(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(manifest)
        pipeline._has_started = True  # simulate post-run

        with pytest.raises(RuntimeError, match="started"):
            pipeline.attach_runtime(mcp_manager=_connected_manager("x", tools_definitions=[]))
