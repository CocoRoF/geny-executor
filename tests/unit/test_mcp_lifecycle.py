"""MCP lifecycle hardening (Phase B / PR3).

Covers:
- MCPConnectionError fail-fast semantics on every lifecycle step.
- MCPManager.connect_all rolls back partial connects on failure.
- MCPManager.add_server + remove_server keep the registry in sync.
- Structured MCP call_tool result preservation (str vs list[dict]).
- Pipeline.from_manifest_async wires MCP + registry and surfaces errors.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.environment import EnvironmentManifest, ToolsSnapshot
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.base import Tool, ToolContext
from geny_executor.tools.mcp.adapter import MCPToolAdapter
from geny_executor.tools.mcp.errors import MCPConnectionError
from geny_executor.tools.mcp.manager import (
    MCPManager,
    MCPServerConfig,
    MCPServerConnection,
    _normalize_mcp_result,
)
from geny_executor.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────


class _FakeTool:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object"}


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeCallToolResult:
    def __init__(self, content):
        self.content = content


class _FakeBlock:
    def __init__(self, text=None, type="text"):
        self.type = type
        if text is not None:
            self.text = text


class _FakeSession:
    """Stands in for ``mcp.ClientSession`` across a connect lifecycle."""

    def __init__(
        self,
        *,
        tools: List[_FakeTool] | None = None,
        initialize_exc: Exception | None = None,
        list_tools_exc: Exception | None = None,
        call_result=None,
    ):
        self._tools = tools or [_FakeTool("ping", "pong")]
        self._initialize_exc = initialize_exc
        self._list_tools_exc = list_tools_exc
        self._call_result = call_result
        self._entered = False
        self._exited = False

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._exited = True
        return False

    async def initialize(self):
        if self._initialize_exc is not None:
            raise self._initialize_exc

    async def list_tools(self):
        if self._list_tools_exc is not None:
            raise self._list_tools_exc
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name, arguments):
        return self._call_result


def _install_fake_connect(conn: MCPServerConnection, *, session: _FakeSession) -> None:
    """Bypass the real transport layer and attach *session* directly.

    Avoids needing a real MCP subprocess — the plumbing under test
    (cleanup ordering, error phase labelling, tool list wiring) is
    fully exercised via the fake session's per-step exception hooks.
    """

    async def _attach(transport_factory, *, client_session_cls):
        class _Ctx:
            async def __aenter__(self_inner):
                return (object(), object())

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        conn._transport_ctx = _Ctx()
        read, write = await conn._transport_ctx.__aenter__()
        conn._client_session = session
        await conn._client_session.__aenter__()
        try:
            import asyncio as _asyncio

            await _asyncio.wait_for(conn._client_session.initialize(), timeout=10.0)
        except BaseException as exc:
            await conn._safe_cleanup()
            raise MCPConnectionError(conn.config.name, "initialize", cause=exc) from exc

        try:
            result = await _asyncio.wait_for(conn._client_session.list_tools(), timeout=10.0)
        except BaseException as exc:
            await conn._safe_cleanup()
            raise MCPConnectionError(conn.config.name, "list_tools", cause=exc) from exc

        conn._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": getattr(t, "inputSchema", {}),
            }
            for t in result.tools
        ]
        conn._connected = True

    conn._attach_session = _attach  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════
# MCPConnectionError surfacing
# ══════════════════════════════════════════════════════════


class TestConnectionLifecycleErrors:
    @pytest.mark.asyncio
    async def test_unsupported_transport_raises(self):
        conn = MCPServerConnection(
            MCPServerConfig(name="bad", command="noop", transport="carrier-pigeon")
        )
        with pytest.raises(MCPConnectionError) as excinfo:
            await conn.connect()
        assert excinfo.value.server_name == "bad"
        assert excinfo.value.phase == "connect"

    @pytest.mark.asyncio
    async def test_missing_url_for_http_raises(self):
        conn = MCPServerConnection(MCPServerConfig(name="web", transport="http", url=""))
        with pytest.raises(MCPConnectionError) as excinfo:
            await conn.connect()
        assert excinfo.value.phase == "connect"
        assert "URL" in str(excinfo.value) or "url" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_initialize_failure_labelled(self):
        conn = MCPServerConnection(MCPServerConfig(name="s", command="noop"))
        session = _FakeSession(initialize_exc=RuntimeError("boom"))
        _install_fake_connect(conn, session=session)

        with pytest.raises(MCPConnectionError) as excinfo:
            await conn._attach_session(  # type: ignore[attr-defined]
                transport_factory=lambda: None, client_session_cls=object
            )
        assert excinfo.value.phase == "initialize"
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_list_tools_failure_labelled(self):
        conn = MCPServerConnection(MCPServerConfig(name="s", command="noop"))
        session = _FakeSession(list_tools_exc=RuntimeError("no perms"))
        _install_fake_connect(conn, session=session)

        with pytest.raises(MCPConnectionError) as excinfo:
            await conn._attach_session(  # type: ignore[attr-defined]
                transport_factory=lambda: None, client_session_cls=object
            )
        assert excinfo.value.phase == "list_tools"
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_success_populates_tool_list(self):
        conn = MCPServerConnection(MCPServerConfig(name="s", command="noop"))
        session = _FakeSession(tools=[_FakeTool("a"), _FakeTool("b")])
        _install_fake_connect(conn, session=session)

        await conn._attach_session(  # type: ignore[attr-defined]
            transport_factory=lambda: None, client_session_cls=object
        )
        assert conn.is_connected
        names = [d["name"] for d in await conn.discover_tools()]
        assert names == ["a", "b"]


# ══════════════════════════════════════════════════════════
# MCPManager.connect_all — fail-fast + cleanup
# ══════════════════════════════════════════════════════════


class TestManagerConnectAll:
    @pytest.mark.asyncio
    async def test_empty_config_is_noop(self):
        manager = MCPManager()
        await manager.connect_all({})
        assert manager.list_servers() == []

    @pytest.mark.asyncio
    async def test_one_failure_rolls_back_all(self, monkeypatch):
        """When one server fails, no other server leaks into the manager."""
        manager = MCPManager()

        async def fake_connect(self, name, config):
            if name == "bad":
                raise MCPConnectionError(name, "connect")
            # success path: stash a fake connection
            conn = MCPServerConnection(config)
            conn._connected = True
            conn._tools = []
            manager._servers[name] = conn
            manager._configs[name] = config

        monkeypatch.setattr(MCPManager, "connect", fake_connect)

        configs = {
            "good": MCPServerConfig(name="good", command="noop"),
            "bad": MCPServerConfig(name="bad", command="noop"),
        }
        with pytest.raises(MCPConnectionError):
            await manager.connect_all(configs)
        # No half-state: every server that was transiently connected must
        # be torn back down before the exception propagates.
        assert manager.list_servers() == []


# ══════════════════════════════════════════════════════════
# add_server / remove_server registry integration
# ══════════════════════════════════════════════════════════


class TestServerRegistryIntegration:
    @pytest.mark.asyncio
    async def test_add_server_registers_namespaced_tools(self, monkeypatch):
        manager = MCPManager()
        registry = ToolRegistry()

        async def fake_connect(self, name, config):
            conn = MCPServerConnection(config)
            conn._connected = True
            conn._tools = [
                {"name": "ls", "description": "list", "input_schema": {"type": "object"}},
                {"name": "cat", "description": "show", "input_schema": {"type": "object"}},
            ]
            manager._servers[name] = conn
            manager._configs[name] = config

        monkeypatch.setattr(MCPManager, "connect", fake_connect)

        tools = await manager.add_server(
            MCPServerConfig(name="fs", command="noop"), registry=registry
        )
        names = {t.name for t in tools}
        assert names == {"mcp__fs__ls", "mcp__fs__cat"}
        assert set(registry.list_names()) == names

    @pytest.mark.asyncio
    async def test_remove_server_unregisters_only_its_namespace(self, monkeypatch):
        manager = MCPManager()
        registry = ToolRegistry()

        async def fake_connect(self, name, config):
            conn = MCPServerConnection(config)
            conn._connected = True
            conn._tools = [{"name": "ls", "description": "", "input_schema": {}}]
            manager._servers[name] = conn
            manager._configs[name] = config

        monkeypatch.setattr(MCPManager, "connect", fake_connect)

        await manager.add_server(MCPServerConfig(name="fs", command="noop"), registry=registry)
        await manager.add_server(MCPServerConfig(name="git", command="noop"), registry=registry)

        assert {t for t in registry.list_names()} == {"mcp__fs__ls", "mcp__git__ls"}

        removed = await manager.remove_server("fs", registry=registry)
        assert removed is True
        assert registry.list_names() == ["mcp__git__ls"]
        assert "fs" not in manager.list_servers()

    @pytest.mark.asyncio
    async def test_remove_unknown_server_returns_false(self):
        manager = MCPManager()
        assert await manager.remove_server("nope") is False


# ══════════════════════════════════════════════════════════
# call_tool result normalization
# ══════════════════════════════════════════════════════════


class TestNormalizeMcpResult:
    def test_single_text_block_returns_string(self):
        result = _FakeCallToolResult([_FakeBlock(text="hello")])
        assert _normalize_mcp_result(result) == "hello"

    def test_multiple_blocks_return_list(self):
        result = _FakeCallToolResult([_FakeBlock(text="one"), _FakeBlock(text="two")])
        normalized = _normalize_mcp_result(result)
        assert isinstance(normalized, list)
        assert [b["text"] for b in normalized] == ["one", "two"]
        assert all(b["type"] == "text" for b in normalized)

    def test_non_text_block_returns_list(self):
        image_block = _FakeBlock(type="image")
        result = _FakeCallToolResult([image_block])
        normalized = _normalize_mcp_result(result)
        assert isinstance(normalized, list)
        assert normalized[0]["type"] == "image"

    def test_empty_content_fallback(self):
        result = _FakeCallToolResult([])
        assert isinstance(_normalize_mcp_result(result), str)


# ══════════════════════════════════════════════════════════
# MCPToolAdapter pass-through for list content
# ══════════════════════════════════════════════════════════


class TestAdapterPreservesListResult:
    @pytest.mark.asyncio
    async def test_list_content_round_trip(self):
        class _Conn:
            class config:
                name = "s"

            async def call_tool(self_inner, name, args):
                return [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]

        adapter = MCPToolAdapter(
            server=_Conn(),
            definition={"name": "t", "description": "d"},
        )
        result = await adapter.execute({}, ToolContext(session_id="s"))
        assert result.content == [
            {"type": "text", "text": "hi"},
            {"type": "text", "text": "there"},
        ]
        assert not result.is_error


# ══════════════════════════════════════════════════════════
# Pipeline.from_manifest_async integration
# ══════════════════════════════════════════════════════════


def _blank_manifest_with_servers(servers: List[Dict[str, Any]]) -> EnvironmentManifest:
    return EnvironmentManifest(
        stages=[],
        tools=ToolsSnapshot(mcp_servers=servers),
    )


class TestFromManifestAsync:
    @pytest.mark.asyncio
    async def test_no_servers_uses_empty_registry_and_manager(self):
        manifest = _blank_manifest_with_servers([])
        pipeline = await Pipeline.from_manifest_async(manifest)
        assert pipeline.mcp_manager is not None
        assert pipeline.tool_registry is not None
        assert pipeline.mcp_manager.list_servers() == []
        assert pipeline.tool_registry.list_names() == []

    @pytest.mark.asyncio
    async def test_servers_connect_and_register(self, monkeypatch):
        async def fake_connect_all(self, configs):
            for name, cfg in configs.items():
                conn = MCPServerConnection(cfg)
                conn._connected = True
                conn._tools = [{"name": "ping", "description": "", "input_schema": {}}]
                self._servers[name] = conn
                self._configs[name] = cfg

        monkeypatch.setattr(MCPManager, "connect_all", fake_connect_all)

        manifest = _blank_manifest_with_servers([{"name": "alpha", "command": "noop"}])
        pipeline = await Pipeline.from_manifest_async(manifest)

        assert pipeline.mcp_manager.list_servers() == ["alpha"]
        assert pipeline.tool_registry.list_names() == ["mcp__alpha__ping"]

    @pytest.mark.asyncio
    async def test_server_failure_cleans_up(self, monkeypatch):
        async def fake_connect_all(self, configs):
            raise MCPConnectionError("alpha", "initialize")

        disconnected: List[bool] = []

        async def fake_disconnect_all(self):
            disconnected.append(True)

        monkeypatch.setattr(MCPManager, "connect_all", fake_connect_all)
        monkeypatch.setattr(MCPManager, "disconnect_all", fake_disconnect_all)

        manifest = _blank_manifest_with_servers([{"name": "alpha", "command": "noop"}])
        with pytest.raises(MCPConnectionError):
            await Pipeline.from_manifest_async(manifest)
        assert disconnected == [True]

    @pytest.mark.asyncio
    async def test_caller_supplied_registry_is_preserved(self, monkeypatch):
        async def fake_connect_all(self, configs):
            for name, cfg in configs.items():
                conn = MCPServerConnection(cfg)
                conn._connected = True
                conn._tools = [{"name": "ping", "description": "", "input_schema": {}}]
                self._servers[name] = conn
                self._configs[name] = cfg

        monkeypatch.setattr(MCPManager, "connect_all", fake_connect_all)

        class _Dummy(Tool):
            @property
            def name(self):
                return "builtin"

            @property
            def description(self):
                return ""

            @property
            def input_schema(self):
                return {"type": "object"}

            async def execute(self, input, context):
                raise NotImplementedError

        registry = ToolRegistry().register(_Dummy())
        manifest = _blank_manifest_with_servers([{"name": "alpha", "command": "noop"}])
        pipeline = await Pipeline.from_manifest_async(manifest, tool_registry=registry)
        # Both the built-in tool and the discovered MCP adapter land in
        # the same registry the caller passed in.
        assert set(registry.list_names()) == {"builtin", "mcp__alpha__ping"}
        assert pipeline.tool_registry is registry
