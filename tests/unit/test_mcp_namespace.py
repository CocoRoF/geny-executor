"""MCP namespace prefix + registry collision warning (Phase A / PR2)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.errors import ToolErrorCode, ToolFailure
from geny_executor.tools.mcp.state import MCPConnectionState
from geny_executor.tools.mcp.adapter import MCPToolAdapter
from geny_executor.tools.mcp.manager import MCPServerConfig, MCPServerConnection
from geny_executor.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────


class _FakeSession:
    """Stands in for mcp.ClientSession for call_tool testing."""

    def __init__(self, *, raise_exc: Exception | None = None, text: str = "ok"):
        self._raise = raise_exc
        self._text = text
        self.called_with: tuple[str, Dict[str, Any]] | None = None

    async def call_tool(self, name, arguments):
        self.called_with = (name, arguments)
        if self._raise is not None:
            raise self._raise

        class _Block:
            def __init__(self, text):
                self.text = text

        class _Result:
            def __init__(self, text):
                self.content = [_Block(text)]

        return _Result(self._text)


def _make_connection(
    server_name: str, *, session: _FakeSession | None = None
) -> MCPServerConnection:
    conn = MCPServerConnection(MCPServerConfig(name=server_name, command="noop"))
    if session is not None:
        conn._client_session = session
        conn._state = MCPConnectionState.CONNECTED
    return conn


class _DummyTool(Tool):
    def __init__(self, name: str):
        self._n = name

    @property
    def name(self) -> str:
        return self._n

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input, context):
        return ToolResult(content="ok")


# ══════════════════════════════════════════════════════════
# MCPToolAdapter naming
# ══════════════════════════════════════════════════════════


class TestMCPToolAdapterNaming:
    def test_name_is_prefixed(self):
        conn = _make_connection("github")
        adapter = MCPToolAdapter(
            server=conn,
            definition={"name": "search_issues", "description": "d"},
        )
        assert adapter.name == "mcp__github__search_issues"
        assert adapter.raw_name == "search_issues"
        assert adapter.server_name == "github"

    def test_api_format_carries_prefixed_name(self):
        conn = _make_connection("fs")
        adapter = MCPToolAdapter(
            server=conn,
            definition={
                "name": "read_file",
                "description": "read",
                "inputSchema": {"type": "object"},
            },
        )
        api = adapter.to_api_format()
        assert api["name"] == "mcp__fs__read_file"

    def test_prefix_applies_across_servers(self):
        conn_a = _make_connection("a")
        conn_b = _make_connection("b")
        defn = {"name": "dup", "description": "dup"}
        assert MCPToolAdapter(conn_a, defn).name == "mcp__a__dup"
        assert MCPToolAdapter(conn_b, defn).name == "mcp__b__dup"

    @pytest.mark.asyncio
    async def test_execute_calls_raw_name(self):
        session = _FakeSession(text="42")
        conn = _make_connection("calc", session=session)
        adapter = MCPToolAdapter(
            server=conn,
            definition={"name": "add", "description": "add"},
        )
        result = await adapter.execute({"a": 1, "b": 2}, ToolContext(session_id="s"))
        assert not result.is_error
        assert result.content == "42"
        # crucially: calls the raw name, not the prefixed display name
        assert session.called_with == ("add", {"a": 1, "b": 2})

    @pytest.mark.asyncio
    async def test_execute_bridges_exception_to_tool_failure(self):
        session = _FakeSession(raise_exc=RuntimeError("network down"))
        conn = _make_connection("remote", session=session)
        adapter = MCPToolAdapter(
            server=conn,
            definition={"name": "ping", "description": "p"},
        )
        with pytest.raises(ToolFailure) as excinfo:
            await adapter.execute({}, ToolContext(session_id="s"))
        err = excinfo.value.error
        assert err.code is ToolErrorCode.TRANSPORT
        assert err.details["server"] == "remote"
        assert err.details["tool"] == "ping"
        assert err.details["exception_type"] == "RuntimeError"


# ══════════════════════════════════════════════════════════
# ToolRegistry collision logging
# ══════════════════════════════════════════════════════════


class TestRegistryCollision:
    def test_collision_emits_warning(self, caplog):
        reg = ToolRegistry()
        reg.register(_DummyTool("x"))
        with caplog.at_level(logging.WARNING, logger="geny_executor.tools.registry"):
            reg.register(_DummyTool("x"))
        assert any("collision" in r.message for r in caplog.records)
        # Second registration still wins
        assert len(reg) == 1

    def test_no_warning_when_same_instance_reregistered(self, caplog):
        reg = ToolRegistry()
        tool = _DummyTool("x")
        reg.register(tool)
        with caplog.at_level(logging.WARNING, logger="geny_executor.tools.registry"):
            reg.register(tool)
        assert not any("collision" in r.message for r in caplog.records)

    def test_distinct_names_do_not_warn(self, caplog):
        reg = ToolRegistry()
        with caplog.at_level(logging.WARNING, logger="geny_executor.tools.registry"):
            reg.register(_DummyTool("a"))
            reg.register(_DummyTool("b"))
        assert not any("collision" in r.message for r in caplog.records)
        assert len(reg) == 2

    def test_mcp_prefixed_tools_never_collide_with_builtins(self, caplog):
        reg = ToolRegistry()
        reg.register(_DummyTool("read_file"))  # built-in style name
        conn = _make_connection("fs")
        adapter = MCPToolAdapter(
            server=conn,
            definition={"name": "read_file", "description": "d"},
        )
        with caplog.at_level(logging.WARNING, logger="geny_executor.tools.registry"):
            reg.register(adapter)
        # Prefix means this is a different slot — no warning fires
        assert not any("collision" in r.message for r in caplog.records)
        assert len(reg) == 2
        assert reg.get("read_file") is not None
        assert reg.get("mcp__fs__read_file") is not None
