"""MCP wrapper tool tests (PR-A.3.3)."""

from __future__ import annotations

from typing import List

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    ListMcpResourcesTool,
    MCPTool,
    McpAuthTool,
    ReadMcpResourceTool,
)


class _FakeMgr:
    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls: List = []

    async def call_tool(self, *, server_name, tool_name, arguments):
        self.calls.append(("call_tool", server_name, tool_name, arguments))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response

    async def list_resources(self, *, server=None, kind=None):
        self.calls.append(("list_resources", server, kind))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response or []

    async def read_resource(self, *, uri):
        self.calls.append(("read_resource", uri))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response

    async def start_oauth(self, *, server_name):
        self.calls.append(("start_oauth", server_name))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def test_all_four_registered():
    for name in ("MCP", "ListMcpResources", "ReadMcpResource", "McpAuth"):
        assert name in BUILT_IN_TOOL_CLASSES


# ── MCP ──────────────────────────────────────────────────────────────


class TestMCPTool:
    @pytest.mark.asyncio
    async def test_calls_manager(self):
        mgr = _FakeMgr(response={"x": 1})
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await MCPTool().execute(
            {"server": "s", "tool": "t", "arguments": {"a": 1}}, ctx,
        )
        assert result.is_error is False
        assert result.content["result"] == {"x": 1}
        assert mgr.calls[0][0] == "call_tool"

    @pytest.mark.asyncio
    async def test_no_manager(self):
        result = await MCPTool().execute(
            {"server": "s", "tool": "t"}, ToolContext(extras={}),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_MANAGER"

    @pytest.mark.asyncio
    async def test_call_failure(self):
        ctx = ToolContext(extras={"mcp_manager": _FakeMgr(raise_exc=RuntimeError("boom"))})
        result = await MCPTool().execute({"server": "s", "tool": "t"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "MCP_CALL_FAILED"


# ── ListMcpResources ─────────────────────────────────────────────────


class TestListResources:
    @pytest.mark.asyncio
    async def test_lists(self):
        mgr = _FakeMgr(response=[{"name": "x"}])
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await ListMcpResourcesTool().execute({}, ctx)
        assert result.content["resources"] == [{"name": "x"}]

    @pytest.mark.asyncio
    async def test_filter_by_server(self):
        mgr = _FakeMgr(response=[])
        ctx = ToolContext(extras={"mcp_manager": mgr})
        await ListMcpResourcesTool().execute({"server": "s1"}, ctx)
        assert mgr.calls[0] == ("list_resources", "s1", None)


# ── ReadMcpResource ──────────────────────────────────────────────────


class TestReadResource:
    @pytest.mark.asyncio
    async def test_reads_uri(self):
        mgr = _FakeMgr(response="content")
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await ReadMcpResourceTool().execute({"uri": "mcp://s/x"}, ctx)
        assert result.content["content"] == "content"
        assert mgr.calls[0] == ("read_resource", "mcp://s/x")


# ── McpAuth ──────────────────────────────────────────────────────────


class TestMcpAuth:
    @pytest.mark.asyncio
    async def test_returns_url_from_dict(self):
        mgr = _FakeMgr(response={"url": "https://auth/start", "state": "x"})
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await McpAuthTool().execute({"server": "s"}, ctx)
        assert result.content["auth_url"] == "https://auth/start"
        assert "instructions" in result.content

    @pytest.mark.asyncio
    async def test_returns_url_from_string(self):
        mgr = _FakeMgr(response="https://auth/raw-string")
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await McpAuthTool().execute({"server": "s"}, ctx)
        assert result.content["auth_url"] == "https://auth/raw-string"

    @pytest.mark.asyncio
    async def test_failure(self):
        mgr = _FakeMgr(raise_exc=RuntimeError("oauth setup failed"))
        ctx = ToolContext(extras={"mcp_manager": mgr})
        result = await McpAuthTool().execute({"server": "s"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "MCP_AUTH_FAILED"
