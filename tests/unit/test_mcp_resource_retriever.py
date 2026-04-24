"""Phase 7 Sprint S7.2 — MCPResourceRetriever tests.

Stage 2's :class:`MCPResourceRetriever` walks the bound
:class:`MCPManager`'s CONNECTED servers, lists resources, filters by
the query, and reads each match into a :class:`MemoryChunk`. Tests
mock ``list_resources`` / ``read_resource`` at the connection level
so we don't need a live MCP SDK or subprocess.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context import (
    MCPResourceRetriever,
    MemoryChunk,
)
from geny_executor.tools.mcp.manager import MCPManager, MCPServerConfig, MCPServerConnection
from geny_executor.tools.mcp.state import MCPConnectionState


# ─────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────


def _make_conn(
    name: str,
    *,
    connected: bool = True,
    resources: List[Dict[str, Any]] | None = None,
    bodies: Dict[str, str] | None = None,
    list_raises: bool = False,
    read_raises: List[str] | None = None,
) -> MCPServerConnection:
    """Build a connection with mocked list_resources / read_resource."""
    conn = MCPServerConnection(MCPServerConfig(name=name))
    if connected:
        conn._state = MCPConnectionState.CONNECTED
    if list_raises:
        conn.list_resources = AsyncMock(side_effect=RuntimeError("list boom"))
    else:
        conn.list_resources = AsyncMock(return_value=list(resources or []))

    body_map = dict(bodies or {})
    raises = set(read_raises or [])

    async def _read(uri: str):
        if uri in raises:
            raise RuntimeError(f"read boom on {uri}")
        return body_map.get(uri)

    conn.read_resource = _read  # type: ignore[assignment]
    return conn


def _make_manager(*conns: MCPServerConnection) -> MCPManager:
    mgr = MCPManager()
    for conn in conns:
        mgr._servers[conn.config.name] = conn
        mgr._configs[conn.config.name] = conn.config
    return mgr


def _state(session_id: str = "s") -> PipelineState:
    return PipelineState(session_id=session_id)


# ─────────────────────────────────────────────────────────────────
# Empty / disconnected paths
# ─────────────────────────────────────────────────────────────────


class TestEmptyAndDisconnected:
    @pytest.mark.asyncio
    async def test_empty_manager_yields_nothing(self):
        retriever = MCPResourceRetriever(MCPManager())
        result = await retriever.retrieve("anything", _state())
        assert result == []

    @pytest.mark.asyncio
    async def test_disconnected_servers_skipped(self):
        conn = _make_conn("dead", connected=False, resources=[{"uri": "x"}])
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("", _state())
        # Disconnected → list_resources should not have been called
        assert result == []

    @pytest.mark.asyncio
    async def test_no_resources_yields_nothing(self):
        conn = _make_conn("live", resources=[])
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("query", _state())
        assert result == []


# ─────────────────────────────────────────────────────────────────
# Basic listing + reading
# ─────────────────────────────────────────────────────────────────


class TestBasicListing:
    @pytest.mark.asyncio
    async def test_empty_query_returns_everything_under_cap(self):
        conn = _make_conn(
            "docs",
            resources=[
                {"uri": "file://a", "name": "Alpha", "mimeType": "text/plain"},
                {"uri": "file://b", "name": "Beta", "mimeType": "text/plain"},
            ],
            bodies={"file://a": "alpha body", "file://b": "beta body"},
        )
        retriever = MCPResourceRetriever(_make_manager(conn), max_resources=10)

        result = await retriever.retrieve("", _state())

        assert [c.key for c in result] == ["file://a", "file://b"]
        assert [c.content for c in result] == ["alpha body", "beta body"]
        assert all(c.source == "mcp_resource" for c in result)
        assert result[0].metadata["server"] == "docs"
        assert result[0].metadata["mimeType"] == "text/plain"

    @pytest.mark.asyncio
    async def test_substring_query_filters(self):
        conn = _make_conn(
            "docs",
            resources=[
                {"uri": "file://readme.md", "name": "README", "description": "project entry"},
                {"uri": "file://changelog.md", "name": "CHANGELOG", "description": "release log"},
                {"uri": "file://license.md", "name": "LICENSE", "description": "terms"},
            ],
            bodies={
                "file://readme.md": "...readme...",
                "file://changelog.md": "...changelog...",
                "file://license.md": "...license...",
            },
        )
        retriever = MCPResourceRetriever(_make_manager(conn), max_resources=10)

        result = await retriever.retrieve("README", _state())
        assert [c.key for c in result] == ["file://readme.md"]

    @pytest.mark.asyncio
    async def test_query_matches_description_too(self):
        conn = _make_conn(
            "docs",
            resources=[
                {
                    "uri": "file://x",
                    "name": "x",
                    "description": "explains the SQL backend",
                },
                {
                    "uri": "file://y",
                    "name": "y",
                    "description": "redis stuff",
                },
            ],
            bodies={"file://x": "X", "file://y": "Y"},
        )
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("sql", _state())
        assert [c.key for c in result] == ["file://x"]


# ─────────────────────────────────────────────────────────────────
# Caps + filter_fn
# ─────────────────────────────────────────────────────────────────


class TestCapsAndFilter:
    @pytest.mark.asyncio
    async def test_max_resources_caps_results(self):
        many = [
            {"uri": f"file://r{i}", "name": f"r{i}", "description": ""}
            for i in range(10)
        ]
        bodies = {r["uri"]: f"body {i}" for i, r in enumerate(many)}
        conn = _make_conn("server", resources=many, bodies=bodies)
        retriever = MCPResourceRetriever(_make_manager(conn), max_resources=3)
        result = await retriever.retrieve("", _state())
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_max_resources_clamped_to_at_least_one(self):
        retriever = MCPResourceRetriever(MCPManager(), max_resources=0)
        # Constructor clamps to 1
        assert retriever.max_resources == 1

    @pytest.mark.asyncio
    async def test_filter_fn_excludes_entries(self):
        conn = _make_conn(
            "server",
            resources=[
                {"uri": "file://a", "name": "a", "mimeType": "text/plain"},
                {"uri": "file://b", "name": "b", "mimeType": "image/png"},
            ],
            bodies={"file://a": "A", "file://b": "B"},
        )
        retriever = MCPResourceRetriever(
            _make_manager(conn),
            filter_fn=lambda raw: raw.get("mimeType", "").startswith("text/"),
        )
        result = await retriever.retrieve("", _state())
        assert [c.key for c in result] == ["file://a"]

    @pytest.mark.asyncio
    async def test_filter_fn_exception_drops_entry(self):
        conn = _make_conn(
            "server",
            resources=[
                {"uri": "file://x", "name": "x"},
                {"uri": "file://y", "name": "y"},
            ],
            bodies={"file://x": "X", "file://y": "Y"},
        )

        def _broken_filter(raw):
            if raw["uri"].endswith("x"):
                raise ValueError("bad")
            return True

        retriever = MCPResourceRetriever(_make_manager(conn), filter_fn=_broken_filter)
        result = await retriever.retrieve("", _state())
        assert [c.key for c in result] == ["file://y"]


# ─────────────────────────────────────────────────────────────────
# Failure isolation
# ─────────────────────────────────────────────────────────────────


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_list_resources_failure_skips_only_that_server(self):
        good = _make_conn(
            "good",
            resources=[{"uri": "file://g"}],
            bodies={"file://g": "good"},
        )
        bad = _make_conn("bad", list_raises=True)
        retriever = MCPResourceRetriever(_make_manager(good, bad))
        result = await retriever.retrieve("", _state())
        # Bad server contributed nothing; good still produced its chunk
        assert [c.key for c in result] == ["file://g"]

    @pytest.mark.asyncio
    async def test_read_resource_failure_skips_one_uri(self):
        conn = _make_conn(
            "server",
            resources=[
                {"uri": "file://ok"},
                {"uri": "file://broken"},
                {"uri": "file://ok2"},
            ],
            bodies={"file://ok": "OK", "file://ok2": "OK2"},
            read_raises=["file://broken"],
        )
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("", _state())
        keys = [c.key for c in result]
        assert "file://ok" in keys
        assert "file://ok2" in keys
        assert "file://broken" not in keys

    @pytest.mark.asyncio
    async def test_read_returns_none_skips_entry(self):
        # No body for the resource → read_resource returns None.
        conn = _make_conn(
            "server",
            resources=[{"uri": "file://a"}, {"uri": "file://b"}],
            bodies={"file://b": "B"},  # 'a' not in map → None
        )
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("", _state())
        assert [c.key for c in result] == ["file://b"]

    @pytest.mark.asyncio
    async def test_resource_without_uri_skipped(self):
        conn = _make_conn(
            "server",
            resources=[{"uri": ""}, {"uri": "file://ok"}],
            bodies={"file://ok": "OK"},
        )
        retriever = MCPResourceRetriever(_make_manager(conn))
        result = await retriever.retrieve("", _state())
        assert [c.key for c in result] == ["file://ok"]


# ─────────────────────────────────────────────────────────────────
# Multi-server behaviour
# ─────────────────────────────────────────────────────────────────


class TestMultiServer:
    @pytest.mark.asyncio
    async def test_aggregates_across_servers(self):
        a = _make_conn(
            "a", resources=[{"uri": "file://a/x"}], bodies={"file://a/x": "AX"}
        )
        b = _make_conn(
            "b", resources=[{"uri": "file://b/y"}], bodies={"file://b/y": "BY"}
        )
        retriever = MCPResourceRetriever(_make_manager(a, b))
        result = await retriever.retrieve("", _state())
        keys = sorted(c.key for c in result)
        assert keys == ["file://a/x", "file://b/y"]
        servers = {c.metadata["server"] for c in result}
        assert servers == {"a", "b"}

    @pytest.mark.asyncio
    async def test_global_cap_across_servers(self):
        a = _make_conn(
            "a",
            resources=[{"uri": f"file://a/{i}"} for i in range(5)],
            bodies={f"file://a/{i}": f"A{i}" for i in range(5)},
        )
        b = _make_conn(
            "b",
            resources=[{"uri": f"file://b/{i}"} for i in range(5)],
            bodies={f"file://b/{i}": f"B{i}" for i in range(5)},
        )
        retriever = MCPResourceRetriever(_make_manager(a, b), max_resources=3)
        result = await retriever.retrieve("", _state())
        # Cap applies globally, not per-server.
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────
# Strategy metadata
# ─────────────────────────────────────────────────────────────────


class TestStrategyMetadata:
    def test_name(self):
        r = MCPResourceRetriever(MCPManager())
        assert r.name == "mcp_resource"

    def test_description_includes_cap(self):
        r = MCPResourceRetriever(MCPManager(), max_resources=7)
        assert "7" in r.description


# ─────────────────────────────────────────────────────────────────
# Integration with MCPServerConnection.list/read_resource helpers
# ─────────────────────────────────────────────────────────────────


class TestConnectionHelpers:
    @pytest.mark.asyncio
    async def test_list_resources_disconnected_returns_empty(self):
        conn = MCPServerConnection(MCPServerConfig(name="x"))
        # Default state PENDING → not connected
        assert await conn.list_resources() == []

    @pytest.mark.asyncio
    async def test_read_resource_disconnected_returns_none(self):
        conn = MCPServerConnection(MCPServerConfig(name="x"))
        assert await conn.read_resource("file://anything") is None
