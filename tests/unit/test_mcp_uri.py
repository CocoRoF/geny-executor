"""Unit tests for the mcp:// URI scheme + manager-level resource API (S8.3)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from geny_executor.tools.mcp import (
    MCP_URI_SCHEME,
    MCPManager,
    MCPServerConfig,
    MCPURIError,
    build_mcp_uri,
    is_mcp_uri,
    parse_mcp_uri,
)
from geny_executor.tools.mcp.manager import MCPServerConnection
from geny_executor.tools.mcp.state import MCPConnectionState


# ── parse_mcp_uri ────────────────────────────────────────────────────────


class TestParseMcpUri:
    def test_simple(self):
        assert parse_mcp_uri("mcp://github") == ("github", "")
        assert parse_mcp_uri("mcp://github/") == ("github", "")

    def test_with_path(self):
        assert parse_mcp_uri("mcp://github/owner/repo/README.md") == (
            "github",
            "owner/repo/README.md",
        )

    def test_with_native_uri_scheme(self):
        # The path portion is opaque — even if it itself looks like a URI.
        assert parse_mcp_uri("mcp://gdrive/file:///docs/abc") == (
            "gdrive",
            "file:///docs/abc",
        )

    def test_server_name_chars(self):
        # Allowed chars: letters, digits, '_', '.', '-'
        assert parse_mcp_uri("mcp://my-server_2.0/x")[0] == "my-server_2.0"

    def test_wrong_scheme_rejected(self):
        with pytest.raises(MCPURIError, match="not an mcp"):
            parse_mcp_uri("https://example/foo")
        with pytest.raises(MCPURIError, match="not an mcp"):
            parse_mcp_uri("mcp:/single-slash")

    def test_missing_server_rejected(self):
        with pytest.raises(MCPURIError, match="missing the server name"):
            parse_mcp_uri("mcp://")
        with pytest.raises(MCPURIError, match="missing the server name"):
            parse_mcp_uri("mcp:///path-without-server")

    def test_invalid_server_chars_rejected(self):
        with pytest.raises(MCPURIError, match="invalid server name"):
            parse_mcp_uri("mcp://has spaces/x")
        with pytest.raises(MCPURIError, match="invalid server name"):
            parse_mcp_uri("mcp://has:colon/x")

    def test_non_string_rejected(self):
        with pytest.raises(MCPURIError):
            parse_mcp_uri(b"mcp://x/y")  # type: ignore[arg-type]


# ── build_mcp_uri ───────────────────────────────────────────────────────


class TestBuildMcpUri:
    def test_with_path(self):
        assert build_mcp_uri("github", "owner/repo") == "mcp://github/owner/repo"

    def test_no_path(self):
        assert build_mcp_uri("github") == "mcp://github/"
        assert build_mcp_uri("github", "") == "mcp://github/"

    def test_strips_leading_slash(self):
        assert build_mcp_uri("github", "/owner") == "mcp://github/owner"

    def test_invalid_server_rejected(self):
        with pytest.raises(MCPURIError):
            build_mcp_uri("has spaces")

    def test_round_trip(self):
        for server, path in [
            ("github", "owner/repo/README.md"),
            ("gdrive", ""),
            ("a.b-c_d", "x"),
        ]:
            uri = build_mcp_uri(server, path)
            assert parse_mcp_uri(uri) == (server, path)


# ── is_mcp_uri / scheme ────────────────────────────────────────────────


class TestSchemeHelpers:
    def test_is_mcp_uri(self):
        assert is_mcp_uri("mcp://x")
        assert is_mcp_uri("mcp://server/path")
        assert not is_mcp_uri("https://x")
        assert not is_mcp_uri("mcp:")
        assert not is_mcp_uri(123)  # type: ignore[arg-type]

    def test_scheme_constant(self):
        assert MCP_URI_SCHEME == "mcp://"


# ── MCPManager-level resource API ──────────────────────────────────────


def _make_conn(
    name: str,
    *,
    connected: bool = True,
    resources: List[Dict[str, Any]] | None = None,
    bodies: Dict[str, str] | None = None,
) -> MCPServerConnection:
    conn = MCPServerConnection(MCPServerConfig(name=name))
    if connected:
        conn._state = MCPConnectionState.CONNECTED
    conn.list_resources = AsyncMock(return_value=list(resources or []))
    body_map = dict(bodies or {})

    async def _read(uri: str):
        return body_map.get(uri)

    conn.read_resource = _read  # type: ignore[assignment]
    return conn


def _make_manager(*conns: MCPServerConnection) -> MCPManager:
    mgr = MCPManager()
    for conn in conns:
        mgr._servers[conn.config.name] = conn
        mgr._configs[conn.config.name] = conn.config
    return mgr


class TestManagerReadMcpResource:
    @pytest.mark.asyncio
    async def test_routes_to_correct_server(self):
        a = _make_conn("a", bodies={"doc1": "alpha-body"})
        b = _make_conn("b", bodies={"doc1": "beta-body"})
        mgr = _make_manager(a, b)

        assert await mgr.read_mcp_resource("mcp://a/doc1") == "alpha-body"
        assert await mgr.read_mcp_resource("mcp://b/doc1") == "beta-body"

    @pytest.mark.asyncio
    async def test_unknown_server_returns_none(self):
        mgr = _make_manager(_make_conn("a"))
        assert await mgr.read_mcp_resource("mcp://ghost/x") is None

    @pytest.mark.asyncio
    async def test_disconnected_server_returns_none(self):
        conn = _make_conn("srv", connected=False, bodies={"x": "body"})
        mgr = _make_manager(conn)
        assert await mgr.read_mcp_resource("mcp://srv/x") is None

    @pytest.mark.asyncio
    async def test_invalid_uri_raises(self):
        mgr = _make_manager(_make_conn("a"))
        with pytest.raises(MCPURIError):
            await mgr.read_mcp_resource("https://x/y")

    @pytest.mark.asyncio
    async def test_path_passed_verbatim_to_connection(self):
        captured: List[str] = []

        async def _read(uri: str):
            captured.append(uri)
            return "ok"

        conn = _make_conn("srv")
        conn.read_resource = _read  # type: ignore[assignment]
        mgr = _make_manager(conn)

        await mgr.read_mcp_resource("mcp://srv/path/to/file.md")
        assert captured == ["path/to/file.md"]


class TestManagerListAllResources:
    @pytest.mark.asyncio
    async def test_aggregates_across_servers(self):
        a = _make_conn(
            "a",
            resources=[
                {"uri": "doc1", "name": "Doc 1", "description": "", "mimeType": "text/plain"}
            ],
        )
        b = _make_conn(
            "b",
            resources=[
                {"uri": "doc2", "name": "Doc 2", "description": "", "mimeType": "text/plain"}
            ],
        )
        mgr = _make_manager(a, b)
        out = await mgr.list_all_resources()
        assert len(out) == 2
        names = {(e["server"], e["name"]) for e in out}
        assert names == {("a", "Doc 1"), ("b", "Doc 2")}
        # mcp_uri populated.
        uris = {e["mcp_uri"] for e in out}
        assert uris == {"mcp://a/doc1", "mcp://b/doc2"}

    @pytest.mark.asyncio
    async def test_skips_disconnected_servers(self):
        a = _make_conn("a", connected=True, resources=[{"uri": "x"}])
        b = _make_conn("b", connected=False, resources=[{"uri": "y"}])
        mgr = _make_manager(a, b)
        out = await mgr.list_all_resources()
        assert len(out) == 1
        assert out[0]["server"] == "a"

    @pytest.mark.asyncio
    async def test_empty_when_no_servers(self):
        mgr = MCPManager()
        assert await mgr.list_all_resources() == []

    @pytest.mark.asyncio
    async def test_preserves_original_fields(self):
        a = _make_conn(
            "a",
            resources=[
                {"uri": "u", "name": "N", "description": "D", "mimeType": "text/x-custom"}
            ],
        )
        out = await _make_manager(a).list_all_resources()
        entry = out[0]
        assert entry["uri"] == "u"
        assert entry["name"] == "N"
        assert entry["description"] == "D"
        assert entry["mimeType"] == "text/x-custom"
