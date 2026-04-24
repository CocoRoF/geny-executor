"""Phase 6 Week 10-11 — MCP connection-state FSM + disable/enable tests."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from geny_executor.tools.mcp import (
    RECONNECTABLE_STATES,
    MCPConnectionState,
    MCPManager,
    MCPServerConfig,
)
from geny_executor.tools.mcp.errors import MCPConnectionError
from geny_executor.tools.mcp.manager import (
    MCPServerConnection,
    _looks_like_auth_failure,
)


# ─────────────────────────────────────────────────────────────────
# State enum surface
# ─────────────────────────────────────────────────────────────────


class TestStateEnum:
    def test_five_states_present(self):
        names = {s.name for s in MCPConnectionState}
        assert names == {"PENDING", "CONNECTED", "FAILED", "NEEDS_AUTH", "DISABLED"}

    def test_only_connected_is_visible(self):
        assert MCPConnectionState.CONNECTED.is_visible is True
        for s in MCPConnectionState:
            if s is not MCPConnectionState.CONNECTED:
                assert s.is_visible is False, f"{s} unexpectedly visible"

    def test_only_disabled_is_terminal(self):
        assert MCPConnectionState.DISABLED.is_terminal is True
        for s in MCPConnectionState:
            if s is not MCPConnectionState.DISABLED:
                assert s.is_terminal is False

    def test_reconnectable_states_set(self):
        assert MCPConnectionState.PENDING in RECONNECTABLE_STATES
        assert MCPConnectionState.FAILED in RECONNECTABLE_STATES
        assert MCPConnectionState.NEEDS_AUTH in RECONNECTABLE_STATES
        assert MCPConnectionState.CONNECTED not in RECONNECTABLE_STATES
        assert MCPConnectionState.DISABLED not in RECONNECTABLE_STATES


# ─────────────────────────────────────────────────────────────────
# Auth-failure classifier
# ─────────────────────────────────────────────────────────────────


class TestAuthClassifier:
    @pytest.mark.parametrize(
        "msg",
        [
            "401 Unauthorized",
            "Forbidden: missing token",
            "AuthenticationError: invalid token",
            "needs_auth marker raised",
            "request returned 403",
        ],
    )
    def test_recognises_auth_shapes(self, msg):
        assert _looks_like_auth_failure(RuntimeError(msg)) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "Connection refused",
            "TimeoutError after 10s",
            "transport handshake failed",
            "no such file or directory",
        ],
    )
    def test_treats_other_failures_as_generic(self, msg):
        assert _looks_like_auth_failure(RuntimeError(msg)) is False


# ─────────────────────────────────────────────────────────────────
# MCPServerConnection FSM
# ─────────────────────────────────────────────────────────────────


def _config(name: str = "fake") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="/bin/true", transport="stdio")


class TestServerConnectionFSM:
    def test_initial_state_is_pending(self):
        conn = MCPServerConnection(_config())
        assert conn.state is MCPConnectionState.PENDING
        assert conn.is_connected is False
        assert conn.last_error is None

    def test_mark_disabled_sets_state(self):
        conn = MCPServerConnection(_config())
        conn.mark_disabled()
        assert conn.state is MCPConnectionState.DISABLED

    def test_mark_pending_clears_error(self):
        conn = MCPServerConnection(_config())
        conn._state = MCPConnectionState.FAILED
        conn._last_error = RuntimeError("old failure")
        conn.mark_pending()
        assert conn.state is MCPConnectionState.PENDING
        assert conn.last_error is None

    @pytest.mark.asyncio
    async def test_connect_from_disabled_raises(self):
        conn = MCPServerConnection(_config())
        conn.mark_disabled()
        with pytest.raises(RuntimeError, match="DISABLED"):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_generic_failure_lands_in_failed(self):
        conn = MCPServerConnection(_config())
        with patch.object(
            conn,
            "_connect_stdio",
            AsyncMock(side_effect=MCPConnectionError("fake", "connect", message="Connection refused")),
        ):
            with pytest.raises(MCPConnectionError):
                await conn.connect()
        assert conn.state is MCPConnectionState.FAILED
        assert "Connection refused" in str(conn.last_error)

    @pytest.mark.asyncio
    async def test_auth_failure_lands_in_needs_auth(self):
        conn = MCPServerConnection(_config())
        with patch.object(
            conn,
            "_connect_stdio",
            AsyncMock(side_effect=MCPConnectionError("fake", "connect", message="401 Unauthorized")),
        ):
            with pytest.raises(MCPConnectionError):
                await conn.connect()
        assert conn.state is MCPConnectionState.NEEDS_AUTH

    @pytest.mark.asyncio
    async def test_unknown_transport_lands_in_failed(self):
        cfg = MCPServerConfig(name="x", transport="quantum")
        conn = MCPServerConnection(cfg)
        with pytest.raises(MCPConnectionError):
            await conn.connect()
        assert conn.state is MCPConnectionState.FAILED

    @pytest.mark.asyncio
    async def test_disconnect_from_connected_returns_to_pending(self):
        conn = MCPServerConnection(_config())
        conn._state = MCPConnectionState.CONNECTED
        await conn.disconnect()
        assert conn.state is MCPConnectionState.PENDING

    @pytest.mark.asyncio
    async def test_disconnect_does_not_reset_disabled(self):
        conn = MCPServerConnection(_config())
        conn.mark_disabled()
        await conn.disconnect()
        assert conn.state is MCPConnectionState.DISABLED

    @pytest.mark.asyncio
    async def test_disconnect_does_not_reset_failed(self):
        """A FAILED → disconnect chain (e.g. cleanup after a botched
        connect) shouldn't silently flip the state to PENDING and lose
        the failure record."""
        conn = MCPServerConnection(_config())
        conn._state = MCPConnectionState.FAILED
        conn._last_error = RuntimeError("orig")
        await conn.disconnect()
        assert conn.state is MCPConnectionState.FAILED


# ─────────────────────────────────────────────────────────────────
# MCPManager — disable / enable
# ─────────────────────────────────────────────────────────────────


def _registered_manager(name: str = "fake") -> tuple[MCPManager, MCPServerConnection]:
    """Build a manager with a registered + 'connected' fake server.

    Bypasses the real connect handshake — sets state directly so we can
    drive the FSM without spawning subprocesses.
    """
    mgr = MCPManager()
    cfg = _config(name)
    conn = MCPServerConnection(cfg)
    conn._state = MCPConnectionState.CONNECTED
    conn._tools = [{"name": f"{name}_t", "description": "x", "input_schema": {}}]
    mgr._servers[name] = conn
    mgr._configs[name] = cfg
    return mgr, conn


class TestManagerDisable:
    @pytest.mark.asyncio
    async def test_disable_marks_state_and_retains_config(self):
        mgr, conn = _registered_manager("svc")
        await mgr.disable_server("svc")
        assert conn.state is MCPConnectionState.DISABLED
        # Config + connection record are retained
        assert "svc" in mgr._servers
        assert "svc" in mgr._configs

    @pytest.mark.asyncio
    async def test_disable_unknown_server_is_noop(self):
        mgr = MCPManager()
        await mgr.disable_server("nope")  # no exception

    @pytest.mark.asyncio
    async def test_disable_idempotent(self):
        mgr, conn = _registered_manager("svc")
        await mgr.disable_server("svc")
        await mgr.disable_server("svc")  # second call is harmless
        assert conn.state is MCPConnectionState.DISABLED

    @pytest.mark.asyncio
    async def test_discover_skips_disabled(self):
        mgr, _ = _registered_manager("svc")
        # Sanity: connected server contributes
        before = await mgr.discover_tools()
        assert len(before) == 1
        await mgr.disable_server("svc")
        after = await mgr.discover_tools()
        assert after == []


class TestManagerEnable:
    @pytest.mark.asyncio
    async def test_enable_unknown_is_noop(self):
        mgr = MCPManager()
        await mgr.enable_server("nope")  # no exception

    @pytest.mark.asyncio
    async def test_enable_already_connected_is_noop(self):
        mgr, conn = _registered_manager("svc")
        # Don't disable first — enable should not bounce a live conn.
        with patch.object(conn, "connect", AsyncMock()) as connect_mock:
            await mgr.enable_server("svc")
        connect_mock.assert_not_awaited()
        assert conn.state is MCPConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_enable_disabled_attempts_reconnect(self):
        mgr, conn = _registered_manager("svc")
        await mgr.disable_server("svc")
        assert conn.state is MCPConnectionState.DISABLED

        async def _fake_connect():
            conn._state = MCPConnectionState.CONNECTED

        with patch.object(conn, "connect", AsyncMock(side_effect=_fake_connect)) as cm:
            await mgr.enable_server("svc")
        cm.assert_awaited_once()
        assert conn.state is MCPConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_enable_propagates_reconnect_failure(self):
        mgr, conn = _registered_manager("svc")
        await mgr.disable_server("svc")

        async def _broken_connect():
            conn._state = MCPConnectionState.FAILED
            conn._last_error = RuntimeError("still broken")
            raise MCPConnectionError("svc", "connect", message="still broken")

        with patch.object(conn, "connect", AsyncMock(side_effect=_broken_connect)):
            with pytest.raises(MCPConnectionError):
                await mgr.enable_server("svc")
        # State reflects the new failure
        assert conn.state is MCPConnectionState.FAILED


# ─────────────────────────────────────────────────────────────────
# list_server_status surfacing the FSM
# ─────────────────────────────────────────────────────────────────


class TestStatusSurfacing:
    def test_includes_state_field(self):
        mgr, _ = _registered_manager("svc")
        status = mgr.list_server_status()
        assert len(status) == 1
        s = status[0]
        assert s["state"] == "connected"
        assert s["connected"] is True
        assert s["last_error"] is None

    @pytest.mark.asyncio
    async def test_disabled_state_in_status(self):
        mgr, _ = _registered_manager("svc")
        await mgr.disable_server("svc")
        status = mgr.list_server_status()
        assert status[0]["state"] == "disabled"
        assert status[0]["connected"] is False

    def test_failed_state_records_last_error(self):
        mgr, conn = _registered_manager("svc")
        conn._state = MCPConnectionState.FAILED
        conn._last_error = RuntimeError("transport handshake failed")
        status = mgr.list_server_status()
        assert status[0]["state"] == "failed"
        assert "handshake" in status[0]["last_error"]
