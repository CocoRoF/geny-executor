"""MCP server manager — connects to MCP servers and discovers tools.

Supports stdio and HTTP (streamable) transports via the ``mcp`` SDK.

As of v0.22.0 the connection lifecycle is **fail-fast**: every failure
mode (SDK missing, transport handshake failure, ``initialize``
timeout, ``list_tools`` error) raises :class:`MCPConnectionError`
instead of silently leaving the server in a zombie "connected but
no-op" state. This makes MCP errors observable at session-start time
rather than surfacing as confusing ``unknown_tool`` failures later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool
from geny_executor.tools.mcp.errors import MCPConnectionError
from geny_executor.tools.mcp.state import MCPConnectionState
from geny_executor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Substring patterns that, when present in a connection error message,
# indicate an authentication problem rather than a transient network
# issue. Matched case-insensitively. Conservative on purpose — false
# positives would push the user to fix credentials they don't actually
# need to fix, but false negatives just mean the host treats it as a
# generic FAILED (still recoverable).
_AUTH_ERROR_HINTS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid token",
    "missing token",
    "needs_auth",
)


def _looks_like_auth_failure(exc: BaseException) -> bool:
    """True when ``exc`` reads like an MCP authentication challenge."""
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_ERROR_HINTS)


def _serialise_mcp_tool(t: Any) -> Dict[str, Any]:
    """Convert one MCP SDK tool object into the cached dict shape.

    Captures ``annotations`` (per the MCP spec) so the adapter can
    map them onto :class:`ToolCapabilities` for orchestration. Hosts
    that bring their own MCP-shaped objects (mocks, custom transports)
    can produce the same dict directly — the adapter only reads from
    it, never from the original SDK type.
    """
    annotations: Dict[str, Any] = {}
    raw_anno = getattr(t, "annotations", None)
    if raw_anno is not None:
        for key in (
            "title",
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ):
            value = getattr(raw_anno, key, None)
            if value is None and isinstance(raw_anno, dict):
                value = raw_anno.get(key)
            if value is not None:
                annotations[key] = value
    return {
        "name": t.name,
        "description": t.description or "",
        "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
        "annotations": annotations,
    }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # stdio | http | sse
    url: str = ""  # for http/sse transport
    headers: Dict[str, str] = field(default_factory=dict)


class MCPServerConnection:
    """Active connection to an MCP server via the ``mcp`` SDK.

    Supports stdio transport (local subprocess) and HTTP transport.
    Raises :class:`MCPConnectionError` on any lifecycle failure so the
    caller can decide (usually: abort session start).
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._tools: List[Dict[str, Any]] = []
        self._client_session: Any = None  # mcp.ClientSession
        self._transport_ctx: Any = None  # context manager for transport
        self._process: Optional[asyncio.subprocess.Process] = None
        # 5-state FSM (Phase 6). PENDING is the canonical start: we
        # haven't tried connecting yet.
        self._state: MCPConnectionState = MCPConnectionState.PENDING
        self._last_error: Optional[BaseException] = None

    @property
    def state(self) -> MCPConnectionState:
        """Current FSM state. See :class:`MCPConnectionState`."""
        return self._state

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception from the last failed connect attempt, if any.

        Cleared on successful connect or admin disable. Useful for
        admin UIs ("why is the github server in FAILED?").
        """
        return self._last_error

    @property
    def is_connected(self) -> bool:
        """Backward-compat shortcut for ``state == CONNECTED``."""
        return self._state is MCPConnectionState.CONNECTED

    def mark_disabled(self) -> None:
        """Move into ``DISABLED`` regardless of current state.

        Idempotent. After cleanup-completes the manager calls this so
        a re-enable later starts fresh from ``PENDING``.
        """
        self._state = MCPConnectionState.DISABLED
        self._last_error = None

    def mark_pending(self) -> None:
        """Move back into ``PENDING`` — admin re-enable / retry path."""
        self._state = MCPConnectionState.PENDING
        self._last_error = None

    async def connect(self) -> None:
        """Connect to the MCP server.

        Drives the FSM:

        * On entry, refuses to reconnect from ``DISABLED`` (admins must
          ``enable_server`` first).
        * On success: ``CONNECTED``.
        * On auth-shaped failure: ``NEEDS_AUTH`` + ``last_error``.
        * On generic failure: ``FAILED`` + ``last_error``.

        Raises:
            MCPConnectionError: For unknown transport, missing SDK, or
                any transport / initialize / list_tools failure. The
                state has already been transitioned to FAILED /
                NEEDS_AUTH before the exception propagates.
            RuntimeError: When called on a ``DISABLED`` connection.
        """
        if self._state is MCPConnectionState.DISABLED:
            raise RuntimeError(
                f"MCP server '{self.config.name}' is DISABLED; call "
                f"enable_server() before reconnecting"
            )
        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            elif self.config.transport in ("http", "sse"):
                await self._connect_http()
            else:
                raise MCPConnectionError(
                    self.config.name,
                    "connect",
                    message=(
                        f"MCP server '{self.config.name}' has unsupported "
                        f"transport '{self.config.transport}' (expected "
                        "stdio | http | sse)"
                    ),
                )
        except BaseException as exc:
            # Classify into NEEDS_AUTH vs generic FAILED so admin UIs
            # can prompt the user differently. Auth-classification is
            # best-effort — false negatives just produce FAILED.
            if _looks_like_auth_failure(exc):
                self._state = MCPConnectionState.NEEDS_AUTH
            else:
                self._state = MCPConnectionState.FAILED
            self._last_error = exc
            raise

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (local subprocess)."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MCPConnectionError(
                self.config.name,
                "sdk_missing",
                cause=exc,
                message=(
                    f"MCP SDK not installed — server "
                    f"'{self.config.name}' cannot connect. "
                    "Install with: pip install mcp"
                ),
            ) from exc

        env = os.environ.copy()
        env.update(self.config.env)

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env,
        )

        await self._attach_session(
            lambda: stdio_client(params),
            client_session_cls=ClientSession,
        )

    async def _connect_http(self) -> None:
        """Connect via HTTP/SSE transport (remote server)."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise MCPConnectionError(
                self.config.name,
                "sdk_missing",
                cause=exc,
                message=(
                    f"MCP SDK not installed — server "
                    f"'{self.config.name}' cannot connect. "
                    "Install with: pip install mcp"
                ),
            ) from exc

        if not self.config.url:
            raise MCPConnectionError(
                self.config.name,
                "connect",
                message=(
                    f"MCP HTTP server '{self.config.name}' is missing a URL "
                    "(set MCPServerConfig.url)"
                ),
            )

        await self._attach_session(
            lambda: sse_client(self.config.url, headers=self.config.headers),
            client_session_cls=ClientSession,
        )

    async def _attach_session(self, transport_factory, *, client_session_cls) -> None:
        """Shared stdio/http attachment: transport → initialize → list_tools.

        Any failure cleans up and re-raises as :class:`MCPConnectionError`
        labelled with the phase it happened in.
        """
        try:
            self._transport_ctx = transport_factory()
            read_stream, write_stream = await self._transport_ctx.__aenter__()
            self._client_session = client_session_cls(read_stream, write_stream)
            await self._client_session.__aenter__()
        except BaseException as exc:
            await self._safe_cleanup()
            raise MCPConnectionError(self.config.name, "connect", cause=exc) from exc

        try:
            await asyncio.wait_for(self._client_session.initialize(), timeout=10.0)
        except BaseException as exc:
            await self._safe_cleanup()
            raise MCPConnectionError(self.config.name, "initialize", cause=exc) from exc

        try:
            result = await asyncio.wait_for(self._client_session.list_tools(), timeout=10.0)
        except BaseException as exc:
            await self._safe_cleanup()
            raise MCPConnectionError(self.config.name, "list_tools", cause=exc) from exc

        self._tools = [_serialise_mcp_tool(t) for t in result.tools]
        self._state = MCPConnectionState.CONNECTED
        self._last_error = None
        logger.info(
            "MCP %s connected: %s (%d tools)",
            self.config.transport,
            self.config.name,
            len(self._tools),
        )

    async def _safe_cleanup(self) -> None:
        try:
            await self._cleanup()
        except BaseException:
            pass

    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        Transitions ``CONNECTED`` → ``PENDING`` (idle ready for
        reconnect). Does not change ``DISABLED`` / ``FAILED`` /
        ``NEEDS_AUTH`` — those are admin-driven states and shouldn't
        flip just because cleanup ran.
        """
        await self._cleanup()
        self._tools = []
        if self._state is MCPConnectionState.CONNECTED:
            self._state = MCPConnectionState.PENDING

    async def _cleanup(self) -> None:
        """Clean up client session and transport."""
        if self._client_session is not None:
            try:
                await self._client_session.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_session = None

        if self._transport_ctx is not None:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_ctx = None

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions discovered at connect time."""
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Returns:
            The raw MCP response as either:
              * ``str`` — single text block (the common, LLM-friendly case).
              * ``list[dict]`` — when the response contains multiple blocks
                or any non-text content (image, resource, …). Each dict
                has the shape ``{"type": "text"|"image"|..., "text": ...}``
                mirroring Anthropic's content-block format. Preserving
                the structure prevents downstream code from having to
                re-parse a flattened string.

        Raises:
            RuntimeError: If the server is not connected.
        """
        if not self.is_connected:
            raise RuntimeError(
                f"MCP server '{self.config.name}' is not connected (state={self._state.value!r}). "
                f"Cannot call tool '{tool_name}'."
            )
        if self._client_session is None:
            raise RuntimeError(
                f"MCP server '{self.config.name}' has no active client session. "
                f"Cannot call tool '{tool_name}'."
            )

        result = await self._client_session.call_tool(tool_name, arguments)
        return _normalize_mcp_result(result)


def _normalize_mcp_result(result: Any) -> Any:
    """Convert an MCP call_tool response into str or list[dict].

    The decision rule: if the response has exactly one text block, return
    the text as a bare string — most tools fit this mould and keeping a
    string preserves the API-compatible result shape everyone already
    expects. Otherwise, return a list of block dicts so multi-block and
    non-text content (image, resource) survive intact.
    """
    content = getattr(result, "content", None)
    if not content:
        return str(result)

    blocks: List[Dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None) or "text"
        if hasattr(block, "text") and isinstance(getattr(block, "text", None), str):
            blocks.append({"type": block_type, "text": block.text})
        elif hasattr(block, "model_dump"):
            try:
                blocks.append(block.model_dump())
                continue
            except Exception:
                pass
            blocks.append({"type": block_type, "text": str(block)})
        else:
            blocks.append({"type": block_type, "text": str(block)})

    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return blocks[0]["text"]
    return blocks


class MCPManager:
    """Manages MCP server connections and tool discovery.

    Usage::

        manager = MCPManager()
        await manager.connect("github", MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_..."},
        ))
        registry = await manager.build_registry()
    """

    def __init__(self) -> None:
        self._servers: Dict[str, MCPServerConnection] = {}
        self._configs: Dict[str, MCPServerConfig] = {}

    async def connect(self, name: str, config: MCPServerConfig) -> None:
        """Connect to an MCP server by config.

        Raises:
            MCPConnectionError: On any connection / initialize / list_tools failure.
        """
        if name in self._servers:
            await self.disconnect(name)
        conn = MCPServerConnection(config)
        self._configs[name] = config
        try:
            await conn.connect()
        except BaseException:
            self._configs.pop(name, None)
            raise
        self._servers[name] = conn

    async def connect_all(self, configs: Dict[str, MCPServerConfig]) -> None:
        """Connect to multiple MCP servers concurrently.

        Fail-fast: on the first failure, already-running tasks are
        cancelled, already-connected servers are disconnected, and the
        failure is re-raised. No caller ever sees a half-connected
        manager.
        """
        if not configs:
            return

        async def _connect_one(name: str, cfg: MCPServerConfig) -> None:
            await self.connect(name, cfg)

        tasks = [asyncio.create_task(_connect_one(name, cfg)) for name, cfg in configs.items()]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Drain cancellations.
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.disconnect_all()
            raise

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        conn = self._servers.pop(name, None)
        self._configs.pop(name, None)
        if conn:
            await conn.disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in list(self._servers.keys()):
            await self.disconnect(name)

    async def disable_server(self, name: str) -> None:
        """Mute a server without losing its configuration.

        Closes any active connection, marks the connection ``DISABLED``,
        and retains both the connection object AND its config so a
        future :meth:`enable_server` is one call away.

        Idempotent — calling on an already-disabled server is a no-op
        (no exception). Calling on an unknown server is also a no-op,
        matching the conservative ergonomics of admin APIs.

        Distinct from :meth:`disconnect`, which evicts the server
        entirely (no in-memory record after the call).
        """
        conn = self._servers.get(name)
        if conn is None:
            return
        if conn.state is MCPConnectionState.DISABLED:
            return
        try:
            await conn.disconnect()
        except Exception:
            logger.warning(
                "MCP server %r disconnect during disable raised; continuing",
                name,
                exc_info=True,
            )
        conn.mark_disabled()

    async def enable_server(self, name: str) -> None:
        """Re-enable a previously disabled server and attempt reconnect.

        Transitions the connection's state from ``DISABLED`` to
        ``PENDING`` and immediately tries to reconnect. On reconnect
        failure, the connection lands in ``FAILED`` / ``NEEDS_AUTH``
        as usual — the exception propagates so the caller can surface
        the reason.

        No-op when the server is unknown OR already in a non-DISABLED
        state (the latter so accidental double-enables don't bounce
        live connections).
        """
        conn = self._servers.get(name)
        if conn is None:
            return
        if conn.state is not MCPConnectionState.DISABLED:
            return
        conn.mark_pending()
        await conn.connect()

    async def discover_tools(self) -> List[Tool]:
        """Discover and wrap all tools from all connected servers."""
        from geny_executor.tools.mcp.adapter import MCPToolAdapter

        tools: List[Tool] = []
        for name, conn in self._servers.items():
            if conn.is_connected:
                definitions = await conn.discover_tools()
                for defn in definitions:
                    tools.append(
                        MCPToolAdapter(
                            server=conn,
                            definition=defn,
                        )
                    )
        return tools

    async def discover_all(self) -> List[Tool]:
        """Alias for :meth:`discover_tools` — readable at session-start."""
        return await self.discover_tools()

    async def build_registry(self, registry: Optional[ToolRegistry] = None) -> ToolRegistry:
        """Discover all tools and register them into *registry* (or a fresh one).

        When *registry* is supplied, adapters are added to it in place so
        built-in / adhoc tools already there are preserved.
        """
        reg = registry if registry is not None else ToolRegistry()
        tools = await self.discover_tools()
        for tool in tools:
            reg.register(tool)
        logger.info(
            "MCP registry populated: %d tools from %d servers",
            len(tools),
            len(self._servers),
        )
        return reg

    def list_servers(self) -> List[str]:
        """List connected server names."""
        return list(self._servers.keys())

    def is_connected(self, name: str) -> bool:
        """Check if a server is connected."""
        conn = self._servers.get(name)
        return conn.is_connected if conn else False

    @classmethod
    def from_config_file(cls, path: str) -> MCPManager:
        """Load MCP configuration from .mcp.json file.

        Compatible with existing Geny MCP config format:
        {"mcpServers": {"name": {"command": "...", "args": [...], "env": {...}}}}
        """
        manager = cls()
        config_path = Path(path)
        if not config_path.exists():
            return manager

        with open(config_path, "r") as f:
            data = json.load(f)

        servers = data.get("mcpServers", data.get("servers", {}))
        for name, server_cfg in servers.items():
            transport = server_cfg.get("transport", "stdio")
            config = MCPServerConfig(
                name=name,
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []),
                env=server_cfg.get("env", {}),
                transport=transport,
                url=server_cfg.get("url", ""),
                headers=server_cfg.get("headers", {}),
            )
            manager._configs[name] = config

        return manager

    async def connect_from_loaded_configs(self) -> None:
        """Connect all servers loaded via from_config_file().

        Useful for async initialization after loading configs synchronously.
        """
        configs = dict(self._configs)
        await self.connect_all(configs)

    # ── Dynamic management (Phase 2 additions) ──────────────

    async def add_server(
        self,
        config: MCPServerConfig,
        *,
        registry: Optional[ToolRegistry] = None,
    ) -> List[Tool]:
        """Add and connect an MCP server at runtime.

        Raises :class:`MCPConnectionError` on failure. When *registry* is
        given, discovered adapters are registered into it immediately so
        the tool becomes routable in a single call.
        """
        await self.connect(config.name, config)
        conn = self._servers.get(config.name)
        if conn is None:
            return []

        from geny_executor.tools.mcp.adapter import MCPToolAdapter

        definitions = await conn.discover_tools()
        adapters = [MCPToolAdapter(server=conn, definition=d) for d in definitions]
        if registry is not None:
            for adapter in adapters:
                registry.register(adapter)
        return adapters

    async def remove_server(
        self,
        name: str,
        *,
        registry: Optional[ToolRegistry] = None,
    ) -> bool:
        """Disconnect and remove an MCP server.

        When *registry* is given, every tool whose name matches the
        server's namespace prefix (``mcp__{name}__*``) is also
        unregistered — guaranteeing no orphan adapters are left
        pointing at a dead session.
        """
        if name not in self._servers:
            return False

        await self.disconnect(name)

        if registry is not None:
            prefix = f"mcp__{name}__"
            for tool_name in [n for n in registry.list_names() if n.startswith(prefix)]:
                registry.unregister(tool_name)
        return True

    def list_server_status(self) -> List[Dict[str, Any]]:
        """Return status for all servers, including the FSM ``state``.

        Phase 6: each entry now carries ``state`` (``pending`` /
        ``connected`` / ``failed`` / ``needs_auth`` / ``disabled``)
        plus ``last_error`` when one is recorded. The legacy
        ``connected`` boolean stays for back-compat — admin UIs that
        already render it keep working.
        """
        statuses = []
        for name, conn in self._servers.items():
            statuses.append(
                {
                    "name": name,
                    "state": conn.state.value,
                    "connected": conn.is_connected,
                    "transport": conn.config.transport,
                    "tool_count": len(conn._tools),
                    "has_session": conn._client_session is not None,
                    "last_error": (str(conn.last_error) if conn.last_error is not None else None),
                }
            )
        return statuses

    async def refresh_tools(self, name: str) -> List[Tool]:
        """Re-discover tools from a connected server."""
        conn = self._servers.get(name)
        if not conn or not conn.is_connected or not conn._client_session:
            return []

        from geny_executor.tools.mcp.adapter import MCPToolAdapter

        try:
            result = await asyncio.wait_for(conn._client_session.list_tools(), timeout=10.0)
            conn._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]
            return [MCPToolAdapter(server=conn, definition=d) for d in conn._tools]
        except Exception as e:
            logger.warning("Failed to refresh tools from '%s': %s", name, e)
            return []

    async def test_connection(self, config: MCPServerConfig) -> Dict[str, Any]:
        """Test connection to an MCP server without persisting it.

        Returns a dict with ``success``, ``tools_discovered``, ``error``.
        """
        import time

        conn = MCPServerConnection(config)
        start = time.monotonic()
        try:
            await conn.connect()
            elapsed = (time.monotonic() - start) * 1000
            tools = await conn.discover_tools()
            await conn.disconnect()
            return {
                "success": True,
                "latency_ms": round(elapsed, 1),
                "tools_discovered": len(tools),
                "error": None,
            }
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            try:
                await conn.disconnect()
            except Exception:
                pass
            return {
                "success": False,
                "latency_ms": round(elapsed, 1),
                "tools_discovered": 0,
                "error": str(e),
            }
