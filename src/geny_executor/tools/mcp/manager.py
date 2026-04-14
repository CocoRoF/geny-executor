"""MCP server manager — connects to MCP servers and discovers tools.

Supports stdio and HTTP (streamable) transports via the ``mcp`` SDK.
Falls back gracefully when the SDK is not installed.
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
from geny_executor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
    Gracefully degrades to a no-op if the ``mcp`` package is missing.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connected = False
        self._tools: List[Dict[str, Any]] = []
        self._client_session: Any = None  # mcp.ClientSession
        self._transport_ctx: Any = None  # context manager for transport
        self._process: Optional[asyncio.subprocess.Process] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.config.transport == "stdio":
            await self._connect_stdio()
        elif self.config.transport in ("http", "sse"):
            await self._connect_http()
        else:
            logger.warning("Unknown MCP transport: %s", self.config.transport)

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (local subprocess)."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning(
                "MCP SDK not installed — server '%s' connected in no-op mode. "
                "Install with: pip install mcp",
                self.config.name,
            )
            self._connected = True  # no-op mode: lifecycle works, call_tool will fail
            return

        env = os.environ.copy()
        env.update(self.config.env)

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env,
        )

        try:
            self._transport_ctx = stdio_client(params)
            read_stream, write_stream = await self._transport_ctx.__aenter__()
            self._client_session = ClientSession(read_stream, write_stream)
            await self._client_session.__aenter__()
            await self._client_session.initialize()
            self._connected = True

            # Discover tools
            result = await self._client_session.list_tools()
            self._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]

            logger.info(
                "MCP stdio connected: %s (%d tools)",
                self.config.name,
                len(self._tools),
            )

        except Exception as e:
            logger.warning(
                "MCP server '%s' connection failed (%s) — running in no-op mode",
                self.config.name, e,
            )
            await self._cleanup()
            self._connected = True  # no-op: lifecycle works, call_tool will fail

    async def _connect_http(self) -> None:
        """Connect via HTTP/SSE transport (remote server)."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            logger.warning(
                "MCP SDK not installed — server '%s' connected in no-op mode.",
                self.config.name,
            )
            self._connected = True  # no-op mode
            return

        if not self.config.url:
            logger.error("MCP HTTP server '%s' has no URL", self.config.name)
            return

        try:
            self._transport_ctx = sse_client(
                self.config.url,
                headers=self.config.headers,
            )
            read_stream, write_stream = await self._transport_ctx.__aenter__()
            self._client_session = ClientSession(read_stream, write_stream)
            await self._client_session.__aenter__()
            await self._client_session.initialize()
            self._connected = True

            result = await self._client_session.list_tools()
            self._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]

            logger.info(
                "MCP HTTP connected: %s (%d tools)",
                self.config.name,
                len(self._tools),
            )

        except Exception as e:
            logger.warning(
                "MCP HTTP server '%s' connection failed (%s) — running in no-op mode",
                self.config.name, e,
            )
            await self._cleanup()
            self._connected = True  # no-op: lifecycle works, call_tool will fail

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        await self._cleanup()
        self._connected = False
        self._tools = []

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

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Returns the tool result as a string.

        Raises:
            RuntimeError: If the server is not connected.
        """
        if not self._connected:
            raise RuntimeError(
                f"MCP server '{self.config.name}' is not connected. "
                f"Cannot call tool '{tool_name}'."
            )
        if self._client_session is None:
            raise RuntimeError(
                f"MCP server '{self.config.name}' is in no-op mode (mcp SDK not installed). "
                f"Cannot call tool '{tool_name}'. Install with: pip install mcp"
            )

        result = await self._client_session.call_tool(tool_name, arguments)

        # Normalize result to string
        if hasattr(result, "content") and result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts)

        return str(result)


class MCPManager:
    """Manages MCP server connections and tool discovery.

    Usage:
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
        """Connect to an MCP server by config."""
        if name in self._servers:
            await self.disconnect(name)
        conn = MCPServerConnection(config)
        self._configs[name] = config
        await conn.connect()
        self._servers[name] = conn

    async def connect_all(self, configs: Dict[str, MCPServerConfig]) -> None:
        """Connect to multiple MCP servers concurrently."""
        tasks = []
        for name, config in configs.items():
            tasks.append(self.connect(name, config))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

    async def build_registry(self) -> ToolRegistry:
        """Discover all tools and return a populated ToolRegistry."""
        registry = ToolRegistry()
        tools = await self.discover_tools()
        for tool in tools:
            registry.register(tool)
        logger.info("MCP registry built: %d tools from %d servers", len(registry), len(self._servers))
        return registry

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
