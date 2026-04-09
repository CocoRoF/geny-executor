"""MCP server manager — connects to MCP servers and discovers tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from geny_executor.tools.base import Tool
from geny_executor.tools.mcp.adapter import MCPToolAdapter


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # stdio | sse


class MCPServerConnection:
    """Active connection to an MCP server.

    NOTE: This is a structural placeholder. Real implementation
    requires the `mcp` SDK package for actual server communication.
    The interface is designed for drop-in replacement.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connected = False
        self._tools: List[Dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to the MCP server."""
        # Placeholder — real implementation uses mcp.ClientSession
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._connected = False
        self._tools = []

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the server."""
        # Placeholder — real implementation calls list_tools()
        return self._tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        # Placeholder — real implementation calls call_tool()
        raise NotImplementedError(
            f"MCP tool call '{tool_name}' requires active MCP server connection. "
            "Install and configure the 'mcp' package for real MCP integration."
        )


class MCPManager:
    """Manages MCP server connections and tool discovery.

    Usage:
        manager = MCPManager()
        await manager.connect("filesystem", MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path"],
        ))
        tools = await manager.discover_tools()
        registry.register_many(tools)
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConnection] = {}

    async def connect(self, name: str, config: MCPServerConfig) -> None:
        """Connect to an MCP server."""
        if name in self._servers:
            await self.disconnect(name)
        conn = MCPServerConnection(config)
        await conn.connect()
        self._servers[name] = conn

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        conn = self._servers.pop(name, None)
        if conn:
            await conn.disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in list(self._servers.keys()):
            await self.disconnect(name)

    async def discover_tools(self) -> List[Tool]:
        """Discover and wrap all tools from all connected servers."""
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

        Compatible with existing Geny MCP config format.
        """
        manager = cls()
        config_path = Path(path)
        if not config_path.exists():
            return manager

        with open(config_path, "r") as f:
            data = json.load(f)

        servers = data.get("mcpServers", data.get("servers", {}))
        for name, server_cfg in servers.items():
            config = MCPServerConfig(
                name=name,
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []),
                env=server_cfg.get("env", {}),
                transport=server_cfg.get("transport", "stdio"),
            )
            # Store config but don't connect yet — connect() is async
            manager._servers[name] = MCPServerConnection(config)

        return manager
