"""MCP (Model Context Protocol) integration."""

from geny_executor.tools.mcp.adapter import MCPToolAdapter
from geny_executor.tools.mcp.errors import MCPConnectionError
from geny_executor.tools.mcp.manager import MCPManager, MCPServerConfig

__all__ = [
    "MCPConnectionError",
    "MCPManager",
    "MCPServerConfig",
    "MCPToolAdapter",
]
