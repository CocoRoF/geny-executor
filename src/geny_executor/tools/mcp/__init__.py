"""MCP (Model Context Protocol) integration."""

from geny_executor.tools.mcp.manager import MCPManager, MCPServerConfig
from geny_executor.tools.mcp.adapter import MCPToolAdapter

__all__ = [
    "MCPManager",
    "MCPServerConfig",
    "MCPToolAdapter",
]
