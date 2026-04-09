"""MCP tool adapter — wraps MCP server tools as Tool interface."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from geny_executor.tools.base import Tool, ToolResult, ToolContext

if TYPE_CHECKING:
    from geny_executor.tools.mcp.manager import MCPServerConnection


class MCPToolAdapter(Tool):
    """Wraps an MCP server tool as a geny-executor Tool.

    Bridges the MCP protocol to our Tool interface.
    """

    def __init__(
        self,
        server: MCPServerConnection,
        definition: Dict[str, Any],
    ):
        self._server = server
        self._definition = definition

    @property
    def name(self) -> str:
        return self._definition.get("name", "unknown_mcp_tool")

    @property
    def description(self) -> str:
        return self._definition.get("description", "MCP tool")

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._definition.get(
            "inputSchema",
            self._definition.get(
                "input_schema",
                {
                    "type": "object",
                    "properties": {},
                },
            ),
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool via MCP server."""
        try:
            result = await self._server.call_tool(self.name, input)
            return ToolResult(
                content=result if isinstance(result, str) else str(result),
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                content=str(e),
                is_error=True,
            )
