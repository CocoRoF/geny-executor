"""Default artifact routers for Stage 10: Tool."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.tools.base import ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s10_tool.interface import ToolRouter


class RegistryRouter(ToolRouter):
    """Routes tool calls via ToolRegistry lookup."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    @property
    def name(self) -> str:
        return "registry"

    @property
    def description(self) -> str:
        return "Routes via ToolRegistry lookup"

    async def route(
        self, tool_name: str, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(
                content=f"Unknown tool: {tool_name}",
                is_error=True,
            )
        try:
            return await tool.execute(tool_input, context)
        except Exception as e:
            return ToolResult(
                content=f"Tool '{tool_name}' failed: {str(e)}",
                is_error=True,
            )
