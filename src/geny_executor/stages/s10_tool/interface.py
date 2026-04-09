"""Stage 10: Tool — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from geny_executor.core.stage import Strategy
from geny_executor.tools.base import ToolContext, ToolResult


class ToolExecutor(Strategy):
    """Base interface for tool execution patterns."""

    @abstractmethod
    async def execute_all(
        self,
        tool_calls: List[Dict[str, Any]],
        router: ToolRouter,
        context: ToolContext,
    ) -> List[Dict[str, Any]]:
        """Execute all pending tool calls. Returns tool_result messages."""
        ...


class ToolRouter(Strategy):
    """Base interface for routing tool calls to implementations."""

    @abstractmethod
    async def route(
        self, tool_name: str, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Route a tool call to its implementation and execute."""
        ...
