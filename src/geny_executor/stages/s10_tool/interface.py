"""Stage 10: Tool — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.tools.base import ToolContext, ToolResult

ToolEventCallback = Callable[[str, Dict[str, Any]], None]


class ToolExecutor(Strategy):
    """Base interface for tool execution patterns."""

    @abstractmethod
    async def execute_all(
        self,
        tool_calls: List[Dict[str, Any]],
        router: ToolRouter,
        context: ToolContext,
        *,
        on_event: Optional[ToolEventCallback] = None,
    ) -> List[Dict[str, Any]]:
        """Execute all pending tool calls. Returns tool_result messages.

        ``on_event`` is an optional keyword-only callback invoked with
        ``(event_type, data)`` for per-call observability events:

        - ``tool.call_start`` — fires *before* each dispatch; carries
          ``{"tool_use_id", "name", "input"}``.
        - ``tool.call_complete`` — fires *after* each dispatch; carries
          ``{"tool_use_id", "name", "is_error", "duration_ms"}``.

        When ``on_event`` is ``None`` (default), no per-call events are
        emitted and behavior matches pre-0.23.0 semantics. Third-party
        executors implementing this protocol are not required to emit
        these events — the kwarg is optional.
        """
        ...


class ToolRouter(Strategy):
    """Base interface for routing tool calls to implementations."""

    @abstractmethod
    async def route(
        self, tool_name: str, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Route a tool call to its implementation and execute."""
        ...
