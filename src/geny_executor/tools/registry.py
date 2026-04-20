"""Tool registry — registration, lookup, preset management."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from geny_executor.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for tools.

    Supports:
      - Registration by name
      - Preset-based filtering
      - API format export
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> ToolRegistry:
        """Register a tool. Chaining supported.

        Registering a tool under a name that already exists overrides the
        previous entry *and* emits a warning — this is almost always a
        bug (two providers claiming the same slot). MCP-sourced tools
        get their own ``mcp__{server}__{tool}`` namespace, so collisions
        here indicate a built-in / adhoc conflict or a duplicate
        registration pass.
        """
        existing = self._tools.get(tool.name)
        if existing is not None and existing is not tool:
            logger.warning(
                "tool name collision: '%s' re-registered "
                "(old=%s, new=%s) — the new entry wins",
                tool.name,
                type(existing).__name__,
                type(tool).__name__,
            )
        self._tools[tool.name] = tool
        return self

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def filter(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tool]:
        """Filter tools by name sets."""
        tools = self.list_all()
        if include is not None:
            tools = [t for t in tools if t.name in include]
        if exclude is not None:
            tools = [t for t in tools if t.name not in exclude]
        return tools

    def to_api_format(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Export tools in Anthropic API format."""
        tools = self.filter(include=include, exclude=exclude)
        return [t.to_api_format() for t in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
