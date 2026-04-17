"""Tool Composer — manages built-in, ad-hoc, and MCP tools with presets.

Provides a unified interface for:
  - Registering/unregistering ad-hoc tools
  - Loading/saving named tool presets
  - Building filtered ToolRegistry instances for stages/sessions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry
from geny_executor.tools.adhoc import (
    AdhocTool,
    AdhocToolDefinition,
    AdhocToolFactory,
)


@dataclass
class ToolInfo:
    """Tool metadata for UI display."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    type: str  # "built_in" | "adhoc" | "mcp"
    source: str  # "geny-executor" | "mcp:server_name" | "user"
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    scope: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None  # AdhocToolDefinition dict
    mcp_server: Optional[str] = None
    execution_count: int = 0
    avg_execution_time_ms: float = 0.0
    error_rate: float = 0.0


@dataclass
class ToolPreset:
    """Named tool collection."""

    name: str
    description: str = ""
    tools: List[str] = field(default_factory=list)
    adhoc_tools: List[AdhocToolDefinition] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": self.tools,
            "adhoc_tools": [t.to_dict() for t in self.adhoc_tools],
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolPreset:
        adhoc_tools = [
            AdhocToolDefinition.from_dict(t)
            for t in data.get("adhoc_tools", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tools=data.get("tools", []),
            adhoc_tools=adhoc_tools,
            tags=data.get("tags", []),
        )


class ToolComposer:
    """Unified tool management: built-in + ad-hoc + MCP with presets."""

    def __init__(self, built_in_registry: Optional[ToolRegistry] = None):
        self._built_in = built_in_registry or ToolRegistry()
        self._adhoc_tools: Dict[str, AdhocTool] = {}
        self._mcp_tools: Dict[str, Tool] = {}  # name → MCPToolAdapter
        self._presets: Dict[str, ToolPreset] = dict(DEFAULT_PRESETS)

    # ── Ad-hoc tool management ──────────────────────────────

    def register_adhoc(self, definition: AdhocToolDefinition) -> AdhocTool:
        """Register an ad-hoc tool from a definition."""
        tool = AdhocToolFactory.create(
            definition, tool_resolver=self._resolve_tool
        )
        self._adhoc_tools[tool.name] = tool
        return tool

    def unregister_adhoc(self, name: str) -> bool:
        """Remove an ad-hoc tool. Returns True if found."""
        return self._adhoc_tools.pop(name, None) is not None

    def get_adhoc(self, name: str) -> Optional[AdhocTool]:
        return self._adhoc_tools.get(name)

    # ── MCP tool management ─────────────────────────────────

    def register_mcp_tool(self, tool: Tool, server_name: str = "") -> None:
        """Register an MCP-discovered tool."""
        self._mcp_tools[tool.name] = tool

    def unregister_mcp_tool(self, name: str) -> bool:
        return self._mcp_tools.pop(name, None) is not None

    def register_mcp_tools(self, tools: List[Tool], server_name: str = "") -> None:
        """Batch register MCP tools."""
        for t in tools:
            self._mcp_tools[t.name] = t

    def clear_mcp_tools(self) -> None:
        self._mcp_tools.clear()

    # ── Preset management ───────────────────────────────────

    def save_preset(self, preset: ToolPreset) -> None:
        self._presets[preset.name] = preset

    def load_preset(self, name: str) -> Optional[ToolPreset]:
        return self._presets.get(name)

    def delete_preset(self, name: str) -> bool:
        return self._presets.pop(name, None) is not None

    def list_presets(self) -> List[ToolPreset]:
        return list(self._presets.values())

    # ── Tool info ───────────────────────────────────────────

    def list_all_tools(self) -> List[ToolInfo]:
        """List metadata for all tools (built-in + ad-hoc + MCP)."""
        infos: List[ToolInfo] = []

        for tool in self._built_in.list_all():
            infos.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    type="built_in",
                    source="geny-executor",
                )
            )

        for tool in self._adhoc_tools.values():
            infos.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    type="adhoc",
                    source="user",
                    tags=tool.definition.tags,
                    definition=tool.definition.to_dict(),
                )
            )

        for tool in self._mcp_tools.values():
            infos.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    type="mcp",
                    source="mcp",
                )
            )

        return infos

    # ── Registry building ───────────────────────────────────

    def build_registry(
        self,
        include_built_in: Optional[Set[str]] = None,
        exclude_built_in: Optional[Set[str]] = None,
        include_adhoc: Optional[Set[str]] = None,
        include_mcp: Optional[Set[str]] = None,
    ) -> ToolRegistry:
        """Build a filtered ToolRegistry from all sources."""
        registry = ToolRegistry()

        # Built-in
        for tool in self._built_in.filter(
            include=include_built_in, exclude=exclude_built_in
        ):
            registry.register(tool)

        # Ad-hoc
        for name, tool in self._adhoc_tools.items():
            if include_adhoc is None or name in include_adhoc:
                registry.register(tool)

        # MCP
        for name, tool in self._mcp_tools.items():
            if include_mcp is None or name in include_mcp:
                registry.register(tool)

        return registry

    def build_registry_from_preset(self, preset_name: str) -> ToolRegistry:
        """Build a registry matching a named preset."""
        preset = self._presets.get(preset_name)
        if preset is None:
            raise KeyError(f"Preset '{preset_name}' not found")

        registry = ToolRegistry()

        # Built-in tools from preset
        tool_names = set(preset.tools)
        for tool in self._built_in.list_all():
            if tool.name in tool_names:
                registry.register(tool)

        # Ad-hoc tools from preset definition
        for defn in preset.adhoc_tools:
            tool = AdhocToolFactory.create(defn, tool_resolver=self._resolve_tool)
            registry.register(tool)

        # Also include any already-registered adhoc tools matching names
        for name, tool in self._adhoc_tools.items():
            if name in tool_names:
                registry.register(tool)

        return registry

    # ── Internal ────────────────────────────────────────────

    def _resolve_tool(self, name: str) -> Optional[Tool]:
        """Resolve a tool by name across all sources."""
        tool = self._built_in.get(name)
        if tool:
            return tool
        tool = self._adhoc_tools.get(name)
        if tool:
            return tool
        return self._mcp_tools.get(name)


# ── Default presets ─────────────────────────────────────────

DEFAULT_PRESETS: Dict[str, ToolPreset] = {
    "coding": ToolPreset(
        name="coding",
        description="Software development tools",
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        tags=["development"],
    ),
    "readonly": ToolPreset(
        name="readonly",
        description="Read-only analysis tools",
        tools=["Read", "Glob", "Grep"],
        tags=["safe"],
    ),
    "analysis": ToolPreset(
        name="analysis",
        description="Data analysis tools",
        tools=["Read", "Bash", "Grep"],
        tags=["data"],
    ),
    "web_agent": ToolPreset(
        name="web_agent",
        description="Web agent tools",
        tools=["Read", "Write", "Bash"],
        tags=["web"],
    ),
    "minimal": ToolPreset(
        name="minimal",
        description="No tools",
        tools=[],
        tags=["empty"],
    ),
}
