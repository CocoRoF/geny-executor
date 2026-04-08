"""Tool base class and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    session_id: str = ""
    working_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    content: Any = ""
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self, tool_use_id: str) -> Dict[str, Any]:
        """Convert to Anthropic API tool_result format."""
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
        }
        if isinstance(self.content, str):
            result["content"] = self.content
        elif isinstance(self.content, list):
            result["content"] = self.content
        else:
            result["content"] = str(self.content)

        if self.is_error:
            result["is_error"] = True

        return result


class Tool(ABC):
    """Tool interface — maps 1:1 to Anthropic API tool definitions.

    Implement this to create custom tools that Claude can call.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool unique name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the model."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON Schema for tool input parameters."""
        ...

    @abstractmethod
    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with given input."""
        ...

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API tools parameter format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
