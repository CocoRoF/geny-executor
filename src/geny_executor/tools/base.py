"""Tool base class and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    """Context passed to tool execution.

    Attributes:
        session_id: Unique session identifier.
        working_dir: Working directory for file operations. Tools should
            resolve relative paths against this directory.
        storage_path: Session-specific storage directory (e.g. for logs,
            session state files). May differ from working_dir.
        env_vars: Environment variables to inject when spawning
            subprocesses (e.g. GITHUB_TOKEN, ANTHROPIC_API_KEY).
        allowed_paths: If set, tools MUST restrict file system access to
            these directories. An empty list means no restriction.
        metadata: Arbitrary key-value metadata forwarded to tools.
    """

    session_id: str = ""
    working_dir: str = ""
    storage_path: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    allowed_paths: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_order: int = 0
    stage_name: str = ""


@dataclass
class ToolResult:
    """Result of a tool execution."""

    content: Any = ""
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self, tool_use_id: str) -> Dict[str, Any]:
        """Convert to Anthropic API tool_result format.

        Structured-error payloads (``content`` is a dict with a top-level
        ``"error"`` object containing ``code`` and ``message``) are
        rendered with a leading ``ERROR <code>: <message>`` header line so
        the model has a predictable affordance to detect failure without
        parsing the JSON body.
        """
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
        }

        content = self.content
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            result["content"] = content
        elif isinstance(content, dict):
            import json as _json

            err_block = content.get("error")
            if (
                isinstance(err_block, dict)
                and isinstance(err_block.get("code"), str)
                and isinstance(err_block.get("message"), str)
            ):
                header = f"ERROR {err_block['code']}: {err_block['message']}"
                body = _json.dumps(content, ensure_ascii=False, default=str)
                result["content"] = f"{header}\n{body}"
            else:
                result["content"] = _json.dumps(content, ensure_ascii=False, default=str)
        else:
            result["content"] = str(content)

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
