"""WriteTool — create or overwrite a file."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate


class WriteTool(Tool):
    """Write content to a file, creating parent directories as needed.

    Overwrites existing files. For partial modifications, use EditTool instead.
    """

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing files. For partial edits, use the Edit tool."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        content = input.get("content", "")

        try:
            resolved = resolve_and_validate(file_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            size = resolved.stat().st_size
            return ToolResult(content=f"Successfully wrote {size} bytes to {resolved}")
        except OSError as e:
            return ToolResult(content=f"Write error: {e}", is_error=True)
