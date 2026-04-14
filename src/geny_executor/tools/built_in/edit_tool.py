"""EditTool — perform exact string replacements in files."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate


class EditTool(Tool):
    """Replace exact string occurrences in a file.

    By default, old_string must appear exactly once (for safety).
    Set replace_all=True to replace every occurrence.
    """

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return (
            "Perform exact string replacements in a file. "
            "old_string must be unique in the file unless replace_all is true."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to modify.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false).",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")
        replace_all = input.get("replace_all", False)

        if not old_string:
            return ToolResult(content="old_string must not be empty", is_error=True)
        if old_string == new_string:
            return ToolResult(content="old_string and new_string must be different", is_error=True)

        try:
            resolved = resolve_and_validate(file_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        if not resolved.exists():
            return ToolResult(content=f"File not found: {resolved}", is_error=True)

        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult(content=f"Read error: {e}", is_error=True)

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                content="old_string not found in file. Ensure the string matches exactly, including whitespace and indentation.",
                is_error=True,
            )

        if not replace_all and count > 1:
            return ToolResult(
                content=f"old_string appears {count} times in file. Provide more context to make it unique, or set replace_all=true.",
                is_error=True,
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult(content=f"Write error: {e}", is_error=True)

        return ToolResult(
            content=f"Successfully edited {resolved} ({count} replacement{'s' if count > 1 else ''})"
        )
