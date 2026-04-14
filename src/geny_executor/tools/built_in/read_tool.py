"""ReadTool — read file contents with line numbers."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate

_DEFAULT_LIMIT = 2000


class ReadTool(Tool):
    """Read a file from the local filesystem.

    Returns content with line numbers (cat -n format).
    Detects binary files and refuses to read them (except images).
    """

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return (
            "Read a file from the filesystem. Returns content with line numbers. "
            "Use offset and limit to read specific portions of large files."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to read. Default: {_DEFAULT_LIMIT}.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        offset = input.get("offset", 0)
        limit = input.get("limit", _DEFAULT_LIMIT)

        try:
            resolved = resolve_and_validate(file_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        if not resolved.exists():
            return ToolResult(content=f"File not found: {resolved}", is_error=True)
        if resolved.is_dir():
            return ToolResult(content=f"Cannot read directory: {resolved}. Use Bash with 'ls' instead.", is_error=True)

        # Binary detection
        mime, _ = mimetypes.guess_type(str(resolved))
        if mime and mime.startswith("image/"):
            # For images, return a placeholder (base64 in production)
            size = resolved.stat().st_size
            return ToolResult(content=f"[Image file: {resolved.name}, {size} bytes, type={mime}]")

        try:
            raw = resolved.read_bytes()
        except OSError as e:
            return ToolResult(content=f"Read error: {e}", is_error=True)

        # Binary guard
        if b"\x00" in raw[:8192]:
            size = len(raw)
            return ToolResult(content=f"[Binary file: {resolved.name}, {size} bytes]")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except Exception:
                return ToolResult(content=f"[Binary file: {resolved.name}, {len(raw)} bytes]")

        lines = text.splitlines(keepends=True)
        total = len(lines)

        selected = lines[offset : offset + limit]
        if not selected and total > 0:
            return ToolResult(content=f"Offset {offset} is beyond file end ({total} lines).", is_error=True)

        # Format with line numbers (1-based display)
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i}\t{line.rstrip()}")

        output = "\n".join(numbered)

        if offset + limit < total:
            output += f"\n\n... ({total - offset - limit} more lines, {total} total)"

        return ToolResult(content=output)
