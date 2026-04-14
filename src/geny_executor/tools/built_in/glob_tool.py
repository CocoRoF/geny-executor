"""GlobTool — find files by pattern matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolContext, ToolResult

_MAX_RESULTS = 500


class GlobTool(Tool):
    """Find files matching a glob pattern.

    Searches from the given directory (or working_dir) and returns
    matching file paths sorted by modification time (newest first).
    """

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). "
            "Returns matching file paths sorted by modification time."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files against.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory.",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        search_path = input.get("path", "") or context.working_dir

        if not pattern:
            return ToolResult(content="pattern must not be empty", is_error=True)

        base = Path(search_path)
        if not base.is_dir():
            return ToolResult(content=f"Directory not found: {search_path}", is_error=True)

        try:
            matches = list(base.glob(pattern))
        except Exception as e:
            return ToolResult(content=f"Glob error: {e}", is_error=True)

        # Filter to files only, sort by mtime descending
        files = [m for m in matches if m.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not files:
            return ToolResult(content=f"No files matching '{pattern}' in {search_path}")

        truncated = len(files) > _MAX_RESULTS
        if truncated:
            files = files[:_MAX_RESULTS]

        output = "\n".join(str(f) for f in files)
        if truncated:
            output += f"\n\n... (showing {_MAX_RESULTS} of {len(matches)} matches)"

        return ToolResult(content=output)
