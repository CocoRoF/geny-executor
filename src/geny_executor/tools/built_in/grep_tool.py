"""GrepTool — search file contents with regex."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from geny_executor.tools.base import Tool, ToolContext, ToolResult

_MAX_FILES = 200
_MAX_MATCHES = 300
_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB


class GrepTool(Tool):
    """Search file contents using regular expressions.

    Supports filtering by file glob pattern and multiple output modes.
    """

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with regex. Supports filtering by glob pattern. "
            "Output modes: 'content' (matching lines), 'files' (file paths only), 'count' (match counts)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}').",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files", "count"],
                    "description": "Output mode. Default: 'files'.",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context to show around matches (for 'content' mode).",
                    "minimum": 0,
                    "maximum": 10,
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search. Default: false.",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern_str = input.get("pattern", "")
        search_path = input.get("path", "") or context.working_dir
        file_glob = input.get("glob")
        output_mode = input.get("output_mode", "files")
        ctx_lines = input.get("context", 0)
        case_insensitive = input.get("case_insensitive", False)

        if not pattern_str:
            return ToolResult(content="pattern must not be empty", is_error=True)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern_str, flags)
        except re.error as e:
            return ToolResult(content=f"Invalid regex: {e}", is_error=True)

        base = Path(search_path)
        if not base.exists():
            return ToolResult(content=f"Path not found: {search_path}", is_error=True)

        # Collect target files
        if base.is_file():
            targets = [base]
        else:
            if file_glob:
                targets = sorted(base.rglob(file_glob))
            else:
                targets = sorted(base.rglob("*"))
            targets = [t for t in targets if t.is_file()]

        # Skip hidden dirs and common noise
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
        filtered: List[Path] = []
        for t in targets:
            parts = set(t.relative_to(base).parts[:-1]) if base.is_dir() else set()
            if not parts.intersection(skip_dirs):
                filtered.append(t)
        targets = filtered[:_MAX_FILES]

        match_files: List[str] = []
        match_lines: List[str] = []
        total_matches = 0

        for fpath in targets:
            if fpath.stat().st_size > _MAX_FILE_SIZE:
                continue

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            lines = text.splitlines()
            file_matches = []

            for i, line in enumerate(lines):
                if regex.search(line):
                    file_matches.append((i, line))

            if not file_matches:
                continue

            match_files.append(str(fpath))
            total_matches += len(file_matches)

            if output_mode == "content":
                for line_num, line_text in file_matches:
                    if total_matches > _MAX_MATCHES:
                        break
                    # Context lines
                    start = max(0, line_num - ctx_lines)
                    end = min(len(lines), line_num + ctx_lines + 1)
                    for ci in range(start, end):
                        prefix = ">" if ci == line_num else " "
                        match_lines.append(f"{fpath}:{ci + 1}:{prefix} {lines[ci]}")
                    if ctx_lines > 0:
                        match_lines.append("--")

        if output_mode == "files":
            if not match_files:
                return ToolResult(content=f"No matches for '{pattern_str}'")
            output = "\n".join(match_files)
            if len(match_files) >= _MAX_FILES:
                output += f"\n\n... (limited to {_MAX_FILES} files)"
            return ToolResult(content=output)

        elif output_mode == "count":
            return ToolResult(content=f"{total_matches} matches in {len(match_files)} files")

        else:  # content
            if not match_lines:
                return ToolResult(content=f"No matches for '{pattern_str}'")
            output = "\n".join(match_lines[: _MAX_MATCHES * 3])
            if total_matches > _MAX_MATCHES:
                output += f"\n\n... ({total_matches} total matches, showing first {_MAX_MATCHES})"
            return ToolResult(content=output)
