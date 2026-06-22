"""GlobTool — find files by pattern matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

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

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Directory-walk glob — no side effects, safe to fan out.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        search_path = input.get("path", "") or context.working_dir

        if not pattern:
            return ToolResult(content="pattern must not be empty", is_error=True)

        # Sandbox: expand the glob inside the container (bash globstar).
        if context.sandbox is not None:
            import shlex

            from geny_executor.tools._sandbox import sb_run

            wd = context.working_dir or "/workspace"
            spath = input.get("path", "") or "."
            # globstar makes ** recurse; nullglob makes a no-match expand to
            # nothing; print only regular files, newest first.
            cmd = (
                f"shopt -s globstar nullglob dotglob; cd {shlex.quote(spath)} 2>/dev/null || exit 0; "
                f"for f in {pattern}; do [ -f \"$f\" ] && printf '%s\\t%s\\n' \"$(stat -c %Y \"$f\" 2>/dev/null)\" \"$f\"; done "
                f"| sort -rn | cut -f2- | head -n {_MAX_RESULTS}"
            )
            rc, out, _err = await sb_run(context.sandbox, cmd, workdir=wd)
            out = out.strip()
            if not out:
                return ToolResult(
                    content=f"No files matching '{pattern}' in {search_path}"
                )
            return ToolResult(content=out)

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
