"""WriteTool — create or overwrite a file."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in._file_witness import (
    is_witnessed,
    path_forms,
    refusal,
    witnessed_mutation,
)
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
        forms = path_forms(file_path, context)
        # An existing file the agent has not seen is not replaced blindly
        # (_file_witness). New or empty files pass.
        if not is_witnessed(context.state_view, forms):
            existing = await self._existing_size(file_path, context)
            if existing is None:
                return ToolResult(
                    content=f"Cannot write: {file_path} is a directory.", is_error=True
                )
            if existing > 0:
                return ToolResult(content=refusal(forms[-1] if forms else file_path), is_error=True)
        result = await self._write(input, context)
        if not result.is_error:
            # The agent knows what it just wrote: Write → Write is fine.
            result.state_mutations = {
                **(result.state_mutations or {}),
                **witnessed_mutation(forms),
            }
        return result

    @staticmethod
    async def _existing_size(file_path: str, context: ToolContext) -> Optional[int]:
        """Bytes in the file now; 0 when absent or unreadable; None for a directory."""
        if context.sandbox is not None:
            from geny_executor.tools._sandbox import sb_read_bytes

            try:
                raw = await sb_read_bytes(
                    context.sandbox, file_path, workdir=context.working_dir or "/workspace"
                )
            except FileNotFoundError:
                return 0
            except Exception as exc:  # noqa: BLE001 — unreadable: nothing to lose
                return None if "directory" in str(exc).lower() else 0
            return len(raw)
        try:
            resolved = resolve_and_validate(file_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError):
            return 0  # the write itself reports this
        if resolved.is_dir():
            return None
        try:
            return resolved.stat().st_size if resolved.exists() else 0
        except OSError:
            return 0

    async def _write(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        content = input.get("content", "")

        # Sandbox: write the file inside the container (docker exec).
        if context.sandbox is not None:
            from geny_executor.tools._sandbox import sb_write_bytes, spoken_path

            wd = context.working_dir or "/workspace"
            try:
                n = await sb_write_bytes(
                    context.sandbox, file_path, content.encode("utf-8"), workdir=wd
                )
                where = spoken_path(context.sandbox, file_path, wd)
                return ToolResult(content=f"Successfully wrote {n} bytes to {where}")
            except PermissionError as e:
                return ToolResult(content=str(e), is_error=True)
            except Exception as e:  # noqa: BLE001
                return ToolResult(content=f"Write error: {e}", is_error=True)

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
