"""BashTool — execute shell commands."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolContext, ToolResult

_DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes
_MAX_TIMEOUT_MS = 600_000  # 10 minutes
_MAX_OUTPUT = 100_000  # characters


class BashTool(Tool):
    """Execute a bash command and return stdout/stderr.

    Commands run in the session's working directory with configurable
    timeout and environment variable injection.
    """

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Returns stdout, stderr, and exit code. "
            "Commands run in the working directory with a configurable timeout."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in milliseconds (default: {_DEFAULT_TIMEOUT_MS}, max: {_MAX_TIMEOUT_MS}).",
                    "minimum": 1000,
                    "maximum": _MAX_TIMEOUT_MS,
                },
            },
            "required": ["command"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "").strip()
        if not command:
            return ToolResult(content="command must not be empty", is_error=True)

        timeout_ms = min(input.get("timeout", _DEFAULT_TIMEOUT_MS), _MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        cwd = context.working_dir or None

        # Build environment: inherit current env + inject context env_vars
        env = os.environ.copy()
        if context.env_vars:
            env.update(context.env_vars)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except OSError as e:
            return ToolResult(content=f"Failed to start process: {e}", is_error=True)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                content=f"Command timed out after {timeout_ms}ms",
                is_error=True,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        # Truncate very large output
        if len(stdout) > _MAX_OUTPUT:
            stdout = stdout[:_MAX_OUTPUT] + f"\n\n... (truncated, {len(stdout_bytes)} bytes total)"
        if len(stderr) > _MAX_OUTPUT:
            stderr = stderr[:_MAX_OUTPUT] + f"\n\n... (truncated, {len(stderr_bytes)} bytes total)"

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        if exit_code != 0:
            parts.append(f"Exit code: {exit_code}")

        output = "\n".join(parts) if parts else "(no output)"

        return ToolResult(
            content=output,
            is_error=exit_code != 0,
            metadata={"exit_code": exit_code},
        )
