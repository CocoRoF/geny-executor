"""Environment Import Validator — validates environment JSON before import.

Enforces size limits, stage count limits, tool count limits,
and runs script security checks on ad-hoc script tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.security.script_sandbox import validate_script


class ImportValidationError(Exception):
    """Raised when import data fails validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Import validation failed: {errors}")


# ── Limits ────────────────────────────────────────────────

MAX_JSON_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_STAGES = 32
MAX_ADHOC_TOOLS = 100
MAX_MCP_SERVERS = 20
MAX_SCRIPT_LENGTH = 10_000
SUPPORTED_VERSIONS = frozenset({"1.0"})

# Commands considered too dangerous for MCP stdio transport.
DANGEROUS_COMMANDS = frozenset({
    "rm", "dd", "mkfs", "kill", "shutdown", "reboot",
    "fdisk", "format", "del", "rmdir",
})


def validate_import(data: Dict[str, Any], *, raw_size: Optional[int] = None) -> List[str]:
    """Validate environment data for safe import.

    Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []

    # ── Raw size ──
    if raw_size is not None and raw_size > MAX_JSON_SIZE:
        errors.append(
            f"File too large: {raw_size:,} bytes (max {MAX_JSON_SIZE:,})"
        )

    # ── Version ──
    version = data.get("version", data.get("metadata", {}).get("version"))
    if version and str(version) not in SUPPORTED_VERSIONS:
        errors.append(f"Unsupported version: {version}")

    # ── Stages ──
    stages = data.get("stages", data.get("pipeline", {}).get("stages", []))
    if isinstance(stages, list) and len(stages) > MAX_STAGES:
        errors.append(f"Too many stages: {len(stages)} (max {MAX_STAGES})")

    # ── Tools ──
    tools = data.get("tools", {})
    if isinstance(tools, dict):
        _validate_adhoc_tools(tools.get("adhoc", []), errors)
        _validate_mcp_servers(tools.get("mcp_servers", []), errors)

    return errors


def _validate_adhoc_tools(adhoc: Any, errors: List[str]) -> None:
    if not isinstance(adhoc, list):
        return
    if len(adhoc) > MAX_ADHOC_TOOLS:
        errors.append(f"Too many adhoc tools: {len(adhoc)} (max {MAX_ADHOC_TOOLS})")

    for tool in adhoc:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "<unnamed>")

        # Script security check
        if tool.get("executor_type") == "script":
            code = (tool.get("script_config") or {}).get("code", "")
            if len(code) > MAX_SCRIPT_LENGTH:
                errors.append(
                    f"Script too long in tool '{name}': "
                    f"{len(code)} chars (max {MAX_SCRIPT_LENGTH})"
                )
            if code:
                violations = validate_script(code)
                if violations:
                    errors.append(
                        f"Script security issue in tool '{name}': {violations}"
                    )


def _validate_mcp_servers(servers: Any, errors: List[str]) -> None:
    if not isinstance(servers, list):
        return
    if len(servers) > MAX_MCP_SERVERS:
        errors.append(f"Too many MCP servers: {len(servers)} (max {MAX_MCP_SERVERS})")

    for server in servers:
        if not isinstance(server, dict):
            continue
        name = server.get("name", "<unnamed>")

        if server.get("transport") == "stdio":
            command = server.get("command", "")
            if isinstance(command, str):
                binary = command.split("/")[-1].split()[0] if command else ""
                if binary in DANGEROUS_COMMANDS:
                    errors.append(
                        f"Dangerous MCP command in server '{name}': {binary}"
                    )


def check_import(data: Dict[str, Any], *, raw_size: Optional[int] = None) -> None:
    """Raise :class:`ImportValidationError` if data is invalid."""
    errors = validate_import(data, raw_size=raw_size)
    if errors:
        raise ImportValidationError(errors)
