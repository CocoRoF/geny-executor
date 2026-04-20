"""MCP-specific exception types for lifecycle failures."""

from __future__ import annotations

from typing import Optional


class MCPConnectionError(RuntimeError):
    """Raised when an MCP server cannot be connected, initialized, or
    queried for its tool list.

    The host surfaces this at session-start time (``Pipeline.from_manifest_async``)
    rather than letting the server enter a zombie "connected but no-op" state.
    Including the *server_name* and *phase* makes the failure site trivially
    debuggable in logs and in the UI.

    Attributes:
        server_name: The MCP server that failed (matches
            ``MCPServerConfig.name``).
        phase: Which lifecycle step blew up — one of ``"connect"`` /
            ``"initialize"`` / ``"list_tools"`` / ``"sdk_missing"``.
        cause: The underlying exception, preserved for logging.
    """

    def __init__(
        self,
        server_name: str,
        phase: str,
        *,
        cause: Optional[BaseException] = None,
        message: Optional[str] = None,
    ) -> None:
        self.server_name = server_name
        self.phase = phase
        self.cause = cause
        msg = message or self._default_message(server_name, phase, cause)
        super().__init__(msg)

    @staticmethod
    def _default_message(name: str, phase: str, cause: Optional[BaseException]) -> str:
        base = f"MCP server '{name}' failed during {phase}"
        if cause is not None:
            return f"{base}: {type(cause).__name__}: {cause}"
        return base
