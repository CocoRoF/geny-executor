"""Structured error types for tool dispatch.

The router (``RegistryRouter``) converts every failure mode into a
``ToolError`` with a stable ``code`` so the model — and any downstream
consumer — can reason about the failure without parsing free-form
English. Tool implementations that want to signal a structured failure
raise ``ToolFailure`` from ``execute`` (or ``run`` on the Geny side);
the router catches it and emits the matching ``ToolError``.

No string fallbacks are kept here; the old ``"Unknown tool: X"`` /
``"Tool 'X' failed: ..."`` strings are gone as of v0.22.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

if TYPE_CHECKING:
    from geny_executor.tools.base import ToolResult


class ToolErrorCode(str, Enum):
    """Stable identifiers for tool failure modes."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_INPUT = "invalid_input"
    TOOL_CRASHED = "tool_crashed"
    TRANSPORT = "transport_error"
    ACCESS_DENIED = "access_denied"


@dataclass(frozen=True)
class ToolError:
    """Structured description of a tool failure.

    Keep ``message`` concise — it's surfaced on the first line of the
    tool_result so the model can pattern-match it. Put everything else
    (paths, expected values, server name, …) in ``details``.
    """

    code: ToolErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Return the wire representation used inside ``ToolResult.content``."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": dict(self.details),
            }
        }

    @classmethod
    def unknown_tool(cls, name: str, *, known: Optional[Iterable[str]] = None) -> "ToolError":
        details: Dict[str, Any] = {"tool_name": name}
        if known is not None:
            details["known_tools"] = sorted(known)
        return cls(
            code=ToolErrorCode.UNKNOWN_TOOL,
            message=f"Unknown tool: {name}",
            details=details,
        )

    @classmethod
    def invalid_input(
        cls, tool_name: str, reason: str, *, path: Optional[str] = None
    ) -> "ToolError":
        details: Dict[str, Any] = {"tool_name": tool_name, "reason": reason}
        if path is not None:
            details["path"] = path
        return cls(
            code=ToolErrorCode.INVALID_INPUT,
            message=f"Invalid input for '{tool_name}': {reason}",
            details=details,
        )

    @classmethod
    def tool_crashed(cls, tool_name: str, exc: BaseException) -> "ToolError":
        return cls(
            code=ToolErrorCode.TOOL_CRASHED,
            message=f"Tool '{tool_name}' crashed: {type(exc).__name__}: {exc}",
            details={
                "tool_name": tool_name,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    @classmethod
    def access_denied(
        cls, tool_name: str, reason: str = "binding disallows this tool"
    ) -> "ToolError":
        return cls(
            code=ToolErrorCode.ACCESS_DENIED,
            message=f"Access denied for '{tool_name}': {reason}",
            details={"tool_name": tool_name, "reason": reason},
        )

    @classmethod
    def transport(cls, server_name: str, reason: str) -> "ToolError":
        return cls(
            code=ToolErrorCode.TRANSPORT,
            message=f"MCP transport error on '{server_name}': {reason}",
            details={"server": server_name, "reason": reason},
        )


class ToolFailure(Exception):
    """Raised by tool implementations to report a structured failure.

    Preferred over returning a JSON blob with an ``"error"`` field —
    the router bridges this into a ``ToolError`` with the given
    ``code`` (default ``TOOL_CRASHED``) and preserves ``details``.

    Example::

        raise ToolFailure(
            "rate limit exceeded",
            code=ToolErrorCode.TRANSPORT,
            details={"retry_after": 30},
        )
    """

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.TOOL_CRASHED,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error = ToolError(code=code, message=message, details=details or {})


def make_error_result(err: ToolError) -> "ToolResult":
    """Wrap a ``ToolError`` into a ``ToolResult`` ready for the API layer.

    Kept out of ``base.py`` to avoid a circular import; routers call
    this helper from within ``stages.s10_tool``.
    """
    from geny_executor.tools.base import ToolResult

    return ToolResult(
        content=err.to_payload(),
        is_error=True,
        metadata={"error_code": err.code.value},
    )


def validate_input(schema: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Validate *payload* against JSON Schema *schema*.

    Raises ``jsonschema.ValidationError`` on failure. Returns ``None``
    on success. The caller (router) converts validation errors into
    ``ToolError.invalid_input``.

    ``jsonschema`` is a required dependency as of v0.22.0.
    """
    import jsonschema

    jsonschema.validate(instance=payload, schema=schema)
