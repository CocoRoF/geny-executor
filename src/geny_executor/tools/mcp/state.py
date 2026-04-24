"""Connection-state finite-state machine for MCP server lifecycles.

Cycle 20260424 executor uplift — Phase 6 Week 10-11.

The pre-Phase-6 manager tracked a server as either *connected* or *not*
— a single boolean. Phase 6 expands that into a five-state FSM so
hosts can distinguish recoverable failures (network glitch on a
remote SSE server) from terminal ones (auth challenge that needs
user input), and so an admin can disable a server without losing its
configuration.

States:

* ``PENDING`` — registered but not yet connected, OR previously
  failed/disabled and now waiting for a (re)connect attempt. The
  default state for a freshly added server.
* ``CONNECTED`` — handshake + ``initialize`` + ``list_tools`` all
  succeeded. The server contributes tools to the registry.
* ``FAILED`` — last connect attempt raised. The error is stashed on
  the connection so admins can inspect it. Reconnect is allowed
  (manual or scheduled retry).
* ``NEEDS_AUTH`` — a specific failure subtype: the server signalled
  that the credentials supplied (env var, OAuth bearer, …) are
  missing or invalid. Hosts treat this as a user-actionable prompt
  ("paste your token") rather than a transient retry.
* ``DISABLED`` — explicitly muted by an admin call. The configuration
  is retained so re-enabling is one call. ``DISABLED`` is the only
  state from which auto-reconnect logic should NOT fire.

Allowed transitions (host-driven; the FSM doesn't enforce them, just
documents them):

::

    PENDING    → CONNECTING (internal) → CONNECTED | FAILED | NEEDS_AUTH
    CONNECTED  → PENDING (admin reset / disconnect) → ...
    FAILED     → PENDING (manual retry) → ...
    NEEDS_AUTH → PENDING (after credentials supplied) → ...
    any        → DISABLED (admin disable_server)
    DISABLED   → PENDING (admin enable_server) → ...

See ``executor_uplift/12_detailed_plan.md`` §6 and
``executor_uplift/07_design_mcp_integration.md`` §4.
"""

from __future__ import annotations

from enum import Enum


class MCPConnectionState(str, Enum):
    """5-state lifecycle of an MCP server connection."""

    PENDING = "pending"
    """Registered but not currently connected. Reconnect attempts are
    allowed from this state."""

    CONNECTED = "connected"
    """Handshake + initialize + list_tools succeeded. The server's
    tools are visible to the registry."""

    FAILED = "failed"
    """Last connect attempt raised. ``connection.last_error`` carries
    the cause. Manual or scheduled reconnect is allowed (transitions
    back to ``PENDING`` first)."""

    NEEDS_AUTH = "needs_auth"
    """Server requested credentials we couldn't supply. Distinct from
    ``FAILED`` so hosts can prompt the user instead of retrying
    blindly."""

    DISABLED = "disabled"
    """Admin-muted. Configuration is retained but no reconnect
    attempts fire from this state. The only transition out is
    ``enable_server`` (→ ``PENDING``)."""

    @property
    def is_visible(self) -> bool:
        """True when the server should contribute tools to the registry.

        Only ``CONNECTED`` servers expose tools — ``PENDING`` /
        ``FAILED`` / ``NEEDS_AUTH`` haven't successfully discovered,
        ``DISABLED`` is explicitly muted.
        """
        return self is MCPConnectionState.CONNECTED

    @property
    def is_terminal(self) -> bool:
        """True when no automatic reconnect should be attempted.

        ``DISABLED`` is the only terminal state — ``FAILED`` /
        ``NEEDS_AUTH`` are recoverable with the right intervention.
        """
        return self is MCPConnectionState.DISABLED


# Convenience set for filtering.
RECONNECTABLE_STATES = frozenset(
    {
        MCPConnectionState.PENDING,
        MCPConnectionState.FAILED,
        MCPConnectionState.NEEDS_AUTH,
    }
)
