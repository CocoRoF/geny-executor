"""Hook event taxonomy and payload schema.

The 13 event kinds cover session lifecycle, each stage's enter/exit,
tool invocation boundaries, permission decisions, and notification
channels. Hook runner (later checkpoint) dispatches subprocess hooks
matching these event names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class HookEvent(str, Enum):
    """User-configurable hook event kinds.

    Naming mirrors claude-code's hook system (PreToolUse / PostToolUse /
    etc.) so the pattern transfers directly — users who write hooks for
    claude-code can reuse the same scripts with minimal adaptation.
    """

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Pipeline lifecycle
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"

    # Stage boundaries (fired for every stage — payload includes stage_order + name)
    STAGE_ENTER = "stage_enter"
    STAGE_EXIT = "stage_exit"

    # User turn
    USER_PROMPT_SUBMIT = "user_prompt_submit"

    # Tool invocation — most common hook targets
    PRE_TOOL_USE = "pre_tool_use"  # After permission ALLOW, before execute()
    POST_TOOL_USE = "post_tool_use"  # After successful execute()
    POST_TOOL_FAILURE = "post_tool_failure"  # After execute() raised or is_error

    # Permission
    PERMISSION_REQUEST = "permission_request"  # PermissionDecision.ASK fired
    PERMISSION_DENIED = "permission_denied"  # PermissionDecision.DENY fired

    # Loop
    LOOP_ITERATION_END = "loop_iteration_end"

    # Environment
    CWD_CHANGED = "cwd_changed"

    # MCP
    MCP_SERVER_STATE = "mcp_server_state"  # FSM transition (Phase 6)

    # Generic
    NOTIFICATION = "notification"  # Host-initiated, user-readable


@dataclass
class HookEventPayload:
    """Data passed to a hook's stdin as JSON.

    Minimum surface that any hook can rely on. Event-specific fields
    live under ``details`` to keep the top-level schema stable.

    Attributes:
        event: The ``HookEvent`` kind.
        session_id: Session identifier (stable across reconnects).
        timestamp: ISO-8601 UTC timestamp of when the event fired.
        pipeline_id: Pipeline instance ID (changes across restarts).
        permission_mode: Current session permission mode.
        stage_order: For stage events — stage order (1-21 post Phase 9).
        stage_name: For stage events — stage name (e.g. ``"tool"``).
        tool_name: For tool events — tool name.
        tool_input: For tool events — input payload (may be redacted).
        tool_output: For POST_TOOL_USE — tool result preview (truncated
            to ``ToolCapabilities.max_result_chars``).
        details: Event-specific free-form bag. Hook scripts MUST tolerate
            unknown keys to remain forward-compatible.
    """

    event: HookEvent
    session_id: str
    timestamp: str
    pipeline_id: Optional[str] = None
    permission_mode: str = "default"
    stage_order: Optional[int] = None
    stage_name: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for ``json.dumps`` → hook stdin."""
        out: Dict[str, Any] = {
            "event": self.event.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "permission_mode": self.permission_mode,
        }
        if self.pipeline_id is not None:
            out["pipeline_id"] = self.pipeline_id
        if self.stage_order is not None:
            out["stage_order"] = self.stage_order
        if self.stage_name is not None:
            out["stage_name"] = self.stage_name
        if self.tool_name is not None:
            out["tool_name"] = self.tool_name
        if self.tool_input is not None:
            out["tool_input"] = self.tool_input
        if self.tool_output is not None:
            out["tool_output"] = self.tool_output
        if self.details:
            out["details"] = self.details
        return out


@dataclass(frozen=True)
class HookOutcome:
    """Result of running a hook.

    Attributes:
        continue_: Whether the pipeline should continue. ``False`` blocks
            the in-flight operation (e.g. cancels the tool before
            execute()). Underscore suffix to avoid Python keyword.
        suppress_output: If True the orchestrator hides the tool output
            from the conversation (still persisted for audit).
        decision: Overrides a permission decision when provided:
            ``"approve"`` forces ALLOW, ``"block"`` forces DENY, ``None``
            leaves the engine's existing verdict untouched.
        stop_reason: Optional human-readable message, logged + shown
            alongside any block/deny action.
        modified_input: When set, the tool is invoked with this payload
            instead of the original. Only applied for PRE_TOOL_USE.
        hook_specific_output: Free-form data returned by the hook;
            some event kinds surface this to downstream stages (e.g.
            NOTIFICATION hooks can attach UI hints here).
    """

    continue_: bool = True
    suppress_output: bool = False
    decision: Optional[str] = None  # 'approve' | 'block' | None
    stop_reason: Optional[str] = None
    modified_input: Optional[Dict[str, Any]] = None
    hook_specific_output: Optional[Dict[str, Any]] = None

    @classmethod
    def passthrough(cls) -> "HookOutcome":
        """Default no-op outcome."""
        return cls()

    @classmethod
    def block(cls, reason: str) -> "HookOutcome":
        """Block the operation with an explanation."""
        return cls(continue_=False, decision="block", stop_reason=reason)

    @classmethod
    def approve(cls, reason: Optional[str] = None) -> "HookOutcome":
        """Override a pending permission request as approved."""
        return cls(decision="approve", stop_reason=reason)

    @classmethod
    def from_response(cls, resp: Dict[str, Any]) -> "HookOutcome":
        """Parse a hook's JSON stdout into an outcome.

        Tolerates missing keys. Unknown keys are ignored.
        """
        return cls(
            continue_=bool(resp.get("continue", True)),
            suppress_output=bool(resp.get("suppress_output", False)),
            decision=resp.get("decision") if isinstance(resp.get("decision"), str) else None,
            stop_reason=resp.get("stop_reason") if isinstance(resp.get("stop_reason"), str) else None,
            modified_input=resp.get("modified_input") if isinstance(resp.get("modified_input"), dict) else None,
            hook_specific_output=resp.get("hook_specific_output") if isinstance(resp.get("hook_specific_output"), dict) else None,
        )

    def combine(self, other: "HookOutcome") -> "HookOutcome":
        """Merge two outcomes — the more restrictive wins.

        Multiple hooks can run for the same event (e.g. audit + gate).
        Combination rules:
        - ``continue_`` — AND (any blocker wins)
        - ``suppress_output`` — OR (any hook wanting to hide wins)
        - ``decision`` — 'block' > 'approve' > None
        - ``stop_reason`` — first non-None
        - ``modified_input`` — later hook's value (last writer wins)
        - ``hook_specific_output`` — merged dict (other wins on conflict)
        """
        decision_priority = {"block": 2, "approve": 1, None: 0}
        decision = (
            self.decision
            if decision_priority.get(self.decision, 0) >= decision_priority.get(other.decision, 0)
            else other.decision
        )
        merged_hso: Optional[Dict[str, Any]] = None
        if self.hook_specific_output or other.hook_specific_output:
            merged_hso = dict(self.hook_specific_output or {})
            merged_hso.update(other.hook_specific_output or {})
        return HookOutcome(
            continue_=self.continue_ and other.continue_,
            suppress_output=self.suppress_output or other.suppress_output,
            decision=decision,
            stop_reason=self.stop_reason or other.stop_reason,
            modified_input=other.modified_input if other.modified_input is not None else self.modified_input,
            hook_specific_output=merged_hso,
        )

    @property
    def blocked(self) -> bool:
        """True if the operation must not proceed."""
        return not self.continue_ or self.decision == "block"
