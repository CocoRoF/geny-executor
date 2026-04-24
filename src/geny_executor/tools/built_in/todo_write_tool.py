"""TodoWrite — structured task list updates for the agent's own use.

Cycle 20260424 executor uplift — Phase 3 Week 6.

Mirrors the Claude Code ``TodoWrite`` pattern: the LLM maintains a
numbered list of in-progress work items and calls this tool to rewrite
the list whenever state changes. Each call replaces the entire list —
idempotent and easy to reason about. The tool validates shape, assigns
stable IDs (so the LLM can cross-reference a specific item later), and
returns a compact summary the model can include in its next prompt.

Design intent:

* The LLM is the source of truth for "what are the outstanding todos".
  This tool just persists the current view so it shows up in the
  conversation history as a tool_result block.
* No server-side storage yet — the todo list lives in the tool result
  the LLM sees. When the executor gains a task registry stage (Phase
  5, 21-stage layout), TodoWrite becomes a thin wrapper over that.
* Status values (``pending``, ``in_progress``, ``completed``) match
  Claude Code so a rebased plan survives.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 6 Workflow).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_VALID_STATUSES = ("pending", "in_progress", "completed")


def _stable_id(content: str, index: int) -> str:
    """Derive a short deterministic id from content + position.

    Used when the caller doesn't supply an explicit ``id``. The hash
    is truncated to 8 hex chars — collision-free enough for a
    per-turn todo list while staying LLM-readable.
    """
    digest = hashlib.sha1(f"{index}:{content}".encode("utf-8")).hexdigest()
    return digest[:8]


def _normalise_todo(raw: Any, index: int) -> Dict[str, Any]:
    """Coerce one raw dict into the canonical todo shape.

    Raises ``ValueError`` with a concrete message on bad input so the
    tool can surface a structured error to the LLM.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"todo #{index} must be an object, got {type(raw).__name__}")

    content = raw.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"todo #{index} missing non-empty string 'content'")

    status = raw.get("status", "pending")
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"todo #{index} has invalid status {status!r}; must be one of {_VALID_STATUSES}"
        )

    active_form = raw.get("activeForm")
    if active_form is not None and not isinstance(active_form, str):
        raise ValueError(f"todo #{index}: activeForm must be a string if provided")

    todo_id = raw.get("id")
    if todo_id is not None:
        if not isinstance(todo_id, str) or not todo_id.strip():
            raise ValueError(f"todo #{index}: id must be a non-empty string if provided")
    else:
        todo_id = _stable_id(content.strip(), index)

    normalised: Dict[str, Any] = {
        "id": todo_id.strip(),
        "content": content.strip(),
        "status": status,
    }
    if active_form is not None:
        normalised["activeForm"] = active_form.strip()
    return normalised


def _format_todos(todos: List[Dict[str, Any]]) -> str:
    """Render as a compact Markdown checklist for the LLM."""
    if not todos:
        return "(no todos)"

    status_markers = {
        "pending": "- [ ]",
        "in_progress": "- [~]",
        "completed": "- [x]",
    }
    lines = []
    for i, todo in enumerate(todos, 1):
        marker = status_markers[todo["status"]]
        text = todo["content"]
        if todo["status"] == "in_progress" and todo.get("activeForm"):
            text = f"{text} — {todo['activeForm']}"
        lines.append(f"{marker} {i}. [{todo['id']}] {text}")
    return "\n".join(lines)


def _counts(todos: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {status: 0 for status in _VALID_STATUSES}
    for todo in todos:
        counts[todo["status"]] += 1
    return counts


class TodoWriteTool(Tool):
    """Replace the agent's todo list with the one provided.

    Each call is a full rewrite: the caller always passes the complete
    current list. Empty lists are allowed (clears the board). The tool
    validates shape, assigns IDs where missing, and returns a
    human-readable summary + a structured metadata payload for any
    host that wants to render the list in a UI.
    """

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return (
            "Rewrite the agent's todo list. Pass the full list every "
            "call — this tool replaces the stored list wholesale. Use "
            "statuses 'pending' | 'in_progress' | 'completed'. Helpful "
            "for multi-step tasks where you want the plan visible in "
            "the conversation history."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": (
                        "Complete current todo list. Each entry is an "
                        "object with 'content' (required), 'status' "
                        "(default 'pending'), optional 'activeForm' "
                        "(present-continuous phrasing shown when status "
                        "is 'in_progress'), and optional 'id' (stable "
                        "identifier; derived from content when omitted)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Imperative-form todo text.",
                                "minLength": 1,
                            },
                            "status": {
                                "type": "string",
                                "enum": list(_VALID_STATUSES),
                                "description": "Todo status.",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": (
                                    "Present-continuous phrasing shown "
                                    "while the todo is in progress."
                                ),
                            },
                            "id": {
                                "type": "string",
                                "description": (
                                    "Stable identifier. Auto-derived from content when omitted."
                                ),
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["todos"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Pure in-turn operation: no network, no filesystem, no state.
        # But marking concurrency_safe=False because the agent usually
        # wants the todo list to reflect the very latest write — two
        # concurrent rewrites would race even if neither touches disk.
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=False,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raw_list = input.get("todos")
        if raw_list is None:
            return ToolResult(content="'todos' field is required", is_error=True)
        if not isinstance(raw_list, list):
            return ToolResult(
                content=f"'todos' must be a list, got {type(raw_list).__name__}",
                is_error=True,
            )

        # Enforce a soft cap so the tool result never balloons; mirrors
        # the max_result_chars protection for other built-ins.
        if len(raw_list) > 100:
            return ToolResult(
                content=f"too many todos: {len(raw_list)} > 100 limit",
                is_error=True,
            )

        try:
            normalised = [_normalise_todo(item, i) for i, item in enumerate(raw_list)]
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True)

        # Reject duplicate IDs — would break downstream tracking.
        seen: set[str] = set()
        for todo in normalised:
            if todo["id"] in seen:
                return ToolResult(
                    content=f"duplicate todo id {todo['id']!r}",
                    is_error=True,
                )
            seen.add(todo["id"])

        counts = _counts(normalised)
        rendered = _format_todos(normalised)
        summary = (
            f"todos updated: {len(normalised)} total "
            f"({counts['pending']} pending, "
            f"{counts['in_progress']} in progress, "
            f"{counts['completed']} completed)"
        )

        return ToolResult(
            content=f"{summary}\n\n{rendered}",
            metadata={
                "todos": normalised,
                "counts": counts,
                "total": len(normalised),
            },
            # Propose a state mutation so hosts that apply state_mutations
            # can read the list back off ``state.shared``. Key is
            # reserved under the executor namespace.
            state_mutations={"executor.todos": normalised},
        )
