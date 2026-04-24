"""EnterPlanMode / ExitPlanMode — toggle restricted-operation flag.

Cycle 20260424 executor uplift — Phase 3 Week 6.

Plan mode is a soft guardrail: while active, the agent should plan
before acting — most hosts wire it to Stage 4 Guard (future) to block
destructive tools and require the LLM to surface a plan first. These
tools don't enforce that themselves; they just flip a well-known flag
on ``state.shared`` so downstream stages / tools can inspect it.

The flag name (``executor.plan_mode``) and its shape (``bool``) are
considered part of the public state contract. Hosts subscribing to the
flag via the Phase 1 permission / guard matrix should look for this
key.

Two tools ship as a pair — the LLM opts in, then opts back out once
the plan is approved or abandoned:

* ``EnterPlanMode({"reason": "..."})`` — sets the flag to ``True`` and
  records the (optional) reason in metadata.
* ``ExitPlanMode({"reason": "..."})`` — clears the flag and records the
  reason.

Both are idempotent: calling Enter twice leaves the state as True;
calling Exit while already off is a no-op with a benign notice.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 6 Meta).
"""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

# Public contract — hosts subscribing to plan mode must use this exact key.
PLAN_MODE_KEY = "executor.plan_mode"


def _current_mode(context: ToolContext) -> bool:
    """Inspect ``state_view`` for the current plan-mode flag.

    Returns ``False`` when the view is absent or the key is unset — the
    "off" state is always the safe default for a flag that gates
    destructive tools.
    """
    view = getattr(context, "state_view", None)
    if view is None:
        return False
    shared = getattr(view, "shared", None)
    if not isinstance(shared, dict):
        return False
    return bool(shared.get(PLAN_MODE_KEY, False))


class _PlanModeBase(Tool):
    """Shared machinery — concrete tools override ``_desired`` + name."""

    _desired: bool = False

    @property
    def description(self) -> str:
        if self._desired:
            return (
                "Signal that the agent is entering plan mode — describe "
                "the plan before taking actions. Flips "
                f"state.shared[{PLAN_MODE_KEY!r}] to True."
            )
        return (
            "Signal that the agent is leaving plan mode and may act "
            f"normally. Flips state.shared[{PLAN_MODE_KEY!r}] to False."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional short note for the audit trail — "
                        "why the agent is entering / leaving plan mode."
                    ),
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=False,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        reason = input.get("reason") or ""
        if not isinstance(reason, str):
            return ToolResult(content="'reason' must be a string", is_error=True)
        reason = reason.strip()

        was = _current_mode(context)
        will_change = was != self._desired

        verb = "entered" if self._desired else "exited"
        headline = f"plan mode {verb}"
        if not will_change:
            headline = f"plan mode already {('on' if self._desired else 'off')} — no change"

        lines = [headline]
        if reason:
            lines.append(f"reason: {reason}")

        return ToolResult(
            content="\n".join(lines),
            metadata={
                "plan_mode": self._desired,
                "was": was,
                "changed": will_change,
                "reason": reason or None,
            },
            state_mutations={PLAN_MODE_KEY: self._desired},
        )


class EnterPlanModeTool(_PlanModeBase):
    _desired = True

    @property
    def name(self) -> str:
        return "EnterPlanMode"


class ExitPlanModeTool(_PlanModeBase):
    _desired = False

    @property
    def name(self) -> str:
        return "ExitPlanMode"
