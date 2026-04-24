"""Permission matrix evaluation — the single entry point.

Usage (from Stage 4 Guard or Stage 10 Tool orchestrator):

    decision = await evaluate_permission(
        tool=my_tool,
        tool_input=input_dict,
        rules=session_rules,
        mode=session.permission_mode,
    )
    if decision.behavior is PermissionBehavior.DENY:
        ...
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from geny_executor.permission.types import (
    PermissionDecision,
    PermissionMode,
    PermissionRule,
    SOURCE_PRIORITY,
)


class _ToolLike(Protocol):
    """Minimal interface we need from a Tool — avoids import cycle."""

    @property
    def name(self) -> str: ...

    async def prepare_permission_matcher(self, input: Dict[str, Any]) -> Callable[[str], bool]: ...


def _sort_by_source_priority(rules: List[PermissionRule]) -> List[PermissionRule]:
    """Stable sort by source (highest priority first)."""
    priority_index = {src: i for i, src in enumerate(SOURCE_PRIORITY)}
    return sorted(
        rules,
        key=lambda r: priority_index.get(r.source, len(SOURCE_PRIORITY)),
    )


async def evaluate_permission(
    *,
    tool: _ToolLike,
    tool_input: Dict[str, Any],
    rules: List[PermissionRule],
    mode: PermissionMode = PermissionMode.DEFAULT,
    capabilities_destructive: bool = False,
    fallback: Optional[Callable[[Dict[str, Any]], Awaitable[PermissionDecision]]] = None,
) -> PermissionDecision:
    """Evaluate whether a tool invocation is permitted.

    Resolution order:
        1. ``BYPASS`` mode → always allow (even over deny rules).
        2. ``PLAN`` mode + destructive → escalate to ASK unless an
           explicit ALLOW rule matches at a higher source than any
           DENY/ASK rule.
        3. Walk rules in source-priority order; first match decides.
        4. Fall back to the tool's ``check_permissions`` callback if
           provided (passed through as ``fallback``).
        5. Default allow when no rule and no fallback override.

    Args:
        tool: Object exposing ``name`` and ``prepare_permission_matcher``.
        tool_input: Validated input payload.
        rules: Rule set collected from all sources. Already-loaded order
            doesn't matter — this function re-sorts by source priority.
        mode: Session permission mode.
        capabilities_destructive: ``True`` when the tool reported
            ``ToolCapabilities.destructive`` for this input. Used to
            implement the PLAN-mode auto-escalation above.
        fallback: Optional async callback the caller may pass to wire in
            the tool's own ``check_permissions``. Kept as a parameter
            (rather than importing Tool directly) to avoid a circular
            dependency.
    """
    # 1. BYPASS short-circuit
    if mode is PermissionMode.BYPASS:
        return PermissionDecision.allow(reason="bypass mode")

    # Resolve the matcher exactly once — some tools may do non-trivial
    # work (parsing the command, building regexes) so we avoid repeating.
    matcher = await tool.prepare_permission_matcher(tool_input)

    ordered = _sort_by_source_priority(rules)
    for rule in ordered:
        if not rule.matches_name(tool.name):
            continue
        if rule.pattern is not None and not matcher(rule.pattern):
            continue
        # First match wins
        reason = (
            rule.reason or f"matched {rule.source.value}: {rule.tool_name}({rule.pattern or '*'})"
        )
        return PermissionDecision(
            behavior=rule.behavior,
            reason=reason,
            matched_rule=rule,
        )

    # 2. PLAN mode auto-escalation (runs after rule walk so an explicit
    #    ALLOW at any source-priority still wins).
    if mode is PermissionMode.PLAN and capabilities_destructive:
        return PermissionDecision.ask(reason="plan mode — destructive operation needs approval")

    # AUTO mode allows even without an explicit rule, but we still want
    # tool-specific logic a chance (e.g. secret scanners).
    if fallback is not None:
        fb = await fallback(tool_input)
        return fb

    # Default — allow (tools opt in to deny via their own check_permissions)
    return PermissionDecision.allow(reason="no matching rule")
