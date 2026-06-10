"""Permission matrix evaluation — the single entry point.

Usage (from Stage 4 Guard or Stage 10 Tool orchestrator):

    decision = await evaluate_permission(
        tool=my_tool,
        tool_input=input_dict,
        rules=session_rules,
        mode=session.permission_mode,
        default_posture=PermissionPosture.DENY,
    )
    if decision.behavior is PermissionBehavior.DENY:
        ...

Posture (2.2.0, audit §1-5 "policy via config, not hardcode"):
the matrix is **allow-by-default** when no rule matches — a 2.x
back-compat artifact, not the design intent. The configurable
``default_posture`` parameter is the migration path: hosts set it to
``DENY`` to get "deny everything except an explicit allowlist" today.
**The philosophy target is deny-by-default and 3.0 flips the
default** — new integrations should pass ``DENY`` explicitly now so
the flip never surprises them.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from geny_executor.permission.types import (
    PermissionDecision,
    PermissionMode,
    PermissionPosture,
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
    default_posture: PermissionPosture = PermissionPosture.ALLOW,
) -> PermissionDecision:
    """Evaluate whether a tool invocation is permitted.

    Resolution order:
        1. ``BYPASS`` mode → always allow (even over deny rules and a
           DENY posture — it is the developer escape hatch and beats
           everything by design).
        2. Walk rules in source-priority order; first match decides.
        3. ``PLAN`` mode + destructive → escalate to ASK (runs after
           the rule walk so an explicit ALLOW still wins; ASK is more
           conservative than allow and gives HITL a chance, so it also
           takes precedence over the posture).
        4. ``default_posture`` is ``DENY`` → deny. The tool's own
           ``check_permissions`` fallback is deliberately NOT consulted
           here: under deny-by-default only *operator-declared* allow
           rules may open the gate — a tool vouching for itself is not
           an allowlist entry.
        5. Fall back to the tool's ``check_permissions`` callback if
           provided (passed through as ``fallback``).
        6. Default allow (the ``ALLOW`` posture, 2.x back-compat).

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
        default_posture: What happens when no rule matches (2.2.0).
            ``ALLOW`` (default, back-compat) preserves the historical
            behaviour. ``DENY`` refuses unmatched invocations — combined
            with an empty rule set this is "deny everything except an
            explicit allowlist". The 3.0 default flips to ``DENY``; see
            :class:`~geny_executor.permission.types.PermissionPosture`.
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
        # First match wins.
        reason = (
            rule.reason or f"matched {rule.source.value}: {rule.tool_name}({rule.pattern or '*'})"
        )
        # PR-B.5.1: ACCEPT_EDITS / DONT_ASK promote ASK rules to ALLOW.
        # DENY rules pass through untouched on both modes.
        promoted = _maybe_promote_ask(rule, mode, tool.name)
        if promoted is not None:
            return PermissionDecision(
                behavior=promoted,
                reason=f"{reason} (mode={mode.value} promoted to allow)",
                matched_rule=rule,
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

    # 3. Posture gate (2.2.0). DENY refuses everything no rule
    #    explicitly allowed — including under AUTO mode (AUTO never
    #    overrode deny rules, and a deny posture is the operator's
    #    blanket deny rule). The fallback is skipped on purpose: see
    #    the docstring's resolution-order rationale.
    if default_posture is PermissionPosture.DENY:
        return PermissionDecision.deny(
            reason="default posture is deny — no explicit allow rule matched"
        )

    # AUTO mode allows even without an explicit rule, but we still want
    # tool-specific logic a chance (e.g. secret scanners).
    if fallback is not None:
        fb = await fallback(tool_input)
        return fb

    # Default — allow (tools opt in to deny via their own check_permissions).
    # This branch IS the ALLOW posture; 3.0 flips the default to DENY.
    return PermissionDecision.allow(reason="no matching rule")


def _maybe_promote_ask(rule, mode, tool_name):
    """Return PermissionBehavior.ALLOW when the mode promotes an ASK
    rule to an ALLOW for this tool. Otherwise return None and the
    caller leaves the rule's behavior unchanged.

    Promotion rules:
      - DONT_ASK     → any ASK becomes ALLOW.
      - ACCEPT_EDITS → ASK on edit-class tools becomes ALLOW.
    """
    from geny_executor.permission.types import (
        EDIT_TOOLS,
        PermissionBehavior,
        PermissionMode,
    )

    if rule.behavior is not PermissionBehavior.ASK:
        return None
    if mode is PermissionMode.DONT_ASK:
        return PermissionBehavior.ALLOW
    if mode is PermissionMode.ACCEPT_EDITS and tool_name in EDIT_TOOLS:
        return PermissionBehavior.ALLOW
    return None
