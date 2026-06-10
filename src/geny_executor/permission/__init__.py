"""Permission rule matrix — scoped, pattern-matched, hierarchical.

Cycle 20260424 executor uplift — Phase 1 Week 2 Checkpoint 2.

Evaluates whether a tool invocation is allowed based on rules loaded
from multiple sources (CLI args, local project, project settings, user
settings, preset defaults). Rule matching delegates to the tool's
``prepare_permission_matcher()`` so tools with structured inputs (Bash,
FileEdit) can implement sub-patterns like ``"Bash(git *)"``.

.. warning:: **Posture: allow-by-default today, deny-by-default at 3.0.**

   When no rule matches, the matrix currently allows — a 2.x
   back-compat artifact directly at odds with this library's declared
   "policy via config, not hardcode / deny-by-default" philosophy
   (called out as a violation by the 2026-06-09 environment-philosophy
   audit, §1-5). The 2.2.0 migration path is the configurable
   ``default_posture`` ('allow' | 'deny'):

   * pass ``default_posture=PermissionPosture.DENY`` to
     :func:`evaluate_permission`, or
   * write ``default_posture: deny`` at the top of the same YAML file
     the rules live in (``load_permission_policy`` /
     ``load_hierarchical_policy`` pick it up), or
   * set ``permission_default_posture`` on the Tool stage's
     ``ToolContext`` so Stage 10's dispatch honours it — including
     when **zero rules are bound** (deny posture + no rules = deny
     everything except an explicit allowlist).

   **3.0 flips the default to deny.** Hosts should declare their
   posture explicitly now so the flip is a non-event for them.

Integration points:
- Stage 4 (Guard) — consults the matrix before dispatching tools
- Stage 10 (Tool) — final check immediately before execute(); since
  2.2.0 ASK decisions route to a bound HITL requester (Stage 15
  contract) and DENY/ASK fire the ``permission_denied`` /
  ``permission_request`` hook events
- Stage 15 (HITL) — supplies the ``Requester`` contract that resolves
  ``ask`` decisions into human verdicts

See ``executor_uplift/09_design_extension_interface.md`` §2.
"""

from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionPosture,
    PermissionRule,
    PermissionSource,
    SOURCE_PRIORITY,
    coerce_posture,
)
from geny_executor.permission.matrix import evaluate_permission
from geny_executor.permission.loader import (
    load_hierarchical_policy,
    load_hierarchical_rules,
    load_permission_policy,
    load_permission_rules,
    parse_permission_policy,
    parse_permission_rules,
)

__all__ = [
    "PermissionBehavior",
    "PermissionDecision",
    "PermissionMode",
    "PermissionPolicy",
    "PermissionPosture",
    "PermissionRule",
    "PermissionSource",
    "SOURCE_PRIORITY",
    "coerce_posture",
    "evaluate_permission",
    "load_hierarchical_policy",
    "load_hierarchical_rules",
    "load_permission_policy",
    "load_permission_rules",
    "parse_permission_policy",
    "parse_permission_rules",
]
