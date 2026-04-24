"""Permission rule matrix — scoped, pattern-matched, hierarchical.

Cycle 20260424 executor uplift — Phase 1 Week 2 Checkpoint 2.

Evaluates whether a tool invocation is allowed based on rules loaded
from multiple sources (CLI args, local project, project settings, user
settings, preset defaults). Rule matching delegates to the tool's
``prepare_permission_matcher()`` so tools with structured inputs (Bash,
FileEdit) can implement sub-patterns like ``"Bash(git *)"``.

Integration points:
- Stage 4 (Guard) — consults the matrix before dispatching tools
- Stage 10 (Tool) — final check immediately before execute()
- Stage 15 (HITL, Phase 9) — receives ``ask`` decisions as approval requests

See ``executor_uplift/09_design_extension_interface.md`` §2.
"""

from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionDecision,
    PermissionMode,
    PermissionRule,
    PermissionSource,
    SOURCE_PRIORITY,
)
from geny_executor.permission.matrix import evaluate_permission
from geny_executor.permission.loader import (
    load_permission_rules,
    parse_permission_rules,
)

__all__ = [
    "PermissionBehavior",
    "PermissionDecision",
    "PermissionMode",
    "PermissionRule",
    "PermissionSource",
    "SOURCE_PRIORITY",
    "evaluate_permission",
    "load_permission_rules",
    "parse_permission_rules",
]
