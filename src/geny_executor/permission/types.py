"""Permission primitives — rule, source, mode, decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class PermissionBehavior(str, Enum):
    """What a rule instructs the engine to do when it matches."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionMode(str, Enum):
    """Session-level default policy. Rules still override when matched.

    Modes:
        ``DEFAULT`` — rules decide; tool's own ``check_permissions`` is
            the fallback. Destructive operations may surface ``ask``.
        ``PLAN`` — read-only stance. Any destructive invocation (per the
            tool's ``ToolCapabilities.destructive``) is auto-escalated to
            ``ask`` regardless of rule presence, unless explicitly allowed.
        ``AUTO`` — allow everything including destructive (useful for
            non-interactive CI). Rules can still ``deny``.
        ``BYPASS`` — allow everything unconditionally. Developer-only;
            intentionally bypasses even explicit ``deny`` rules.
    """

    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"
    BYPASS = "bypass"


class PermissionSource(str, Enum):
    """Where a rule was defined. Higher priority sources win on conflict.

    Order from **highest** to **lowest** priority:
        CLI_ARG      — ``geny run --allow "Bash(git *)"`` at launch
        LOCAL        — ``<project>/.geny/permissions.local.yaml`` (gitignored)
        PROJECT      — ``<project>/.geny/permissions.yaml`` (committed)
        USER         — ``~/.geny/permissions.yaml``
        PRESET_DEFAULT — shipped with a preset for sensible defaults
    """

    CLI_ARG = "cli_arg"
    LOCAL = "local"
    PROJECT = "project"
    USER = "user"
    PRESET_DEFAULT = "preset_default"


SOURCE_PRIORITY: Tuple[PermissionSource, ...] = (
    PermissionSource.CLI_ARG,
    PermissionSource.LOCAL,
    PermissionSource.PROJECT,
    PermissionSource.USER,
    PermissionSource.PRESET_DEFAULT,
)
"""Lookup order — first match wins. Lower index = higher priority."""


@dataclass(frozen=True)
class PermissionRule:
    """A single rule in the permission matrix.

    Attributes:
        tool_name: Exact tool name, or ``"*"`` for all tools.
        pattern: Optional sub-pattern delegated to
            ``Tool.prepare_permission_matcher``. When ``None`` the rule
            applies to every input. Example patterns: ``"git *"``,
            ``"read-only"``.
        behavior: Allow / deny / ask.
        source: Which source defined this rule (for priority ordering +
            audit logs).
        reason: Optional human-readable note, shown in audit log and
            ``ask`` prompts.
    """

    tool_name: str
    behavior: PermissionBehavior
    source: PermissionSource
    pattern: Optional[str] = None
    reason: Optional[str] = None

    def matches_name(self, tool_name: str) -> bool:
        """True if ``self.tool_name`` targets ``tool_name``."""
        return self.tool_name == "*" or self.tool_name == tool_name


@dataclass(frozen=True)
class PermissionDecision:
    """Final verdict for a permission check.

    Mirrors ``geny_executor.tools.base.PermissionDecision`` — kept here
    as a standalone type so the permission package has no hard import
    on the tools package (loop safety).
    """

    behavior: PermissionBehavior
    reason: Optional[str] = None
    matched_rule: Optional[PermissionRule] = None
    updated_input: Optional[Dict] = None

    @classmethod
    def allow(
        cls, reason: Optional[str] = None, matched_rule: Optional[PermissionRule] = None
    ) -> "PermissionDecision":
        return cls(behavior=PermissionBehavior.ALLOW, reason=reason, matched_rule=matched_rule)

    @classmethod
    def deny(
        cls, reason: str, matched_rule: Optional[PermissionRule] = None
    ) -> "PermissionDecision":
        return cls(behavior=PermissionBehavior.DENY, reason=reason, matched_rule=matched_rule)

    @classmethod
    def ask(
        cls, reason: str, matched_rule: Optional[PermissionRule] = None
    ) -> "PermissionDecision":
        return cls(behavior=PermissionBehavior.ASK, reason=reason, matched_rule=matched_rule)
