"""Permission primitives — rule, source, mode, posture, decision."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PermissionBehavior(str, Enum):
    """What a rule instructs the engine to do when it matches."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionPosture(str, Enum):
    """What the engine does when **no rule matches** (2.2.0, audit §1-5).

    This is the configurable baseline underneath the whole matrix:

    - ``ALLOW`` — unmatched invocations proceed (then the tool's own
      ``check_permissions`` fallback gets a say). This is the 2.x
      default purely for back-compat: both production hosts were built
      against allow-by-default and a silent flip would break them.
    - ``DENY`` — unmatched invocations are refused. With no rules
      bound at all, DENY means "deny everything except an explicit
      allowlist" — the matrix is consulted even for an empty rule set.

    .. warning:: The philosophy target is **deny-by-default**
       ("policy via config, not hardcode" — the declared posture of
       this library). 3.0 flips the default to ``DENY``; hosts should
       set the posture explicitly today so the flip is a no-op for
       them. Until then the default stays ``ALLOW`` because 2.2.0 is
       a minor release and may not change runtime behaviour under
       existing configs.
    """

    ALLOW = "allow"
    DENY = "deny"


def coerce_posture(
    raw: Any, *, default: Optional["PermissionPosture"] = None
) -> "PermissionPosture":
    """Best-effort coercion of a posture value (str / enum / None).

    ``None`` → the 2.x back-compat default (``ALLOW``) — hosts that
    never heard of postures keep their behaviour. Unknown strings log
    a WARNING and fall back to ``ALLOW`` rather than raising: this
    path receives *programmatic* host values (context attributes),
    where a hard error would take down dispatch for a typo. The YAML
    loader, by contrast, raises on unknown postures — a config file
    typo silently downgrading ``deny`` to ``allow`` is exactly the
    masked-degradation class the 2026-06-09 audit condemns, and at
    parse time we can still refuse loudly.
    """
    fallback = default if default is not None else PermissionPosture.ALLOW
    if raw is None:
        return fallback
    if isinstance(raw, PermissionPosture):
        return raw
    try:
        return PermissionPosture(str(raw).strip().lower())
    except ValueError:
        logger.warning(
            "unknown permission posture %r — falling back to %s (valid: 'allow' | 'deny')",
            raw,
            fallback.value,
        )
        return fallback


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
        ``ACCEPT_EDITS`` — like DEFAULT, but ASKs on edit-class tools
            (Write / Edit / NotebookEdit / MultiEdit) are auto-promoted
            to ALLOW. Other ASK rules untouched. (PR-B.5.1)
        ``DONT_ASK`` — every ASK becomes ALLOW. DENY rules still apply.
            For chat-style sessions where users want fewer prompts.
            (PR-B.5.1)
    """

    DEFAULT = "default"
    PLAN = "plan"
    AUTO = "auto"
    BYPASS = "bypass"
    ACCEPT_EDITS = "acceptEdits"
    DONT_ASK = "dontAsk"


# Tools whose destructiveness is "edit a file" — gated by ACCEPT_EDITS.
EDIT_TOOLS: Tuple[str, ...] = (
    "Write",
    "Edit",
    "NotebookEdit",
    "MultiEdit",
)


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


@dataclass(frozen=True)
class PermissionPolicy:
    """A rule set plus the posture that applies when no rule matches.

    Introduced in 2.2.0 so the posture travels on the same surface the
    rules do (one YAML file, one loader call) instead of being a
    second, easy-to-forget channel. ``load_permission_policy`` /
    ``parse_permission_policy`` in :mod:`~geny_executor.permission.loader`
    produce these.

    Attributes:
        rules: Flat rule list, same shape ``evaluate_permission`` takes.
        default_posture: Baseline when no rule matches. Defaults to
            ``ALLOW`` for 2.x back-compat (see
            :class:`PermissionPosture` for the 3.0 flip plan).
        posture_declared: True when the source file *explicitly* wrote
            a ``default_posture`` key. Lets hierarchical loading
            distinguish "this file says allow" from "this file is
            silent" — silence must not override a lower-priority file's
            explicit ``deny``.
    """

    rules: List[PermissionRule] = field(default_factory=list)
    default_posture: PermissionPosture = PermissionPosture.ALLOW
    posture_declared: bool = False
