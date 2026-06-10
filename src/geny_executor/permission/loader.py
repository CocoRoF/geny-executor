"""YAML loader for permission rule files.

Expected file shape (all sections optional):

    default_posture: deny   # 2.2.0 — 'allow' | 'deny'; omitted → allow
    allow:
      - { tool: Read,   pattern: "*" }
      - { tool: Bash,   pattern: "git *", reason: "needed for CI" }
    deny:
      - { tool: Bash,   pattern: "rm -rf *" }
    ask:
      - { tool: Edit,   pattern: "*" }

Rules from a given file carry a single ``PermissionSource``; the caller
tells the loader which source the file represents.

``default_posture`` (2.2.0, audit §1-5) lives in the same file as the
rules on purpose — the posture and the allowlist it guards are one
policy and must travel together. ``parse_permission_policy`` /
``load_permission_policy`` return both as a ``PermissionPolicy``; the
older rules-only functions stay untouched for back-compat.

YAML is optional: if PyYAML is not installed the loader falls back to
``json`` (same schema works as JSON), so basic deployments don't need
an extra dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionPolicy,
    PermissionPosture,
    PermissionRule,
    PermissionSource,
)


def _load_document(path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON document from ``path`` into a dict."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load .yaml permission files. Install it or switch to JSON."
            ) from e
        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return data


def parse_permission_rules(
    data: Dict[str, Any], *, source: PermissionSource
) -> List[PermissionRule]:
    """Convert a parsed dict into a list of ``PermissionRule`` instances."""
    out: List[PermissionRule] = []
    for section_key, behavior in (
        ("allow", PermissionBehavior.ALLOW),
        ("deny", PermissionBehavior.DENY),
        ("ask", PermissionBehavior.ASK),
    ):
        section = data.get(section_key) or []
        if not isinstance(section, list):
            raise ValueError(
                f"'{section_key}' section must be a list, got {type(section).__name__}"
            )
        for entry in section:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"rule entry must be a mapping, got {type(entry).__name__}: {entry!r}"
                )
            tool = entry.get("tool")
            if not isinstance(tool, str) or not tool:
                raise ValueError(f"rule entry missing/invalid 'tool': {entry!r}")
            pattern_raw = entry.get("pattern")
            pattern = pattern_raw if isinstance(pattern_raw, str) else None
            reason_raw = entry.get("reason")
            reason = reason_raw if isinstance(reason_raw, str) else None
            out.append(
                PermissionRule(
                    tool_name=tool,
                    pattern=pattern,
                    behavior=behavior,
                    source=source,
                    reason=reason,
                )
            )
    return out


def load_permission_rules(path: Path, *, source: PermissionSource) -> List[PermissionRule]:
    """Load and parse a permission file.

    Returns an empty list when the file doesn't exist (absence is not an
    error — fresh projects may not have any permission config).
    """
    if not path.exists():
        return []
    data = _load_document(path)
    return parse_permission_rules(data, source=source)


def _parse_posture(data: Dict[str, Any], *, source_label: str) -> Optional[PermissionPosture]:
    """Read the ``default_posture`` key. ``None`` when not declared.

    Unknown values raise ``ValueError`` instead of warning-and-falling-
    back: a typo'd ``deny`` silently becoming ``allow`` would be a
    security downgrade masked behind a green load — the exact decoy-
    field failure mode the 2026-06-09 audit catalogued. The field is
    new in 2.2.0 so a hard error breaks no existing file.
    """
    raw = data.get("default_posture")
    if raw is None:
        return None
    try:
        return PermissionPosture(str(raw).strip().lower())
    except ValueError:
        raise ValueError(
            f"{source_label}: invalid 'default_posture' {raw!r} — must be 'allow' or 'deny'"
        ) from None


def parse_permission_policy(data: Dict[str, Any], *, source: PermissionSource) -> PermissionPolicy:
    """Convert a parsed dict into a :class:`PermissionPolicy` (rules + posture).

    The 2.2.0 superset of :func:`parse_permission_rules` — same rule
    sections, plus the optional top-level ``default_posture`` key. An
    absent key yields the ``ALLOW`` back-compat posture with
    ``posture_declared=False`` so hierarchical merging can tell
    silence apart from an explicit choice.
    """
    rules = parse_permission_rules(data, source=source)
    posture = _parse_posture(data, source_label=source.value)
    if posture is None:
        return PermissionPolicy(rules=rules)
    return PermissionPolicy(rules=rules, default_posture=posture, posture_declared=True)


def load_permission_policy(path: Path, *, source: PermissionSource) -> PermissionPolicy:
    """Load and parse a permission file into a :class:`PermissionPolicy`.

    Returns an empty allow-posture policy when the file doesn't exist
    (absence is not an error — mirrors :func:`load_permission_rules`).
    """
    if not path.exists():
        return PermissionPolicy()
    data = _load_document(path)
    return parse_permission_policy(data, source=source)


def load_hierarchical_rules(
    *,
    cli_rules: Optional[List[PermissionRule]] = None,
    local_path: Optional[Path] = None,
    project_path: Optional[Path] = None,
    user_path: Optional[Path] = None,
    preset_rules: Optional[List[PermissionRule]] = None,
) -> List[PermissionRule]:
    """Collect rules from every source into one flat list.

    Priority ordering is performed later by
    ``geny_executor.permission.matrix.evaluate_permission`` — this
    function just concatenates.
    """
    rules: List[PermissionRule] = []
    if cli_rules:
        rules.extend(cli_rules)
    if local_path is not None:
        rules.extend(load_permission_rules(local_path, source=PermissionSource.LOCAL))
    if project_path is not None:
        rules.extend(load_permission_rules(project_path, source=PermissionSource.PROJECT))
    if user_path is not None:
        rules.extend(load_permission_rules(user_path, source=PermissionSource.USER))
    if preset_rules:
        rules.extend(preset_rules)
    return rules


def load_hierarchical_policy(
    *,
    cli_rules: Optional[List[PermissionRule]] = None,
    local_path: Optional[Path] = None,
    project_path: Optional[Path] = None,
    user_path: Optional[Path] = None,
    preset_rules: Optional[List[PermissionRule]] = None,
) -> PermissionPolicy:
    """Collect rules from every source AND resolve the effective posture.

    The 2.2.0 policy-aware sibling of :func:`load_hierarchical_rules`
    (which is preserved untouched). Rules are concatenated exactly as
    before; the posture is taken from the **highest-priority file that
    explicitly declares one** (LOCAL > PROJECT > USER, matching
    ``SOURCE_PRIORITY``). Files that stay silent don't vote — a local
    override file that only adds an allow rule must not accidentally
    reset a project-level ``default_posture: deny`` back to allow.

    No file declaring a posture → ``ALLOW`` with
    ``posture_declared=False`` (the 2.x back-compat baseline; 3.0
    flips this default — see :class:`PermissionPosture`).
    """
    rules = load_hierarchical_rules(
        cli_rules=cli_rules,
        local_path=local_path,
        project_path=project_path,
        user_path=user_path,
        preset_rules=preset_rules,
    )
    # Highest-priority declared posture wins. CLI rules / preset rules
    # are bare rule lists (no file to carry a posture), so only the
    # three file tiers participate.
    for path, source in (
        (local_path, PermissionSource.LOCAL),
        (project_path, PermissionSource.PROJECT),
        (user_path, PermissionSource.USER),
    ):
        if path is None or not path.exists():
            continue
        policy = parse_permission_policy(_load_document(path), source=source)
        if policy.posture_declared:
            return PermissionPolicy(
                rules=rules,
                default_posture=policy.default_posture,
                posture_declared=True,
            )
    return PermissionPolicy(rules=rules)
