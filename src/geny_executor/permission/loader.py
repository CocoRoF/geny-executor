"""YAML loader for permission rule files.

Expected file shape (all sections optional):

    allow:
      - { tool: Read,   pattern: "*" }
      - { tool: Bash,   pattern: "git *", reason: "needed for CI" }
    deny:
      - { tool: Bash,   pattern: "rm -rf *" }
    ask:
      - { tool: Edit,   pattern: "*" }

Rules from a given file carry a single ``PermissionSource``; the caller
tells the loader which source the file represents.

YAML is optional: if PyYAML is not installed the loader falls back to
``json`` (same schema works as JSON), so basic deployments don't need
an extra dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.permission.types import (
    PermissionBehavior,
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
