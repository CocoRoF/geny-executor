"""Skill loader — turn SKILL.md files on disk into :class:`Skill` instances.

Cycle 20260424 executor uplift — Phase 4 Week 7.

Public entry points:

* :func:`parse_skill_file` — read and parse a single SKILL.md.
* :func:`load_skills_dir` — walk a directory tree, finding every
  ``SKILL.md`` inside ``<dir>/<skill-name>/`` layout.

Directory layout convention mirrors Claude Code::

    .skills/
        refactor-ts/
            SKILL.md
        deploy-helper/
            SKILL.md
            assets/...

The skill's id is the parent directory name (``refactor-ts``,
``deploy-helper``). This lets skill authors colocate assets with the
skill file itself; the loader only consumes SKILL.md.

Malformed skills:

* Missing or unreadable file → raise :class:`SkillLoadError`.
* Missing required metadata (``name``, ``description``) → raise.
* Invalid ``execution_mode`` value → raise.
* Non-dict ``allowed_tools`` entries → raise.

``load_skills_dir`` catches per-skill errors and logs them at
WARNING by default so one malformed skill doesn't kill the whole
load pass — but it returns the list of *successfully* loaded skills
and a parallel list of errors so the caller can surface the rest
to the user / UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from geny_executor.skills.frontmatter import parse_frontmatter
from geny_executor.skills.types import Skill, SkillMetadata, validate_execution_mode

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"


class SkillLoadError(Exception):
    """Raised when a single SKILL.md fails to parse into a valid Skill."""


@dataclass
class SkillLoadReport:
    """Outcome of :func:`load_skills_dir`.

    ``loaded`` and ``errors`` are parallel lists in discovery order.
    Callers wanting "all or nothing" semantics check ``errors`` and
    raise; callers happy with best-effort (most hosts) use ``loaded``
    directly and surface ``errors`` separately.
    """

    loaded: List[Skill]
    errors: List[Tuple[Path, SkillLoadError]]


def parse_skill_file(path: Path) -> Skill:
    """Load a single SKILL.md file and return the :class:`Skill`.

    Raises:
        SkillLoadError: for any failure — file missing, bad
            frontmatter, missing required metadata, invalid
            execution_mode.
    """
    if not path.exists():
        raise SkillLoadError(f"SKILL.md not found: {path}")
    if not path.is_file():
        raise SkillLoadError(f"not a file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillLoadError(f"could not read {path}: {exc}") from exc

    meta_dict, body = parse_frontmatter(raw)
    if not meta_dict:
        raise SkillLoadError(f"{path}: missing or invalid YAML frontmatter (expected '---' block)")

    name = meta_dict.get("name")
    description = meta_dict.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillLoadError(f"{path}: 'name' is required and must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise SkillLoadError(f"{path}: 'description' is required and must be a non-empty string")

    raw_allowed = meta_dict.get("allowed_tools", [])
    if raw_allowed in (None, ""):
        allowed: Tuple[str, ...] = ()
    elif isinstance(raw_allowed, list):
        if not all(isinstance(x, str) for x in raw_allowed):
            raise SkillLoadError(f"{path}: every entry in 'allowed_tools' must be a string")
        allowed = tuple(raw_allowed)
    else:
        raise SkillLoadError(
            f"{path}: 'allowed_tools' must be a list of strings (got {type(raw_allowed).__name__})"
        )

    execution_mode = str(meta_dict.get("execution_mode", "inline"))
    try:
        execution_mode = validate_execution_mode(execution_mode)
    except ValueError as exc:
        raise SkillLoadError(f"{path}: {exc}") from exc

    version = meta_dict.get("version")
    if version is not None and not isinstance(version, str):
        version = str(version)

    model_override = meta_dict.get("model_override")
    if model_override is not None and not isinstance(model_override, str):
        raise SkillLoadError(f"{path}: 'model_override' must be a string or absent")
    if isinstance(model_override, str) and not model_override.strip():
        model_override = None

    # Extras: every key we didn't explicitly consume — gives hosts a
    # growth surface without forcing executor schema changes.
    consumed = {
        "name",
        "description",
        "version",
        "allowed_tools",
        "model_override",
        "execution_mode",
    }
    extras = {k: v for k, v in meta_dict.items() if k not in consumed}

    meta = SkillMetadata(
        name=name.strip(),
        description=description.strip(),
        version=version,
        allowed_tools=allowed,
        model_override=model_override,
        execution_mode=execution_mode,
        extras=extras,
    )

    # Derive id from the parent directory name so collocated assets
    # work naturally. Files dropped directly (rare) fall back to the
    # filename stem.
    skill_id = path.parent.name if path.parent.name and path.parent != path else path.stem

    return Skill(id=skill_id, metadata=meta, body=body, source=path)


def load_skills_dir(root: Path, *, strict: bool = False) -> SkillLoadReport:
    """Walk ``root`` and load every ``SKILL.md`` found.

    Search layout: ``<root>/<skill-id>/SKILL.md``. Other files in
    each skill directory (assets, .gitkeep, etc.) are ignored.

    Args:
        root: Directory to scan. If it doesn't exist, an empty report
            is returned — missing skill directories are not themselves
            errors.
        strict: If True, re-raise the first :class:`SkillLoadError`
            instead of collecting it. Off by default so bulk loads
            degrade gracefully.

    Returns:
        :class:`SkillLoadReport` with the loaded skills and any
        per-skill errors.
    """
    loaded: List[Skill] = []
    errors: List[Tuple[Path, SkillLoadError]] = []

    if not root.exists() or not root.is_dir():
        return SkillLoadReport(loaded=loaded, errors=errors)

    # Sort for deterministic discovery order — tests and UIs benefit.
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / SKILL_FILENAME
        if not skill_file.exists():
            continue
        try:
            skill = parse_skill_file(skill_file)
        except SkillLoadError as exc:
            if strict:
                raise
            logger.warning("skill load failed at %s: %s", skill_file, exc)
            errors.append((skill_file, exc))
            continue
        loaded.append(skill)

    return SkillLoadReport(loaded=loaded, errors=errors)
