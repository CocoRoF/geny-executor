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
from typing import Any, List, Tuple

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

    # PR-B.4.1 — additional richer-schema fields. All optional.
    category = meta_dict.get("category")
    if category is not None and not isinstance(category, str):
        category = str(category)
    if isinstance(category, str) and not category.strip():
        category = None
    elif isinstance(category, str):
        category = category.strip()

    effort = meta_dict.get("effort")
    if effort is not None and not isinstance(effort, str):
        effort = str(effort)
    if isinstance(effort, str) and not effort.strip():
        effort = None
    elif isinstance(effort, str):
        effort = effort.strip()

    raw_examples = meta_dict.get("examples", [])
    if raw_examples in (None, ""):
        examples: Tuple[str, ...] = ()
    elif isinstance(raw_examples, list):
        if not all(isinstance(x, str) for x in raw_examples):
            raise SkillLoadError(f"{path}: every entry in 'examples' must be a string")
        examples = tuple(raw_examples)
    elif isinstance(raw_examples, str):
        examples = (raw_examples,)
    else:
        raise SkillLoadError(
            f"{path}: 'examples' must be a list/string (got {type(raw_examples).__name__})"
        )

    # Phase 10.1 — declared argument names. Body uses ${name} for
    # interpolation. Accepts a list, a single string (single-arg
    # convenience), or empty.
    raw_arguments = meta_dict.get("arguments", [])
    if raw_arguments in (None, ""):
        arguments: Tuple[str, ...] = ()
    elif isinstance(raw_arguments, list):
        if not all(isinstance(x, str) for x in raw_arguments):
            raise SkillLoadError(f"{path}: every entry in 'arguments' must be a string")
        arguments = tuple(s.strip() for s in raw_arguments if s.strip())
    elif isinstance(raw_arguments, str):
        arguments = (raw_arguments.strip(),) if raw_arguments.strip() else ()
    else:
        raise SkillLoadError(
            f"{path}: 'arguments' must be a list/string (got {type(raw_arguments).__name__})"
        )

    # Phase 10.1 — argument_hint / when_to_use are pure copy fields.
    argument_hint = meta_dict.get("argument_hint")
    if argument_hint is not None and not isinstance(argument_hint, str):
        argument_hint = str(argument_hint)
    if isinstance(argument_hint, str) and not argument_hint.strip():
        argument_hint = None
    elif isinstance(argument_hint, str):
        argument_hint = argument_hint.strip()

    when_to_use = meta_dict.get("when_to_use")
    if when_to_use is not None and not isinstance(when_to_use, str):
        when_to_use = str(when_to_use)
    if isinstance(when_to_use, str) and not when_to_use.strip():
        when_to_use = None
    elif isinstance(when_to_use, str):
        when_to_use = when_to_use.strip()

    # Phase 10.1 — invocation flags. Accept the full menagerie of
    # YAML-loadable booleanish values so authors aren't surprised.
    def _coerce_bool(value: Any, *, default: bool, field_name: str) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in ("true", "yes", "y", "on", "1"):
                return True
            if normalised in ("false", "no", "n", "off", "0", ""):
                return False
        raise SkillLoadError(f"{path}: '{field_name}' must be a boolean (got {value!r})")

    user_invocable = _coerce_bool(
        meta_dict.get("user_invocable"),
        default=True,
        field_name="user_invocable",
    )
    disable_model_invocation = _coerce_bool(
        meta_dict.get("disable_model_invocation"),
        default=False,
        field_name="disable_model_invocation",
    )

    # Phase 10.2 — `paths` conditional activation. Accepts a list, a
    # comma-separated string, or single string. Trailing whitespace
    # is stripped per pattern; empty entries are dropped.
    raw_paths = meta_dict.get("paths", [])
    if raw_paths in (None, ""):
        paths_tuple: Tuple[str, ...] = ()
    elif isinstance(raw_paths, list):
        if not all(isinstance(x, str) for x in raw_paths):
            raise SkillLoadError(f"{path}: every entry in 'paths' must be a string")
        paths_tuple = tuple(s.strip() for s in raw_paths if s.strip())
    elif isinstance(raw_paths, str):
        # Allow comma-separated single-line form for convenience.
        parts = [s.strip() for s in raw_paths.split(",")]
        paths_tuple = tuple(p for p in parts if p)
    else:
        raise SkillLoadError(
            f"{path}: 'paths' must be a list/string (got {type(raw_paths).__name__})"
        )

    # Extras: every key we didn't explicitly consume — gives hosts a
    # growth surface without forcing executor schema changes.
    consumed = {
        "name",
        "description",
        "version",
        "allowed_tools",
        "model_override",
        "execution_mode",
        "category",
        "effort",
        "examples",
        "arguments",
        "argument_hint",
        "when_to_use",
        "user_invocable",
        "disable_model_invocation",
        "paths",
    }
    extras = {k: v for k, v in meta_dict.items() if k not in consumed}

    meta = SkillMetadata(
        name=name.strip(),
        description=description.strip(),
        version=version,
        allowed_tools=allowed,
        model_override=model_override,
        execution_mode=execution_mode,
        category=category,
        effort=effort,
        examples=examples,
        arguments=arguments,
        argument_hint=argument_hint,
        when_to_use=when_to_use,
        user_invocable=user_invocable,
        disable_model_invocation=disable_model_invocation,
        paths=paths_tuple,
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
