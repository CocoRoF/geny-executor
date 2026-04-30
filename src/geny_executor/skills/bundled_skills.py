"""Bundled skill loader — Phase 10.4.

The executor ships a small catalog of operational skills under
``geny_executor/skills/bundled/<id>/SKILL.md``. They're packaged with
the wheel (see ``pyproject.toml`` `tool.hatch.build.targets.wheel`
include rules) so installs always carry them; hosts that want a
lighter footprint can opt out by passing ``include_bundled=False``
to :func:`load_bundled_skills`.

Convention mirrors disk-loaded skills exactly — the bundled tree is
just another root passed to :func:`load_skills_dir`. This means hosts
can replace / extend the catalog by dropping their own SKILL.md files
into a higher-priority directory; the registry's first-wins policy
keeps the host's version live.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from geny_executor.skills.loader import SkillLoadReport, load_skills_dir
from geny_executor.skills.types import Skill

logger = logging.getLogger(__name__)


def bundled_skills_dir() -> Path:
    """Return the path the executor's bundled skills live at.

    Resolves relative to *this* module so pip-installed wheels and
    source checkouts both find the directory.
    """
    return Path(__file__).parent / "bundled"


def load_bundled_skills(*, strict: bool = False) -> SkillLoadReport:
    """Load every bundled SKILL.md.

    Args:
        strict: Re-raise the first :class:`SkillLoadError` instead of
            collecting it. Off by default — a malformed bundled skill
            should never block the host's session start, but the
            failure should be visible in logs.

    Returns:
        A :class:`SkillLoadReport`. Hosts hand ``report.loaded`` to
        :meth:`SkillRegistry.register_many`.
    """
    root = bundled_skills_dir()
    if not root.exists():
        logger.debug("load_bundled_skills: %s does not exist; returning empty", root)
        return SkillLoadReport(loaded=[], errors=[])
    return load_skills_dir(root, strict=strict)


def bundled_skill_ids() -> List[str]:
    """Cheap directory scan that returns just the bundled skill ids
    without parsing every SKILL.md. Useful for hosts that want to
    show the catalog in a UI before deciding whether to load."""
    root = bundled_skills_dir()
    if not root.exists():
        return []
    return sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and (entry / "SKILL.md").exists()
    )


__all__ = [
    "bundled_skill_ids",
    "bundled_skills_dir",
    "load_bundled_skills",
    "Skill",
]
