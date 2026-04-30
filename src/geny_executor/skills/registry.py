"""In-memory Skill registry.

Cycle 20260424 executor uplift — Phase 4 Week 7.

Mirrors the shape of :class:`~geny_executor.tools.registry.ToolRegistry`
for consistency: a flat name → object map with register / get / list /
filter helpers. The registry is responsibly *just a map* — lifecycle
concerns (loading from disk, hot-reload, watching a directory) belong
in :mod:`~geny_executor.skills.loader` and future watcher modules.

``register`` policy: first wins. A second registration with the same id
is rejected with ``ValueError`` so hosts can detect bundled-vs-project
collisions at load time instead of silently shadowing one with the
other. Callers that *want* override semantics should explicitly call
``unregister(id)`` first.

Hosts typically build one registry per AgentSession, seed it from
bundled skills + project ``.skills/`` + user ``~/.skills/``, and hand
it to the :class:`SkillTool` wrapper.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from geny_executor.skills.types import Skill


class SkillRegistry:
    """Name → :class:`Skill` map."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> "SkillRegistry":
        """Add *skill* to the registry. Returns self for chaining.

        Raises:
            ValueError: when another skill already holds this id.
        """
        if skill.id in self._skills:
            existing = self._skills[skill.id]
            raise ValueError(
                f"skill {skill.id!r} already registered (from {existing.source!r}); "
                f"call unregister({skill.id!r}) first to override"
            )
        self._skills[skill.id] = skill
        return self

    def register_many(self, skills: Iterable[Skill]) -> "SkillRegistry":
        """Register each skill in ``skills`` — stops on first collision."""
        for skill in skills:
            self.register(skill)
        return self

    def unregister(self, skill_id: str) -> None:
        """Remove the skill with this id. No-op if not present."""
        self._skills.pop(skill_id, None)

    def clear(self) -> None:
        """Remove every registered skill. Phase 10.7 — used by
        :class:`SkillRegistryWatcher` to atomically reload the
        catalog after an on-disk change."""
        self._skills.clear()

    def get(self, skill_id: str) -> Optional[Skill]:
        """Return the skill or ``None`` if not registered."""
        return self._skills.get(skill_id)

    def list_all(self) -> List[Skill]:
        """Every registered skill, in id order."""
        return [self._skills[k] for k in sorted(self._skills.keys())]

    def list_ids(self) -> List[str]:
        """Sorted list of every registered skill id."""
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills
