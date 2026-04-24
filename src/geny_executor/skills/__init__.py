"""Skill system — code-free capability units.

A Skill is a ``SKILL.md`` file (YAML frontmatter + markdown body) that
lets users extend an agent's capability without writing Python. The
executor loads skills at session start, exposes them through a Tool
wrapper, and routes LLM invocations back through the skill's prompt
body.

Phase 4 Week 7 ships the foundation:
- :class:`Skill` + :class:`SkillMetadata` + :class:`SkillContext` types
- :func:`parse_skill_file` / :func:`load_skills_dir` loaders
- :class:`SkillRegistry` in-memory store
- :func:`parse_frontmatter` YAML parser (stdlib + pyyaml)

Later phases add:
- ``SkillTool`` — exposes a skill as a ``Tool`` (Phase 4 Week 8)
- Bundled skills + CLI helpers (Phase 4 Week 9)
"""

from geny_executor.skills.frontmatter import parse_frontmatter
from geny_executor.skills.loader import (
    SKILL_FILENAME,
    SkillLoadError,
    SkillLoadReport,
    load_skills_dir,
    parse_skill_file,
)
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.skill_tool import (
    SkillTool,
    SkillToolProvider,
    build_skill_tool,
)
from geny_executor.skills.types import (
    Skill,
    SkillContext,
    SkillMetadata,
    validate_execution_mode,
)

__all__ = [
    "Skill",
    "SkillContext",
    "SkillMetadata",
    "SkillRegistry",
    "SkillTool",
    "SkillToolProvider",
    "SkillLoadError",
    "SkillLoadReport",
    "SKILL_FILENAME",
    "load_skills_dir",
    "parse_skill_file",
    "parse_frontmatter",
    "validate_execution_mode",
    "build_skill_tool",
]
