"""SkillTool — expose a :class:`Skill` as a callable :class:`Tool`.

Cycle 20260424 executor uplift — Phase 4 Week 8.

The Phase 4 foundation (types, frontmatter, loader, registry) gave us
the shape of a skill. This module connects it to the LLM surface:
every registered skill becomes a tool the model can call by id.

Execution modes:

* **inline** (shipped here) — the tool returns the rendered skill body
  as text. The LLM treats the body as instructions, executes the
  steps itself using whatever other tools it already has access to,
  and returns to the host with the final result. Simple, cheap, and
  good enough for most skills.
* **fork** (Phase 7) — the tool spawns a sub-pipeline with the skill's
  restricted tool roster + model override, lets it run to completion,
  and returns a summary. Requires the AgentTool runtime, which lands
  alongside isolation worktrees.

This module ships the inline path. Fork mode is stubbed with a clean
error so skills marked ``execution_mode: fork`` fail fast instead of
silently running inline.

Argument interpolation is minimal by design: the skill body is
rendered with Python ``str.format_map`` over ``invoke_args``, using a
``_SafeFormatDict`` that leaves unknown placeholders untouched. Skill
authors who want structured args declare them as frontmatter ``extras``
and describe them in the body; the LLM reads the description and
passes appropriate values.
"""

from __future__ import annotations

from typing import Any, Dict, List

from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.types import Skill, SkillContext
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.provider import ToolProvider


class _SafeFormatDict(dict):
    """dict subclass that returns ``{key}`` for missing keys.

    Keeps un-interpolated braces intact so a skill body containing
    example template syntax doesn't explode when ``invoke_args`` is
    empty.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_body(skill: Skill, invoke_args: Dict[str, Any]) -> str:
    """Interpolate ``invoke_args`` into the skill body.

    Uses ``str.format_map`` with a safe dict so unknown ``{placeholders}``
    pass through unchanged. Non-string values are coerced with ``str``.
    """
    if not invoke_args:
        return skill.body
    safe = _SafeFormatDict({k: v for k, v in invoke_args.items()})
    try:
        return skill.body.format_map(safe)
    except (ValueError, KeyError, IndexError):
        # Malformed format spec in the body — return the body as-is
        # rather than crashing. The LLM can still follow the
        # guidance even if one placeholder is miswritten.
        return skill.body


class SkillTool(Tool):
    """Exposes a single skill as a Tool the LLM can call.

    The tool's ``name`` is the skill's id (not its human-readable
    ``metadata.name``, which is often non-identifier text). Its
    description comes from the skill metadata so the model sees a
    one-liner when deciding whether to invoke.

    The input schema defines a single ``args`` object — any structured
    parameters the skill author declared in frontmatter ``extras``
    should be documented in the skill body itself. Keeping the schema
    uniform across all skills avoids the LLM having to relearn
    per-skill shapes.
    """

    def __init__(self, skill: Skill):
        self._skill = skill
        self._name = skill.id  # id is the registry key + tool name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        suffix = f" [skill, {self._skill.metadata.execution_mode}]"
        return self._skill.description + suffix

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "object",
                    "description": (
                        "Optional arguments for the skill. Schema "
                        "defined by the skill body itself — consult "
                        "the skill description."
                    ),
                    "additionalProperties": True,
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Inline skills are pure prompt templates — no side effects,
        # safe to run in parallel. Fork-mode skills *would* need
        # to opt out; we keep them safe here because the fork runtime
        # (Phase 7) will override.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,  # the LLM consuming the body may differ turn-to-turn
        )

    @property
    def skill(self) -> Skill:
        """The underlying :class:`Skill` — useful for audit / debug."""
        return self._skill

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mode = self._skill.metadata.execution_mode
        if mode == "fork":
            return ToolResult(
                content=(
                    f"skill {self._skill.id!r} declares execution_mode='fork' "
                    f"but the fork runtime (AgentTool isolation, Phase 7) is "
                    f"not yet available in this release. Mark the skill as "
                    f"execution_mode='inline' to run now."
                ),
                is_error=True,
            )

        raw_args = input.get("args") or {}
        if not isinstance(raw_args, dict):
            return ToolResult(
                content=f"'args' must be an object, got {type(raw_args).__name__}",
                is_error=True,
            )

        body = _render_body(self._skill, raw_args)

        header_lines = [
            f"Skill invoked: {self._skill.name} ({self._skill.id})",
        ]
        if self._skill.metadata.version:
            header_lines.append(f"version: {self._skill.metadata.version}")
        if self._skill.metadata.allowed_tools:
            tools_joined = ", ".join(self._skill.metadata.allowed_tools)
            header_lines.append(f"allowed tools: {tools_joined}")
        if self._skill.metadata.model_override:
            header_lines.append(f"model override: {self._skill.metadata.model_override} (advisory)")

        header = "\n".join(header_lines)
        content = f"{header}\n\n{body}".rstrip() + "\n"

        # Populate a SkillContext for observability — hosts that want
        # to log skill invocations can read it off metadata.
        skill_ctx = SkillContext(
            skill=self._skill,
            parent_tool_use_id=context.parent_tool_use_id,
            invoke_args=raw_args,
            session_id=context.session_id,
            working_dir=context.working_dir,
        )

        return ToolResult(
            content=content,
            metadata={
                "skill_id": self._skill.id,
                "skill_name": self._skill.name,
                "execution_mode": mode,
                "allowed_tools": list(self._skill.metadata.allowed_tools),
                "model_override": self._skill.metadata.model_override,
                "args": raw_args,
                "body_chars": len(body),
                "rendered_template": body != self._skill.body,
                "skill_context": {
                    "session_id": skill_ctx.session_id,
                    "parent_tool_use_id": skill_ctx.parent_tool_use_id,
                },
            },
        )


class SkillToolProvider(ToolProvider):
    """Exposes every skill in a :class:`SkillRegistry` as a Tool.

    Plug this into :meth:`Pipeline.from_manifest_async` alongside
    :class:`BuiltInToolProvider` to give the LLM access to all
    registered skills without enumerating names in the manifest.

    Example::

        registry = SkillRegistry()
        report = load_skills_dir(Path(".skills"))
        registry.register_many(report.loaded)

        pipeline = await Pipeline.from_manifest_async(
            manifest,
            tool_providers=[
                BuiltInToolProvider(features=["filesystem"]),
                SkillToolProvider(registry),
            ],
        )

    The provider takes a live registry handle — if the caller mutates
    the registry between provider construction and pipeline startup,
    :meth:`list_tools` reflects the latest state. After the pipeline
    has started, mutations to the registry do *not* propagate (Stage
    10's tool registry snapshot owns the set).
    """

    def __init__(self, registry: SkillRegistry, *, name: str = "skills"):
        self._registry = registry
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        count = len(self._registry)
        return f"Skill tools ({count} skill{'s' if count != 1 else ''})"

    def list_tools(self) -> List[Tool]:
        return [SkillTool(skill) for skill in self._registry.list_all()]


def build_skill_tool(skill: Skill) -> SkillTool:
    """Factory equivalent to ``SkillTool(skill)``.

    Symmetric with :func:`~geny_executor.tools.base.build_tool`; handy
    for test harnesses that want a one-liner construction path.
    """
    return SkillTool(skill)
