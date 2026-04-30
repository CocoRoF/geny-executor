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
* **fork** (Phase 10.5) — the tool spawns a sub-pipeline with the
  skill's restricted tool roster + model override, lets it run to
  completion, and returns a summary. Requires the AgentTool runtime,
  which lands alongside isolation worktrees.

This module ships the inline path. Fork mode is stubbed with a clean
error so skills marked ``execution_mode: fork`` fail fast instead of
silently running inline.

Argument interpolation (Phase 10.1):
    Skills declare argument names in frontmatter (``arguments: [foo,
    bar]``) and reference them in the body as ``${foo}`` / ``${bar}``.
    The renderer substitutes invoke_args at execution time. Names
    that aren't passed in are replaced with the empty string, *not*
    the literal placeholder, so a skill body that depends on an
    optional argument doesn't dump ``${foo}`` into the LLM's mouth.
    Brace-style ``{name}`` tokens are still tolerated for
    backward-compat with skills written before 10.1.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.types import Skill, SkillContext
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.provider import ToolProvider


# ${name} placeholder. Names follow Python identifier rules (the only
# subset we need — every skill author is comfortable with that). The
# pattern is intentionally non-greedy so adjacent placeholders parse
# cleanly: ``${a}${b}`` matches twice.
_DOLLAR_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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

    Three-stage substitution:
      1. ``${name}`` placeholders (Phase 10.1, the recommended form) —
         missing names become empty string so optional-argument skills
         don't leak literal ``${foo}`` to the LLM.
      2. ``${SKILL_DIR}`` (Phase 10.3) — resolves to the directory the
         skill lives in (its ``assets_dir``). Missing source / in-code
         skills resolve to the empty string so authors get a clear
         "no assets here" signal in the rendered body.
      3. ``{name}`` brace placeholders (legacy, pre-10.1) — passed
         through unchanged when the key is missing so skill bodies
         containing genuine example braces don't crash.

    Non-string values are coerced with ``str``.
    """
    body = skill.body

    # Build the substitution map. Argument names always win over
    # built-in placeholders so a skill author who really wants to
    # rebind ``SKILL_DIR`` (rare, but possible) can.
    coerced: Dict[str, str] = {}
    skill_dir = skill.assets_dir
    coerced["SKILL_DIR"] = str(skill_dir) if skill_dir is not None else ""
    for k, v in (invoke_args or {}).items():
        coerced[k] = "" if v is None else str(v)

    def _replace(match: "re.Match[str]") -> str:
        return coerced.get(match.group(1), "")

    body = _DOLLAR_PLACEHOLDER.sub(_replace, body)

    # ``{name}`` brace style — legacy. Only run when the body still
    # contains braces and the operator passed args, to keep the common
    # case cheap and avoid disturbing literal markdown braces in
    # bodies that don't use brace-style at all.
    if invoke_args and "{" in body and "}" in body:
        safe = _SafeFormatDict(coerced)
        try:
            body = body.format_map(safe)
        except (ValueError, KeyError, IndexError):
            # Malformed format spec — return what we have rather
            # than blow up. The ${...} pass already substituted
            # everything we expected to substitute.
            pass

    return body


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
        head = self._skill.description + suffix
        # Phase 10.1 — surface the longer "when to use" copy to the
        # model so it can disambiguate between similarly-named skills.
        when = self._skill.metadata.when_to_use
        if when:
            head = f"{head}\n\nWhen to use: {when}"
        return head

    @property
    def input_schema(self) -> Dict[str, Any]:
        # Phase 10.1 — when the skill declares its arguments, surface
        # them to the model so it knows what shape ``args`` should be.
        # We document them in the description rather than hoisting to
        # named top-level properties so all skills present a uniform
        # schema (one ``args`` object) — easier for the LLM to learn.
        args_doc = (
            "Optional arguments for the skill. Schema "
            "defined by the skill body itself — consult "
            "the skill description."
        )
        declared = self._skill.metadata.arguments
        if declared:
            args_doc = (
                f"Arguments (declared by the skill author): "
                f"{', '.join(declared)}. "
                f"Reference them in the body as ${{name}}."
            )
            hint = self._skill.metadata.argument_hint
            if hint:
                args_doc = f"{args_doc} Hint: {hint}"
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "object",
                    "description": args_doc,
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
                    f"but the fork runtime (AgentTool isolation, Phase 10.5) "
                    f"is not yet available in this release. Mark the skill "
                    f"as execution_mode='inline' to run now."
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

        # Phase 10.2 — grant the skill's declared allowed_tools by
        # appending ALLOW rules to the live `ToolContext.permission_rules`.
        # The grant is *additive*: tools that were already allowed stay
        # allowed; tools the parent's permission rules denied get an
        # explicit ALLOW rule with lowest source priority. Persists for
        # the lifetime of this ToolContext (one session in the typical
        # wiring) — matches claude-code-main semantics where the
        # skill's permission context merges into the outer agent's.
        granted_tools = self._grant_allowed_tools(context)

        # Phase 10.3 — execute embedded shell blocks. ``${name}``
        # substitution already ran inside _render_body, so any args
        # the author wove into the shell command are already there.
        # MCP-sourced skills are stripped (trust_shell=False) so a
        # remote server can't inject shell commands into the host.
        from geny_executor.skills.shell_blocks import (
            execute_blocks,
            is_trusted_source,
        )

        trusted = is_trusted_source(self._skill.source, self._skill.metadata.extras)
        shell_summary = await execute_blocks(
            body,
            shell=self._skill.metadata.shell,
            cwd=context.working_dir or None,
            env=context.env_vars,
            timeout_s=self._skill.metadata.shell_timeout_s,
            trust_shell=trusted,
        )
        body = shell_summary.rendered_body

        header_lines = [
            f"Skill invoked: {self._skill.name} ({self._skill.id})",
        ]
        if self._skill.metadata.version:
            header_lines.append(f"version: {self._skill.metadata.version}")
        if self._skill.metadata.allowed_tools:
            tools_joined = ", ".join(self._skill.metadata.allowed_tools)
            granted_str = " (granted)" if granted_tools else " (already allowed)"
            header_lines.append(f"allowed tools: {tools_joined}{granted_str}")
        if self._skill.metadata.model_override:
            # Inline mode can't actually switch models — fork mode
            # (Phase 10.5) will. Keep the marker honest.
            header_lines.append(
                f"model override: {self._skill.metadata.model_override} (advisory in inline mode)"
            )
        if shell_summary.outcomes:
            ran = sum(1 for o in shell_summary.outcomes if not o.skipped)
            skipped = sum(1 for o in shell_summary.outcomes if o.skipped)
            failed = sum(
                1
                for o in shell_summary.outcomes
                if (not o.skipped) and (o.exit_code != 0 or o.timed_out)
            )
            parts = [f"{ran} ran"]
            if failed:
                parts.append(f"{failed} failed")
            if skipped:
                parts.append(f"{skipped} skipped (untrusted source)")
            header_lines.append(f"shell blocks: {', '.join(parts)}")

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
                "granted_tools": granted_tools,
                "model_override": self._skill.metadata.model_override,
                "args": raw_args,
                "body_chars": len(body),
                "rendered_template": body != self._skill.body,
                "shell_blocks_run": sum(1 for o in shell_summary.outcomes if not o.skipped),
                "shell_blocks_skipped": sum(1 for o in shell_summary.outcomes if o.skipped),
                "shell_blocks_failed": sum(
                    1
                    for o in shell_summary.outcomes
                    if (not o.skipped) and (o.exit_code != 0 or o.timed_out)
                ),
                "skill_context": {
                    "session_id": skill_ctx.session_id,
                    "parent_tool_use_id": skill_ctx.parent_tool_use_id,
                },
            },
        )

    def _grant_allowed_tools(self, context: ToolContext) -> List[str]:
        """Append ALLOW rules for the skill's declared tools to
        ``context.permission_rules``. Returns the list of tool names
        that received a *new* grant (already-permitted ones are
        skipped silently).

        The grant uses the executor's permission types so the matrix
        evaluator picks them up alongside any rules the host already
        loaded. Source is :class:`PermissionSource.PRESET_DEFAULT` —
        lowest priority, so an explicit user-level DENY still wins
        over a skill grant. This is the safe default: a skill saying
        "I want Bash" can be overridden by a sandbox env saying "no
        Bash, ever".
        """
        if not self._skill.metadata.allowed_tools:
            return []
        try:
            from geny_executor.permission.types import (
                PermissionBehavior,
                PermissionRule,
                PermissionSource,
            )
        except Exception:
            # Permission subsystem unavailable in this build — stay
            # advisory like pre-10.2 behaviour.
            return []
        existing: List[Any] = list(context.permission_rules)
        granted: List[str] = []
        # Track (tool_name, reason) tuples already granted so a
        # rapid re-invocation of the same skill doesn't pile up dupes.
        existing_grants = {
            (getattr(r, "tool_name", None), getattr(r, "reason", None)) for r in existing
        }
        reason = f"granted by skill {self._skill.id}"
        for tool_name in self._skill.metadata.allowed_tools:
            key = (tool_name, reason)
            if key in existing_grants:
                continue
            rule = PermissionRule(
                tool_name=tool_name,
                behavior=PermissionBehavior.ALLOW,
                source=PermissionSource.PRESET_DEFAULT,
                pattern=None,
                reason=reason,
            )
            existing.append(rule)
            granted.append(tool_name)
        # Re-bind in case ``permission_rules`` was a tuple or otherwise
        # immutable on a particular ToolContext implementation.
        context.permission_rules = existing
        return granted


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

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        name: str = "skills",
        active_paths: Iterable[str] = (),
    ):
        """
        Args:
            registry: Skill registry to expose.
            name: Provider name surfaced in pipeline diagnostics.
            active_paths: Paths the session is currently working on.
                When non-empty, skills with declared ``paths``
                patterns are filtered to only those that match — the
                model's tool roster shrinks to skills relevant to
                the current task. Skills without declared ``paths``
                are always included. Phase 10.2 conditional
                activation; mutate via :meth:`set_active_paths`.
        """
        self._registry = registry
        self._name = name
        self._active_paths: List[str] = list(active_paths)

    def set_active_paths(self, paths: Iterable[str]) -> None:
        """Update the active path set. Hosts call this when a tool
        Read / Write / Edit touches a new path; the next
        :meth:`list_tools` call reflects the change."""
        self._active_paths = list(paths)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        count = len(self._registry)
        return f"Skill tools ({count} skill{'s' if count != 1 else ''})"

    def list_tools(self) -> List[Tool]:
        # Phase 10.1 — honour the per-skill
        # ``disable_model_invocation`` flag so user-only slash commands
        # don't leak into the model's tool roster. The registry still
        # has them (so a user-driven slash command path can resolve
        # them); only this provider filters.
        # Phase 10.2 — also filter by `paths` conditional activation.
        out: List[Tool] = []
        for skill in self._registry.list_all():
            if skill.metadata.disable_model_invocation:
                continue
            if not _skill_active_for_paths(skill, self._active_paths):
                continue
            out.append(SkillTool(skill))
        return out


def _skill_active_for_paths(skill: Skill, active_paths: Iterable[str]) -> bool:
    """Return True iff the skill should be exposed given the active
    path set.

    A skill with no declared ``paths`` is always active. A skill with
    declared ``paths`` is active when any of those patterns matches
    any of the active paths. Empty active_paths + declared paths
    means the skill is hidden.
    """
    declared = skill.metadata.paths
    if not declared:
        return True
    if not active_paths:
        return False
    from geny_executor.skills.path_match import compile_patterns, match_any

    # Compile per-call — the active path set churns more often than
    # the skill list, so caching the regexes on the Skill would help
    # only marginally; compile cost is microseconds for the small N
    # we expect (<10 patterns per skill, <50 skills).
    compiled = compile_patterns(declared)
    return match_any(active_paths, compiled)


def build_skill_tool(skill: Skill) -> SkillTool:
    """Factory equivalent to ``SkillTool(skill)``.

    Symmetric with :func:`~geny_executor.tools.base.build_tool`; handy
    for test harnesses that want a one-liner construction path.
    """
    return SkillTool(skill)
