"""Skill type system.

Cycle 20260424 executor uplift — Phase 4 Week 7 (Skills foundation).

A **Skill** is a code-free capability unit: a ``SKILL.md`` file with a
YAML frontmatter block and a markdown body. The frontmatter declares
metadata (name, description, model override, allowed tools, execution
mode); the body is the prompt the model sees when the skill is invoked.

This module defines the in-memory representation. Loading from disk
lives in :mod:`~geny_executor.skills.loader`; registration lives in
:mod:`~geny_executor.skills.registry`; the Tool wrapper that exposes
a skill to the LLM lives in :mod:`~geny_executor.skills.skill_tool`.

Design decisions:

* ``id`` is the unique identifier, derived from the skill's location
  on disk (``<parent-dir>/<skill-name>``) or set explicitly by bundled
  skills. Hosts that expose skills through the Tool surface see the
  id used as the tool name.
* ``allowed_tools`` is a tuple of tool names the skill may call. Empty
  tuple means "no restriction" — the skill inherits the parent agent's
  full tool roster. Non-empty means strict allowlist.
* ``execution_mode`` encodes how the host should run the skill:
  ``inline`` (same pipeline, shares state) or ``fork`` (spawn a
  subagent). Phase 4 ships inline; fork arrives with Phase 7.
* ``model_override`` lets a skill opt into a specific model (e.g. a
  Skill that does heavy reasoning picks opus even when the host
  session runs haiku). ``None`` means "inherit host".
* ``source`` records where the skill was loaded from — useful when
  audit / debug surfaces want to show "this skill came from
  ``/project/.skills/refactor/SKILL.md``" vs "shipped with the
  executor".

See ``executor_uplift/08_design_skills.md`` and
``executor_uplift/12_detailed_plan.md`` §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_VALID_EXECUTION_MODES = ("inline", "fork")


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata parsed from a SKILL.md frontmatter block.

    Only the fields the executor understands are surfaced here — any
    extra keys in the frontmatter go to :attr:`extras` so hosts /
    plugins can carry their own metadata without a schema change.

    Field requirements:
        * ``name`` — required. Human-readable label; not necessarily
          unique (multiple skill files can share a name if they live
          in different directories).
        * ``description`` — required. One-line summary shown to the
          LLM via the Tool description.
        * ``version`` — optional semver or arbitrary string.
        * ``allowed_tools`` — tuple of tool names. Empty = no
          restriction (inherit parent's tool roster).
        * ``model_override`` — canonical model id (e.g.
          ``"claude-opus-4-7"``) or ``None`` to inherit host.
        * ``execution_mode`` — ``"inline"`` or ``"fork"``.
        * ``category`` / ``effort`` / ``examples`` — discovery hints
          (PR-B.4.1).
        * ``arguments`` — declared argument names. Body uses
          ``${name}`` placeholders to interpolate them; missing names
          render as the empty string instead of leaking the literal
          ``${...}`` to the model. (Phase 10.1.)
        * ``argument_hint`` — short human-readable hint shown by CLI /
          slash-command autocomplete (e.g. ``"<file> [count]"``).
        * ``when_to_use`` — extra usage guidance for *model-side*
          discovery; surfaced separately from ``description`` so the
          short one-liner stays scannable while the longer rationale
          is available when the model is choosing whether to call the
          skill.
        * ``user_invocable`` — if False, the skill cannot be triggered
          via the user's slash-command interface (still callable by
          the model through ``SkillTool``). Defaults to True for
          on-disk skills; bridges (e.g. MCP prompts) can set False to
          keep them out of the slash-command palette.
        * ``disable_model_invocation`` — if True, the model cannot
          call the skill via ``SkillTool``; only user-driven slash
          commands fire it. Use for skills that *must* originate from
          the human in the loop (e.g. risky destructive workflows).
        * ``extras`` — every frontmatter key the executor doesn't own.
    """

    name: str
    description: str
    version: Optional[str] = None
    allowed_tools: Tuple[str, ...] = ()
    model_override: Optional[str] = None
    execution_mode: str = "inline"
    # PR-B.4.1 — richer schema. All optional with defaults so existing
    # SKILL.md files load unchanged. ``category`` slots the skill into
    # the help / discovery UI, ``effort`` hints expected token+time
    # cost, ``examples`` are the LLM-visible "use it like this" snippets.
    category: Optional[str] = None
    effort: Optional[str] = None  # "low" | "medium" | "high" (free string for forward-compat)
    examples: Tuple[str, ...] = ()
    # Phase 10.1 (skills uplift) — argument schema + invocation flags +
    # discovery copy. All optional, all backward compatible.
    arguments: Tuple[str, ...] = ()
    argument_hint: Optional[str] = None
    when_to_use: Optional[str] = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    # Phase 10.2 — conditional activation. Empty = always active.
    # Non-empty = the skill is hidden from `SkillToolProvider` until
    # one of the patterns matches a path the session is working with.
    # Patterns use a small gitignore-ish subset (``*``, ``**``, ``?``,
    # leading ``/`` to anchor at root, trailing ``/`` for dir-only);
    # see :mod:`geny_executor.skills.path_match`.
    paths: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Skill:
    """A loaded skill — metadata + prompt body + source hint.

    Immutable on purpose: once a skill has been parsed from disk (or
    built in code for bundled skills) it should not mutate. Hosts that
    want to override behaviour per-invocation should do so through
    :class:`SkillContext` rather than rewriting the :class:`Skill`.
    """

    id: str
    metadata: SkillMetadata
    body: str
    source: Optional[Path] = None

    @property
    def name(self) -> str:
        """Shortcut to ``metadata.name`` — the human-readable label."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Shortcut to ``metadata.description``."""
        return self.metadata.description


@dataclass
class SkillContext:
    """Runtime data passed to a skill when it is invoked.

    Gives the skill access to the subset of pipeline state it needs
    without exposing the full ``PipelineState`` (which would make the
    skill promise too much). Hosts building a :class:`SkillTool`
    populate this on each call.

    Attributes:
        skill: The :class:`Skill` being invoked.
        parent_tool_use_id: ID of the LLM tool_use block that triggered
            this invocation. Lets audit trails link sub-invocations
            back to the parent turn.
        invoke_args: Arbitrary dict of arguments the skill author can
            declare in frontmatter. Mirrors the pattern of
            ``ToolContext.extras``.
        session_id: Propagates from the outer session so
            ``{storage_path}/skills/...`` logs can attribute work
            correctly.
        working_dir: Same semantics as ``ToolContext.working_dir``.
    """

    skill: Skill
    parent_tool_use_id: Optional[str] = None
    invoke_args: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    working_dir: str = ""


# Helpers ------------------------------------------------------------


def validate_execution_mode(mode: str) -> str:
    """Return *mode* if it is valid, else raise ``ValueError``.

    Kept as a small helper so loader / registry can reuse the same
    allowlist without importing the tuple from elsewhere.
    """
    if mode not in _VALID_EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {_VALID_EXECUTION_MODES}; got {mode!r}")
    return mode
