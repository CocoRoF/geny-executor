"""Generalized, app-neutral sub-agent / sub-worker type catalog.

geny-executor ships a small set of GENERALIZED sub-agent types — ``worker``,
``researcher``, ``summarizer``, ``critic`` — as plain specifications
(:class:`SubagentTypeSpec`): an ``agent_type``, a human description, an
``allowed_tools`` shape, and a strong default ``system_prompt``. These are
deliberately app-neutral; nothing here knows about any particular product.

A host turns a spec into a runnable
:class:`~geny_executor.stages.s12_agent.subagent_type.SubagentTypeDescriptor`
by attaching its own pipeline factory (the library does not prescribe how a
host builds pipelines) — see :func:`specs_to_descriptors`. Hosts are expected
to subset / override / extend the catalog freely.

:data:`DEFAULT_PERSISTENT_SUBAGENT_PROMPT` is the strong default persona for an
owned, long-lived *companion* sub-agent when the host pins no custom role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

__all__ = [
    "SubagentTypeSpec",
    "BUILTIN_SUBAGENT_TYPES",
    "DEFAULT_PERSISTENT_SUBAGENT_PROMPT",
    "default_subagent_specs",
    "specs_to_descriptors",
]


@dataclass(frozen=True)
class SubagentTypeSpec:
    """App-neutral specification of a generalized sub-agent type.

    Carries everything a host needs to construct a descriptor *except* the
    factory (host-owned). ``allowed_tools=()`` means "inherit / unrestricted"
    by host convention; ``provider=None`` means "inherit the parent's".
    """

    agent_type: str
    description: str
    allowed_tools: Tuple[str, ...] = ()
    system_prompt: Optional[str] = None
    provider: Optional[str] = None


_WORKER_PROMPT = (
    "You are an autonomous worker sub-agent. You receive a single, "
    "fully-specified task and complete it end-to-end with the tools available "
    "to you. Work methodically: understand the goal, take the necessary "
    "actions, verify the result, and return a concise report of what you did "
    "and any caveats. Do not ask clarifying questions unless genuinely "
    "blocked — make reasonable assumptions and state them. Stay strictly "
    "within the delegated task."
)

_RESEARCHER_PROMPT = (
    "You are a research sub-agent restricted to read-only investigation. Use "
    "only read/search tools and never mutate state. Gather evidence from the "
    "codebase and, where available, the web; cross-check claims; and return "
    "well-organized findings with concrete citations (file:line, URLs). "
    "Distinguish what you verified from what you inferred. Be exhaustive on "
    "the question asked and nothing more."
)

_SUMMARIZER_PROMPT = (
    "You are a summarization sub-agent. Produce a faithful, compact summary "
    "that preserves the key facts, decisions, action items, and unresolved "
    "questions of the source material. Never invent details or drop "
    "load-bearing specifics. Prefer structured output (short sections / "
    "bullets) when it aids clarity."
)

_CRITIC_PROMPT = (
    "You are a rigorous review sub-agent. Examine the target for real defects "
    "— correctness bugs, security issues, edge cases, and unmet requirements "
    "— using read-only inspection. Report only substantiated findings, each "
    "with a concrete location and why it matters; rank by severity. Prefer "
    "surfacing a genuine concern over staying silent, but do not pad with "
    "speculation."
)


#: The strong default persona for an owned, persistent *companion* sub-agent.
DEFAULT_PERSISTENT_SUBAGENT_PROMPT = (
    "You are a persistent companion sub-agent owned by a primary agent. The "
    "primary delegates whole tasks to you and trusts you to carry them out "
    "autonomously and reliably while it continues its own work.\n\n"
    "Operate end-to-end: clarify the objective, plan briefly, execute with "
    "the tools available to you, verify your output, then report back a "
    "clear, actionable result — what you accomplished, anything you changed, "
    "and any follow-ups or risks. You persist across turns, so carry context "
    "forward and build on prior work rather than restarting.\n\n"
    "Be proactive and thorough. Make sound assumptions and state them instead "
    "of stalling; surface a blocker only when you genuinely cannot proceed. "
    "Stay within the scope you were given and keep your final report concise "
    "and high-signal."
)


#: The generalized catalog. App-neutral; hosts subset / override / extend.
BUILTIN_SUBAGENT_TYPES: Tuple[SubagentTypeSpec, ...] = (
    SubagentTypeSpec(
        agent_type="worker",
        description=(
            "General-purpose autonomous worker. Full default toolset; "
            "completes a delegated task end-to-end."
        ),
        allowed_tools=("*",),
        system_prompt=_WORKER_PROMPT,
    ),
    SubagentTypeSpec(
        agent_type="researcher",
        description=(
            "Read-only investigation. Read / Grep / Glob / WebFetch / "
            "WebSearch only — cannot mutate state."
        ),
        allowed_tools=("Read", "Grep", "Glob", "WebFetch", "WebSearch"),
        system_prompt=_RESEARCHER_PROMPT,
    ),
    SubagentTypeSpec(
        agent_type="summarizer",
        description=(
            "Faithful, compact summarization of supplied material. Suited "
            "for context-overflow compaction."
        ),
        allowed_tools=(),
        system_prompt=_SUMMARIZER_PROMPT,
    ),
    SubagentTypeSpec(
        agent_type="critic",
        description=(
            "Rigorous read-only review — surfaces substantiated defects "
            "ranked by severity."
        ),
        allowed_tools=("Read", "Grep", "Glob"),
        system_prompt=_CRITIC_PROMPT,
    ),
)


def default_subagent_specs() -> Tuple[SubagentTypeSpec, ...]:
    """Return the generalized built-in sub-agent type specs."""
    return BUILTIN_SUBAGENT_TYPES


def specs_to_descriptors(
    factory: Callable[..., Any],
    specs: Optional[Tuple[SubagentTypeSpec, ...]] = None,
) -> List[Any]:
    """Build :class:`SubagentTypeDescriptor` s from specs + a host factory.

    Convenience for hosts that want the generalized catalog wired with their
    own pipeline factory. Tolerates older descriptor signatures by dropping
    fields the installed dataclass doesn't accept.
    """
    from geny_executor.stages.s12_agent.subagent_type import (
        SubagentTypeDescriptor,
    )

    out: List[Any] = []
    for s in specs or BUILTIN_SUBAGENT_TYPES:
        kwargs = dict(
            agent_type=s.agent_type,
            factory=factory,
            description=s.description,
            allowed_tools=s.allowed_tools,
            provider=s.provider,
            system_prompt=s.system_prompt,
        )
        try:
            out.append(SubagentTypeDescriptor(**kwargs))
            continue
        except TypeError:
            pass
        kwargs.pop("system_prompt", None)
        try:
            out.append(SubagentTypeDescriptor(**kwargs))
        except TypeError:
            out.append(
                SubagentTypeDescriptor(
                    agent_type=s.agent_type,
                    factory=factory,
                    description=s.description,
                )
            )
    return out
