"""PersonaProvider Protocol + PersonaResolution dataclass.

Cycle 20260424 executor uplift — Phase 7 Sprint S7.1.

A PersonaProvider produces a fresh :class:`PersonaResolution` on every
turn. The :class:`DynamicPersonaPromptBuilder` calls
``provider.resolve(state, session_meta=...)`` from inside Stage 3's
build path and weaves the returned blocks into the system prompt.

``resolve`` is **synchronous** by contract — Stage 3's
``execute → builder.build`` chain is sync, so smuggling async through
that boundary would require an event-loop hack. All provider-side I/O
(character markdown reads, DB lookups, persona graph queries) should
be cached at provider construction time or memoised on first use.

Implementations must be safe for concurrent sessions — typically by
keying any mutable state on ``session_meta["session_id"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from geny_executor.stages.s03_system.interface import PromptBlock


@dataclass(frozen=True)
class PersonaResolution:
    """Single-turn persona snapshot returned by :meth:`PersonaProvider.resolve`.

    Attributes:
        persona_blocks: Ordered :class:`PromptBlock` instances that
            produce the persona portion of the system prompt. Rendered
            in list order. Empty list is valid (yields an empty
            persona section).
        system_tail: Optional text appended **after** all other
            system-prompt blocks. Reserved for short, high-volatility
            content that should not sit in a cache-controlled block —
            e.g. "Today is [date]" lines or per-turn mood updates.
        cache_key: Stable token summarising the inputs that produced
            this resolution. Identical ``cache_key`` across turns
            signals to cache-aware downstream components that the
            persona section can be reused unchanged. Empty string
            means "do not cache".
    """

    persona_blocks: List[PromptBlock] = field(default_factory=list)
    system_tail: Optional[str] = None
    cache_key: str = ""


@runtime_checkable
class PersonaProvider(Protocol):
    """Per-turn persona resolver.

    Pass an instance to :class:`DynamicPersonaPromptBuilder`.
    Geny-style hosts typically implement a provider that:

    * loads character markdown / persona graph at construction time;
    * caches per-session VTuber state keyed on
      ``session_meta["session_id"]``;
    * returns a :class:`PersonaResolution` whose ``persona_blocks``
      include a persona block + a current-mood block.

    The protocol is structural (``runtime_checkable``) so duck-typed
    implementations work without inheritance.
    """

    def resolve(
        self,
        state: Any,
        *,
        session_meta: Dict[str, Any],
    ) -> PersonaResolution:
        """Return the persona snapshot to use for this turn.

        Args:
            state: Live :class:`~geny_executor.core.state.PipelineState`.
                Typed as ``Any`` to avoid an import cycle from
                Protocol consumers.
            session_meta: Provider-agnostic session-scoped mapping
                (``session_id``, ``character_id``, host-specific
                fields). Frozen at builder construction time.

        Returns:
            A :class:`PersonaResolution` describing this turn's
            persona surface.
        """
        ...
