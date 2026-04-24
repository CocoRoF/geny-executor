"""DynamicPersonaPromptBuilder — calls a PersonaProvider on every build.

Cycle 20260424 executor uplift — Phase 7 Sprint S7.1.

Replaces the fixed-block ``ComposablePromptBuilder`` attached to
Stage 3 when the host needs per-turn persona resolution. The builder
holds no persona state itself — every call to :meth:`build` flows
through ``provider.resolve`` so updates to the provider are visible
on the next turn without rebuilding the pipeline.

Build sequence:

1. Read ``session_meta`` (frozen at construction; session-scoped).
2. Call ``provider.resolve(state, session_meta=...)`` to get the
   current :class:`PersonaResolution`.
3. Compose a fresh :class:`ComposablePromptBuilder` from the
   resolved ``persona_blocks`` + the static ``tail_blocks``
   (DateTimeBlock, MemoryContextBlock, …) supplied at construction.
4. If ``resolution.system_tail`` is set, append it as one final
   inline block.
5. Return the composed output (string or content-block list)
   matching :class:`ComposablePromptBuilder`'s shape — Stage 3
   ``execute`` accepts both.

The builder is deliberately thin so the provider implementation
(host-specific) can evolve without touching the builder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.artifact.default.builders import (
    ComposablePromptBuilder,
)
from geny_executor.stages.s03_system.interface import PromptBlock, PromptBuilder
from geny_executor.stages.s03_system.persona.provider import (
    PersonaProvider,
    PersonaResolution,
)


class DynamicPersonaPromptBuilder(PromptBuilder):
    """PromptBuilder that calls a :class:`PersonaProvider` on every build.

    See module docstring for the full contract. Construct once per
    session with a stable ``session_meta`` mapping; the executor's
    Stage 3 will keep calling ``build`` once per turn.
    """

    def __init__(
        self,
        provider: PersonaProvider,
        *,
        session_meta: Optional[Dict[str, Any]] = None,
        tail_blocks: Optional[List[PromptBlock]] = None,
        separator: str = "\n\n",
        use_content_blocks: bool = False,
    ):
        self._provider = provider
        # Defensive copy — the host may keep mutating the dict it
        # passed in, but the builder needs a stable snapshot.
        self._session_meta = dict(session_meta or {})
        self._tail_blocks: List[PromptBlock] = list(tail_blocks or [])
        self._separator = separator
        self._use_content_blocks = use_content_blocks

    @property
    def name(self) -> str:
        return "dynamic_persona"

    @property
    def description(self) -> str:
        return "Per-turn persona resolution via PersonaProvider"

    def configure(self, config: Dict[str, Any]) -> None:
        # Builder has no plain-config knobs (provider + session_meta
        # are runtime objects); nothing to configure from a dict.
        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "session_meta_keys": sorted(self._session_meta.keys()),
            "tail_block_names": [b.name for b in self._tail_blocks],
            "use_content_blocks": self._use_content_blocks,
        }

    @property
    def session_meta(self) -> Dict[str, Any]:
        """Read-only view of the session-scoped mapping."""
        return dict(self._session_meta)

    @property
    def provider(self) -> PersonaProvider:
        """The wrapped provider (audit / debug / hot-swap helper)."""
        return self._provider

    def build(self, state: PipelineState) -> Union[str, List[Dict[str, Any]]]:
        resolution: PersonaResolution = self._provider.resolve(
            state, session_meta=self._session_meta
        )

        blocks: List[PromptBlock] = []
        blocks.extend(resolution.persona_blocks)
        blocks.extend(self._tail_blocks)
        if resolution.system_tail:
            blocks.append(_TailTextBlock(resolution.system_tail))

        inner = ComposablePromptBuilder(
            blocks=blocks,
            separator=self._separator,
            use_content_blocks=self._use_content_blocks,
        )
        return inner.build(state)


class _TailTextBlock(PromptBlock):
    """Inline wrapper for ``PersonaResolution.system_tail`` text.

    Emits a single block named ``persona_tail`` carrying the tail
    text verbatim. Kept private — hosts that want richer tail
    content should add full :class:`PromptBlock` instances to
    ``tail_blocks`` instead.
    """

    def __init__(self, text: str):
        self._text = text

    @property
    def name(self) -> str:
        return "persona_tail"

    def render(self, state: PipelineState) -> str:
        return self._text
