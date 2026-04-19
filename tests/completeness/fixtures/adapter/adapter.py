"""`GenyManagerAdapter` — MemoryProvider facade over an in-memory delegate.

This is the C7-parity fixture. Its job is to present the same
`MemoryProvider` surface as the native providers but tag its
descriptor as a distinct backend name (`"geny-adapter"`) so the
parity suite can tell outputs apart.

In Phase 3 the delegate will switch to a wrapper around Geny's
`SessionMemoryManager`. Until then the adapter holds an
`EphemeralMemoryProvider` instance and forwards every Protocol call
to it — which gives C7 a stable round-trippable implementation to
author scenario recordings against.

Keeping the facade separate from the delegate (rather than inheriting)
is intentional: it forces the Phase 3 swap to change exactly one
class and preserves Protocol conformance as a testable boundary.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from geny_executor.memory.provider import (
    BackendInfo,
    CuratedHandle,
    ExecutionSummary,
    GlobalHandle,
    IndexHandle,
    Insight,
    LTMHandle,
    Layer,
    MemoryDescriptor,
    MemoryProvider,
    MemorySnapshot,
    NoteRef,
    NotesHandle,
    RecordReceipt,
    ReflectionContext,
    RetrievalQuery,
    RetrievalResult,
    STMHandle,
    Scope,
    Turn,
    VectorHandle,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider


class GenyManagerAdapter(MemoryProvider):
    """MemoryProvider wrapping an in-memory delegate for C7 parity.

    The adapter *is* a MemoryProvider (duck-typed via the Protocol);
    it's not a subclass of EphemeralMemoryProvider because Phase 3
    needs the freedom to swap the delegate for a Geny
    `SessionMemoryManager` wrapper without touching call sites.
    """

    NAME = "geny-adapter"
    VERSION = "0.1.0"

    def __init__(
        self,
        *,
        scope: Scope = Scope.SESSION,
    ) -> None:
        self._delegate = EphemeralMemoryProvider(scope=scope)

    @property
    def descriptor(self) -> MemoryDescriptor:
        inner = self._delegate.descriptor
        # Re-tag the descriptor so parity tests can distinguish the
        # adapter's output from the native provider's output even when
        # the layer/capability sets match.
        return replace(
            inner,
            name=self.NAME,
            version=self.VERSION,
            backends=(
                BackendInfo(
                    layer=Layer.NOTES,
                    backend="geny-adapter",
                    metadata={"delegate": inner.name},
                ),
            ),
        )

    # ── lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        await self._delegate.initialize()

    async def close(self) -> None:
        await self._delegate.close()

    # ── layer handles ───────────────────────────────────────────────

    def stm(self) -> STMHandle:
        return self._delegate.stm()

    def ltm(self) -> LTMHandle:
        return self._delegate.ltm()

    def notes(self) -> NotesHandle:
        return self._delegate.notes()

    def vector(self) -> Optional[VectorHandle]:
        return self._delegate.vector()

    def curated(self) -> Optional[CuratedHandle]:
        return self._delegate.curated()

    def global_(self) -> Optional[GlobalHandle]:
        return self._delegate.global_()

    def index(self) -> IndexHandle:
        return self._delegate.index()

    # ── cross-layer ─────────────────────────────────────────────────

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return await self._delegate.retrieve(query)

    async def record_turn(self, turn: Turn) -> None:
        await self._delegate.record_turn(turn)

    async def record_execution(self, summary: ExecutionSummary) -> RecordReceipt:
        return await self._delegate.record_execution(summary)

    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]:
        return await self._delegate.reflect(ctx)

    async def snapshot(self) -> MemorySnapshot:
        inner = await self._delegate.snapshot()
        return replace(inner, provider=self.NAME, version=self.VERSION)

    async def restore(self, snap: MemorySnapshot) -> None:
        await self._delegate.restore(replace(snap, provider=self._delegate.NAME))

    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef:
        return await self._delegate.promote(ref, to)


__all__ = ["GenyManagerAdapter"]
