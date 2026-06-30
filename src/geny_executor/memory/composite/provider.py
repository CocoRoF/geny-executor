"""CompositeMemoryProvider — per-layer routing across native providers.

Holds a `LayerRouting` table mapping each `Layer` to the underlying
`MemoryProvider` that owns it. The composite delegates per-handle
calls (`stm()` → routing.STM.stm(), `ltm()` → routing.LTM.ltm(), …)
and orchestrates cross-layer methods (`record_execution`, `retrieve`,
`snapshot`, `restore`, `promote`) so callers see one provider with
one descriptor.

The composite is the only provider where `promote(ref, to)` does
real work: when `routing.scope_providers` declares a provider for the
target scope, the note is copied from its source-scope provider into
the target-scope provider's `notes()` handle, then deleted from the
source. This is what lets a session note become a curated user-scoped
note without the calling stage needing to know which backends are
behind which scopes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set

from geny_executor.memory.composite.handles import (
    _CompositeCuratedHandle,
    _CompositeGlobalHandle,
)
from geny_executor.memory.composite.routing import LayerRouting
from geny_executor.memory.graph_rank import personalized_pagerank
from geny_executor.memory.composite.snapshot import decode_snapshot, encode_snapshot
from geny_executor.memory.provider import (
    BackendInfo,
    Capability,
    CuratedHandle,
    EmbeddingDescriptor,
    ExecutionSummary,
    GlobalHandle,
    Importance,
    IndexHandle,
    Insight,
    Layer,
    LTMHandle,
    MemoryDescriptor,
    MemoryHooks,
    MemoryProvider,
    MemorySnapshot,
    NoteDraft,
    NoteRef,
    NotesHandle,
    RecordReceipt,
    ReflectionContext,
    RetrievalQuery,
    RetrievalResult,
    Scope,
    STMHandle,
    Turn,
    VectorHandle,
)
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)


class CompositeMemoryProvider(MemoryProvider):
    """Routes each `Layer` to a distinct underlying provider.

    Construction is cheap; `initialize()` initialises every distinct
    delegate exactly once. `close()` mirrors that, in reverse.
    """

    NAME = "composite"
    VERSION = "1.0.0"

    def __init__(
        self,
        routing: LayerRouting,
        *,
        scope: Scope = Scope.SESSION,
        session_id: str = "",
        user_id: str = "",
    ) -> None:
        self._routing = routing
        self._scope = scope
        self._session_id = session_id
        self._user_id = user_id
        self._descriptor = self._build_descriptor()

    # ── MemoryProvider: descriptor + lifecycle ─────────────────────

    @property
    def descriptor(self) -> MemoryDescriptor:
        return self._descriptor

    @property
    def routing(self) -> LayerRouting:
        return self._routing

    async def initialize(self) -> None:
        for delegate in self._routing.distinct_providers():
            await delegate.initialize()

    async def close(self) -> None:
        for delegate in self._routing.distinct_providers():
            await delegate.close()

    def set_hooks(self, hooks: MemoryHooks) -> None:
        """Forward `MemoryHooks` to every distinct scope provider.

        The composite itself doesn't own STM/Notes — it only routes
        layer calls to underlying scope providers (session, user_curated,
        global). Hooks must reach the actual store layer where
        ``after_record_turn`` / ``after_note_write`` actually fire,
        so we install on every distinct delegate.
        """
        self._hooks = hooks
        for delegate in self._routing.distinct_providers():
            if hasattr(delegate, "set_hooks"):
                try:
                    delegate.set_hooks(hooks)
                except Exception:  # noqa: BLE001
                    # Hook installation is best-effort; a misbehaving
                    # delegate must not abort the composite. Hosts that
                    # need load-bearing behaviour can inspect each
                    # delegate themselves.
                    pass

    # ── layer handles ───────────────────────────────────────────────

    def stm(self) -> STMHandle:
        return self._require(Layer.STM).stm()

    def ltm(self) -> LTMHandle:
        return self._require(Layer.LTM).ltm()

    def notes(self) -> NotesHandle:
        return self._require(Layer.NOTES).notes()

    def vector(self) -> Optional[VectorHandle]:
        prov = self._routing.provider_for(Layer.VECTOR)
        if prov is None:
            return None
        return prov.vector()

    def curated(self) -> Optional[CuratedHandle]:
        """Resolve the user-scoped curated handle.

        Two routing paths are accepted:
          1. ``layers[CURATED] = <provider>`` — explicit per-layer
             routing, the rest of the composite already understands
             this shape.
          2. ``scope_providers[USER] = <provider>`` — preferred when
             curated knowledge lives at user scope alongside other
             user-only artefacts. Picked if `layers[CURATED]` is
             absent.

        The returned handle wraps the target provider's `NotesHandle`
        / `VectorHandle` and binds `promote_from_session` to the
        composite's session-scope source, so a stage that calls
        ``provider.curated().promote_from_session(ref)`` does not need
        to know which underlying providers serve which scope.
        """
        target = self._routing.provider_for(Layer.CURATED) or self._routing.scope_provider(
            Scope.USER
        )
        if target is None:
            return None
        # Native curated handle wins if the target provider implements
        # one (e.g. a future host-side provider that owns the curated
        # plane natively); otherwise wrap the target's notes layer.
        native = target.curated()
        if native is not None:
            return native
        source = self._routing.scope_provider(Scope.SESSION) or self._require(Layer.NOTES)
        return _CompositeCuratedHandle(
            user_id=self._user_id,
            target=target,
            source=source,
        )

    def global_(self) -> Optional[GlobalHandle]:
        """Resolve the cross-session global handle.

        Mirrors `curated()`: accepts either ``layers[GLOBAL]`` or
        ``scope_providers[GLOBAL]``. Wraps the target provider's
        notes/vector handles and binds `promote_from` to the
        composite's session source.
        """
        target = self._routing.provider_for(Layer.GLOBAL) or self._routing.scope_provider(
            Scope.GLOBAL
        )
        if target is None:
            return None
        native = target.global_()
        if native is not None:
            return native
        source = self._routing.scope_provider(Scope.SESSION) or self._require(Layer.NOTES)
        return _CompositeGlobalHandle(target=target, source=source)

    def index(self) -> IndexHandle:
        return self._require(Layer.INDEX).index()

    def _require(self, layer: Layer) -> MemoryProvider:
        prov = self._routing.provider_for(layer)
        if prov is None:
            raise RuntimeError(
                f"composite provider has no delegate for required layer {layer.value!r}"
            )
        return prov

    # ── cross-layer ─────────────────────────────────────────────────

    async def record_turn(self, turn: Turn) -> None:
        await self._require(Layer.STM).stm().append(turn)

    async def record_execution(self, summary: ExecutionSummary) -> RecordReceipt:
        files: List[str] = []
        receipt = RecordReceipt()
        if not summary.final_text:
            receipt.files_updated = files
            return receipt

        qa_body = f"## Q\n{summary.user_input}\n\n## A\n{summary.final_text}".strip()
        ltm_ref = await self._require(Layer.LTM).ltm().write_dated(qa_body)
        files.append(ltm_ref.filename)

        notes = self._require(Layer.NOTES).notes()
        meta = await notes.write(
            NoteDraft(
                title=(summary.user_input or "execution")[:80],
                body=summary.final_text,
                importance=Importance.MEDIUM,
                tags=list(summary.tags),
                category="insights",
                scope=self._scope,
            )
        )
        files.append(meta.ref.filename)
        receipt.notes_written = 1

        # Auto-vector wiring inside the underlying notes store has
        # already embedded the body. Surface the chunk count for
        # callers that report on it.
        if self.vector() is not None:
            receipt.vector_chunks = 1

        return _attach_files(receipt, files)

    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]:
        # Composite is a router, not a reflector; the orchestrating
        # stage is expected to plug an LLM in via MemoryHooks.
        return ()

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        chunks: List[MemoryChunk] = []
        breakdown: Dict[Layer, int] = {}

        if Layer.STM in query.layers and self._routing.has_layer(Layer.STM):
            recent = await self._require(Layer.STM).stm().recent(n=query.max_per_layer)
            stm_chunks = [
                MemoryChunk(
                    key=f"stm-{i}",
                    content=_turn_to_text(t),
                    source="recent_message",
                    relevance_score=0.0,
                )
                for i, t in enumerate(recent)
            ]
            chunks.extend(stm_chunks)
            breakdown[Layer.STM] = len(stm_chunks)

        if Layer.LTM in query.layers and self._routing.has_layer(Layer.LTM):
            ltm = self._require(Layer.LTM).ltm()
            main_text = await ltm.read_main()
            ltm_chunks: List[MemoryChunk] = []
            if main_text:
                ltm_chunks.append(
                    MemoryChunk(
                        key="MEMORY.md",
                        content=main_text[:2000],
                        source="long_term",
                        relevance_score=1.0,
                    )
                )
            if query.text:
                ltm_chunks.extend(await ltm.search(query.text, limit=query.max_per_layer))
            chunks.extend(ltm_chunks)
            breakdown[Layer.LTM] = len(ltm_chunks)

        if Layer.NOTES in query.layers and self._routing.has_layer(Layer.NOTES) and query.text:
            note_chunks = (
                await self._require(Layer.NOTES)
                .notes()
                .search(
                    query.text,
                    limit=query.max_per_layer,
                    importance_floor=query.importance_floor,
                )
            )
            chunks.extend(note_chunks)
            breakdown[Layer.NOTES] = len(note_chunks)

        if Layer.VECTOR in query.layers and query.text:
            vector = self.vector()
            if vector is not None:
                vec_chunks = await vector.search(query.text, top_k=query.max_per_layer)
                chunks.extend(vec_chunks)
                breakdown[Layer.VECTOR] = len(vec_chunks)

        # ── Graph-aware additive expansion (opt-in via hooks.graph_aware) ──
        # Append graph-connected notes (Personalized PageRank over the
        # knowledge-graph edges, seeded by the direct note hits) AFTER the
        # direct hits, so the char-budget loop below can only ever drop the
        # graph extras — never a direct hit. Best-effort: any failure leaves
        # retrieval exactly as it was.
        hooks = getattr(self, "_hooks", None)
        if hooks is not None and getattr(hooks, "graph_aware", False) and query.text:
            try:
                expanded = await self._graph_expand(chunks, hooks)
                chunks.extend(expanded)
                if expanded:
                    breakdown[Layer.NOTES] = breakdown.get(Layer.NOTES, 0) + len(expanded)
            except Exception:  # noqa: BLE001 — graph enrichment never breaks retrieval
                logging.getLogger(__name__).debug("graph expansion skipped", exc_info=True)

        kept: List[MemoryChunk] = []
        used = 0
        for c in chunks:
            cost = len(c.content)
            if used + cost > query.max_chars and kept:
                break
            kept.append(c)
            used += cost

        return RetrievalResult(
            chunks=kept,
            layer_breakdown=breakdown,
            total_chars=used,
        )

    async def _graph_expand(
        self, chunks: List[MemoryChunk], hooks: "MemoryHooks"
    ) -> List[MemoryChunk]:
        """Return up to ``hooks.graph_top_k`` graph-connected notes not already
        present, ranked by Personalized PageRank seeded by the direct note hits.
        Returns ``[]`` when the index can't supply edges (older executor / no
        graph) or no seed lands on a graph node."""
        index = self.index()
        edges_fn = getattr(index, "graph_edges", None)
        if edges_fn is None:
            return []
        edges = await edges_fn()
        if not edges:
            return []
        node_set: Set[str] = set()
        for e in edges:
            node_set.add(e.get("source"))
            node_set.add(e.get("target"))
        existing = {c.key for c in chunks}
        seeds = {
            c.key: max(float(c.relevance_score), 0.05)
            for c in chunks
            if c.key in node_set
        }
        if not seeds:
            return []
        ranked = personalized_pagerank(
            edges, seeds, alpha=float(getattr(hooks, "graph_alpha", 0.5))
        )
        top_k = max(0, int(getattr(hooks, "graph_top_k", 5)))
        fresh = [
            (n, sc)
            for n, sc in sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)
            if n not in existing and sc > 0.0
        ][:top_k]
        if not fresh:
            return []
        notes = self.notes()
        out: List[MemoryChunk] = []
        for name, score in fresh:
            try:
                note = await notes.read(name)
            except Exception:  # noqa: BLE001
                note = None
            if note is None:
                continue
            body = note.body or ""
            out.append(
                MemoryChunk(
                    key=name,
                    content=body[:800],
                    source="graph",
                    # kept below direct hits (which use their own scores up to 1.0)
                    relevance_score=round(min(0.5, float(score)), 4),
                    metadata={"graph_expanded": True, "ppr": round(float(score), 5)},
                )
            )
        return out

    async def snapshot(self) -> MemorySnapshot:
        by_id: Dict[str, MemorySnapshot] = {}
        for provider_id, delegate in self._routing.by_id().items():
            by_id[provider_id] = await delegate.snapshot()
        payload, checksum = encode_snapshot(by_id)
        layers = sorted(self._descriptor.layers, key=lambda layer: layer.value)
        return MemorySnapshot(
            provider=self.NAME,
            version=self.VERSION,
            layers=layers,
            payload=payload,
            size_bytes=len(payload),
            checksum=checksum,
        )

    async def restore(self, snap: MemorySnapshot) -> None:
        if snap.provider != self.NAME:
            raise ValueError(f"snapshot from {snap.provider!r} cannot restore into {self.NAME!r}")
        if not isinstance(snap.payload, (bytes, bytearray)):
            raise TypeError(
                f"CompositeMemoryProvider snapshot payload must be bytes, "
                f"got {type(snap.payload)!r}"
            )
        sub = decode_snapshot(bytes(snap.payload), snap.checksum)
        delegates = self._routing.by_id()
        for provider_id, sub_snap in sub.items():
            delegate = delegates.get(provider_id)
            if delegate is None:
                logger.warning(
                    "composite restore: snapshot delegate %r has no live binding; skipping",
                    provider_id,
                )
                continue
            await delegate.restore(sub_snap)

    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef:
        if to == ref.scope:
            return ref
        source = self._routing.scope_provider(ref.scope) or self._require(Layer.NOTES)
        target = self._routing.scope_provider(to)
        if target is None or target is source:
            # No distinct target backend → fall back to the source's
            # own promote (typically a same-row scope rewrite).
            return await source.promote(ref, to)

        note = await source.notes().read(ref.filename)
        if note is None:
            raise KeyError(f"cannot promote: {ref.filename!r} not found in source provider")

        meta = await target.notes().write(
            NoteDraft(
                title=note.title,
                body=note.body,
                importance=note.importance,
                tags=list(note.tags),
                category=note.category,
                filename=note.ref.filename,
                frontmatter=dict(note.frontmatter),
                scope=to,
            )
        )
        await source.notes().delete(ref.filename)
        # The composite owns the scope axis; the target provider may be
        # scope-agnostic and tag rows with its own configured scope.
        # Force the returned ref to reflect the requested target scope.
        return meta.ref.with_scope(to)

    # ── descriptor builder ──────────────────────────────────────────

    def _build_descriptor(self) -> MemoryDescriptor:
        layers: Set[Layer] = set(self._routing.declared_layers())
        capabilities: Set[Capability] = set()
        backends: List[BackendInfo] = []
        embedding: Optional[EmbeddingDescriptor] = None

        for delegate in self._routing.distinct_providers():
            sub = delegate.descriptor
            capabilities.update(sub.capabilities)
            for info in sub.backends:
                backends.append(
                    BackendInfo(
                        layer=info.layer,
                        backend=info.backend,
                        location=info.location,
                        metadata={
                            **dict(info.metadata),
                            "delegate": sub.name,
                            "delegate_version": sub.version,
                        },
                    )
                )
            if embedding is None and sub.embedding is not None:
                embedding = sub.embedding

        # Composite always supports SNAPSHOT (it composes them) and
        # READ/WRITE/SEARCH (the required handles deliver them).
        capabilities.update({Capability.READ, Capability.WRITE, Capability.SEARCH})
        capabilities.add(Capability.SNAPSHOT)
        if Layer.VECTOR in layers:
            capabilities.add(Capability.REINDEX)
        if any(self._routing.scope_providers.values()):
            capabilities.add(Capability.PROMOTE)
        # Surface CURATED / GLOBAL on the descriptor so callers can
        # capability-gate without hand-rolling the same scope-routing
        # check the handle resolution does. The native check on the
        # delegate provider's descriptor is preserved — if the delegate
        # already advertises CURATED itself we never override it.
        if self._routing.scope_provider(Scope.USER) is not None:
            layers.add(Layer.CURATED)
        if self._routing.scope_provider(Scope.GLOBAL) is not None:
            layers.add(Layer.GLOBAL)

        delegate_summary = [
            {
                "id": pid,
                "name": delegate.descriptor.name,
                "version": delegate.descriptor.version,
                "layers": [layer.value for layer in delegate.descriptor.layers],
            }
            for pid, delegate in self._routing.by_id().items()
        ]

        return MemoryDescriptor(
            name=self.NAME,
            version=self.VERSION,
            layers=layers,
            capabilities=capabilities,
            backends=backends,
            scope=self._scope,
            config_schema=None,
            embedding=embedding,
            description=(
                "Composite provider routing each layer to an underlying "
                "MemoryProvider. Promote() copies notes across scope-bound "
                "providers; snapshot() bundles per-delegate snapshots."
            ),
            metadata={
                "session_id": self._session_id,
                "delegates": delegate_summary,
                "scope_routes": {
                    scope.value: type(provider).__name__
                    for scope, provider in self._routing.scope_providers.items()
                },
            },
        )


# ── helpers ──────────────────────────────────────────────────────────


def _attach_files(receipt: RecordReceipt, files: List[str]) -> RecordReceipt:
    receipt.files_updated = files
    return receipt


def _turn_to_text(turn: Turn) -> str:
    if isinstance(turn.content, str):
        return f"[{turn.role}] {turn.content}"
    return f"[{turn.role}] {turn.content!r}"


__all__ = ["CompositeMemoryProvider"]
