"""FileMemoryProvider — disk-persistent MemoryProvider.

Hosts STM / LTM / Notes / Index on the filesystem in a layout that a
legacy Geny reader can consume without modification. Vector / Curated
/ Global return `None` in Phase 2a — those wire in subsequent PRs:

  * Phase 2b — VectorHandle via EmbeddingClient Protocol.
  * Phase 2d — CuratedHandle + GlobalHandle via CompositeMemoryProvider.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from geny_executor.memory.provider import (
    BackendInfo,
    Capability,
    CuratedHandle,
    EmbeddingDescriptor,
    ExecutionSummary,
    GlobalHandle,
    Importance,
    Insight,
    Layer,
    LTMHandle,
    MemoryDescriptor,
    MemoryProvider,
    MemorySnapshot,
    NoteDraft,
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
from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.providers.file.config import file_provider_config_schema
from geny_executor.memory.providers.file.index_store import _FileIndexStore
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.ltm_store import _MarkdownLTMStore
from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore
from geny_executor.memory.providers.file.snapshot import build_tarball, restore_tarball
from geny_executor.memory.providers.file.stm_store import _JSONLSTMStore
from geny_executor.memory.providers.file.timezone import resolve_timezone
from geny_executor.memory.providers.file.vector_store import _FileVectorStore
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)


class FileMemoryProvider(MemoryProvider):
    """`MemoryProvider` whose layers are files on disk.

    Construction does not touch disk; call `initialize()` to create
    the directory skeleton. `close()` is a no-op (files are always
    flushed at write time).
    """

    NAME = "file"
    VERSION = "1.0.0"

    def __init__(
        self,
        root: Path,
        *,
        scope: Scope = Scope.SESSION,
        session_id: str = "",
        timezone_name: Optional[str] = None,
        embedding: Optional[EmbeddingDescriptor] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._scope = scope
        self._session_id = session_id
        self._tz = resolve_timezone(timezone_name)
        self._embedding_client = embedding_client
        self._embedding = embedding or (
            embedding_client.descriptor if embedding_client is not None else None
        )
        self._layout = DirectoryLayout(self._root)
        self._stm = _JSONLSTMStore(self._layout.stm_jsonl, tz=self._tz)
        self._ltm = _MarkdownLTMStore(self._layout, tz=self._tz, scope=scope)
        self._notes = _FilesystemNotesStore(self._layout, tz=self._tz, scope=scope)
        self._index = _FileIndexStore(self._notes, layout=self._layout, tz=self._tz)
        self._vector = self._build_vector_store()
        # Auto-vector wiring — every successful note write/update
        # forwards the body to the vector store. The indexer is plugged
        # in *after* the notes store is constructed because the vector
        # store depends on the notes store's body lookup.
        if self._vector is not None:
            self._notes.attach_vector_indexer(self._vector.index)
        self._initialized = False
        self._descriptor = self._build_descriptor()

    def _build_vector_store(self) -> Optional[_FileVectorStore]:
        if self._embedding_client is None:
            return None
        return _FileVectorStore(
            self._layout,
            client=self._embedding_client,
            notes_text_lookup=self._lookup_note_text,
        )

    async def _lookup_note_text(self, filename: str) -> Optional[str]:
        note = await self._notes.read(filename)
        if note is None:
            return None
        return note.body

    # ── MemoryProvider: descriptor + lifecycle ─────────────────────

    @property
    def descriptor(self) -> MemoryDescriptor:
        return self._descriptor

    @property
    def root(self) -> Path:
        return self._root

    async def initialize(self) -> None:
        self._layout.ensure()
        self._initialized = True

    async def close(self) -> None:
        # All writes are flushed at op time; nothing to do.
        return None

    # ── layer handles ───────────────────────────────────────────────

    def stm(self) -> STMHandle:
        return self._stm  # type: ignore[return-value]

    def ltm(self) -> LTMHandle:
        return self._ltm  # type: ignore[return-value]

    def notes(self) -> NotesHandle:
        return self._notes

    def vector(self) -> Optional[VectorHandle]:
        return self._vector  # type: ignore[return-value]

    def curated(self) -> Optional[CuratedHandle]:
        return None

    def global_(self) -> Optional[GlobalHandle]:
        return None

    def index(self) -> _FileIndexStore:
        return self._index

    # ── cross-layer ─────────────────────────────────────────────────

    async def record_turn(self, turn: Turn) -> None:
        await self._stm.append(turn)

    async def record_execution(self, summary: ExecutionSummary) -> RecordReceipt:
        files: List[str] = []
        receipt = RecordReceipt()

        if summary.final_text:
            qa_body = f"## Q\n{summary.user_input}\n\n## A\n{summary.final_text}".strip()
            ref_dated = await self._ltm.write_dated(qa_body)
            files.append(ref_dated.filename)

            note_meta = await self._notes.write(
                NoteDraft(
                    title=(summary.user_input or "execution")[:80],
                    body=summary.final_text,
                    importance=Importance.MEDIUM,
                    tags=list(summary.tags),
                    category="insights",
                    scope=self._scope,
                )
            )
            files.append(note_meta.ref.filename)
            receipt.notes_written = 1
            # Auto-vector hook (set up in __init__) embeds the body
            # inside notes.write — no second index() call needed.
            if self._vector is not None:
                receipt.vector_chunks = 1

        # Refresh the index cache so subsequent retrieves see the new note
        await self._index.rebuild()

        receipt.files_updated = files
        return receipt

    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]:
        # File provider has no LLM; reflection wires in via MemoryHooks
        # / the orchestrating stage. Default is "no insights".
        return ()

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        chunks: List[MemoryChunk] = []
        breakdown: Dict[Layer, int] = {}

        if Layer.STM in query.layers:
            recent = await self._stm.recent(n=query.max_per_layer)
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

        if Layer.LTM in query.layers:
            main_text = await self._ltm.read_main()
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
                ltm_chunks.extend(await self._ltm.search(query.text, limit=query.max_per_layer))
            chunks.extend(ltm_chunks)
            breakdown[Layer.LTM] = len(ltm_chunks)

        if Layer.NOTES in query.layers and query.text:
            note_chunks = await self._notes.search(
                query.text,
                limit=query.max_per_layer,
                importance_floor=query.importance_floor,
            )
            chunks.extend(note_chunks)
            breakdown[Layer.NOTES] = len(note_chunks)

        if Layer.VECTOR in query.layers and self._vector is not None and query.text:
            vec_chunks = await self._vector.search(
                query.text,
                top_k=query.max_per_layer,
            )
            chunks.extend(vec_chunks)
            breakdown[Layer.VECTOR] = len(vec_chunks)

        # Char budget trim — preserves order, always keeps at least one
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

    async def snapshot(self) -> MemorySnapshot:
        # Make sure the derived index is materialised before archiving.
        await self._index.snapshot()
        layers = [Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX]
        if self._vector is not None:
            layers.append(Layer.VECTOR)
        payload, checksum = build_tarball(self._root)
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
                f"FileMemoryProvider snapshot payload must be bytes, got {type(snap.payload)!r}"
            )
        restore_tarball(self._root, bytes(snap.payload), snap.checksum)
        # Reattach stores against the new on-disk state
        self._layout = DirectoryLayout(self._root)
        self._stm = _JSONLSTMStore(self._layout.stm_jsonl, tz=self._tz)
        self._ltm = _MarkdownLTMStore(self._layout, tz=self._tz, scope=self._scope)
        self._notes = _FilesystemNotesStore(self._layout, tz=self._tz, scope=self._scope)
        self._index = _FileIndexStore(self._notes, layout=self._layout, tz=self._tz)
        self._vector = self._build_vector_store()
        if self._vector is not None:
            self._notes.attach_vector_indexer(self._vector.index)
        await self._index.rebuild()

    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef:
        # File provider has only SESSION scope in Phase 2a. Promotion
        # to USER/TENANT/GLOBAL becomes meaningful once the Composite
        # provider (PR #5) aggregates per-scope stores.
        if to == ref.scope:
            return ref
        note = await self._notes.read(ref.filename)
        if note is None:
            raise KeyError(f"cannot promote: {ref.filename!r} not found")
        # Rewrite the ref with the new scope; no disk motion yet.
        note.ref = ref.with_scope(to)
        return note.ref

    # ── descriptor builder ──────────────────────────────────────────

    def _build_descriptor(self) -> MemoryDescriptor:
        layers = {Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX}
        capabilities = {
            Capability.READ,
            Capability.WRITE,
            Capability.SEARCH,
            Capability.LINK,
            Capability.SNAPSHOT,
        }
        backends = [
            BackendInfo(
                layer=Layer.STM,
                backend="filesystem",
                location=str(self._layout.stm_jsonl),
            ),
            BackendInfo(
                layer=Layer.LTM,
                backend="filesystem",
                location=str(self._layout.memory),
            ),
            BackendInfo(
                layer=Layer.NOTES,
                backend="filesystem",
                location=str(self._layout.memory),
            ),
            BackendInfo(
                layer=Layer.INDEX,
                backend="filesystem",
                location=str(self._layout.index_json),
            ),
        ]
        if self._vector is not None:
            layers.add(Layer.VECTOR)
            capabilities.add(Capability.REINDEX)
            backends.append(
                BackendInfo(
                    layer=Layer.VECTOR,
                    backend="filesystem",
                    location=str(self._layout.vector_metadata),
                    metadata={
                        "embedding_provider": self._vector.descriptor.provider,
                        "embedding_model": self._vector.descriptor.model,
                        "dimension": self._vector.descriptor.dimension,
                    },
                )
            )
        vector_note = (
            "Vector layer wired via EmbeddingClient. "
            if self._vector is not None
            else "Vector / Curated / Global not wired in this release."
        )
        return MemoryDescriptor(
            name=self.NAME,
            version=self.VERSION,
            layers=layers,
            capabilities=capabilities,
            backends=backends,
            scope=self._scope,
            config_schema=file_provider_config_schema(),
            embedding=self._embedding,
            description=(
                "Filesystem-backed memory provider. Geny-compatible layout: "
                "STM JSONL, LTM markdown (dated + topic), notes with YAML frontmatter. "
                + vector_note
            ),
            metadata={
                "root": str(self._root),
                "session_id": self._session_id,
                "timezone": str(self._tz),
            },
        )


# ── helpers ──────────────────────────────────────────────────────────


def _turn_to_text(turn: Turn) -> str:
    if isinstance(turn.content, str):
        return f"[{turn.role}] {turn.content}"
    return f"[{turn.role}] {turn.content!r}"
