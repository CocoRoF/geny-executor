"""SQLMemoryProvider — SQL-backed `MemoryProvider`.

Default dialect: SQLite via stdlib `sqlite3`. The dialect choice
flows through the `_SQLiteConnection` wrapper; replacing it with a
Postgres + pgvector implementation is a per-PR follow-up that does
not change the public surface.

Layer mapping mirrors `FileMemoryProvider` exactly so the cross-
provider contract suite passes against both. The Vector layer is
optional and lights up only when an `EmbeddingClient` is supplied at
construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from geny_executor.memory.embedding.client import EmbeddingClient
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
from geny_executor.memory.providers.file.timezone import resolve_timezone
from geny_executor.memory.providers.sql.config import sql_provider_config_schema
from geny_executor.memory.providers.sql.connection import _SQLiteConnection
from geny_executor.memory.providers.sql.index_store import _SQLIndexStore
from geny_executor.memory.providers.sql.ltm_store import _SQLLTMStore
from geny_executor.memory.providers.sql.notes_store import _SQLNotesStore
from geny_executor.memory.providers.sql.snapshot import build_snapshot, restore_snapshot
from geny_executor.memory.providers.sql.stm_store import _SQLSTMStore
from geny_executor.memory.providers.sql.vector_store import _SQLVectorStore
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)


DSN = Union[str, Path]


class SQLMemoryProvider(MemoryProvider):
    """`MemoryProvider` whose layers are SQL tables.

    Construction is cheap; `initialize()` opens the connection and
    creates the schema. `close()` flushes and closes the connection.
    """

    NAME = "sql"
    VERSION = "1.0.0"

    def __init__(
        self,
        dsn: DSN,
        *,
        scope: Scope = Scope.SESSION,
        session_id: str = "",
        timezone_name: Optional[str] = None,
        embedding: Optional[EmbeddingDescriptor] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self._dsn = str(dsn)
        self._scope = scope
        self._session_id = session_id
        self._tz = resolve_timezone(timezone_name)
        self._embedding_client = embedding_client
        self._embedding = embedding or (
            embedding_client.descriptor if embedding_client is not None else None
        )
        self._conn = _SQLiteConnection(self._dsn)
        self._stm = _SQLSTMStore(self._conn, tz=self._tz)
        self._ltm = _SQLLTMStore(self._conn, tz=self._tz, scope=scope)
        self._notes = _SQLNotesStore(self._conn, tz=self._tz, scope=scope)
        self._index = _SQLIndexStore(self._notes, conn=self._conn, tz=self._tz)
        self._vector = self._build_vector_store()
        self._initialized = False
        self._descriptor = self._build_descriptor()

    def _build_vector_store(self) -> Optional[_SQLVectorStore]:
        if self._embedding_client is None:
            return None
        return _SQLVectorStore(
            self._conn,
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
    def dsn(self) -> str:
        return self._dsn

    async def initialize(self) -> None:
        await self._conn.open()
        self._initialized = True

    async def close(self) -> None:
        await self._conn.close()

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

    def index(self) -> _SQLIndexStore:
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

            if self._vector is not None:
                await self._vector.index(note_meta.ref, summary.final_text)

        receipt.files_updated = files
        return receipt

    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]:
        # SQL provider has no LLM; reflection wires in via MemoryHooks
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
        layers = [Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX]
        if self._vector is not None:
            layers.append(Layer.VECTOR)
        payload, checksum = await build_snapshot(self._conn)
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
                f"SQLMemoryProvider snapshot payload must be bytes, got {type(snap.payload)!r}"
            )
        await restore_snapshot(self._conn, bytes(snap.payload), snap.checksum)

    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef:
        if to == ref.scope:
            return ref
        # Same semantics as the file provider — no cross-scope motion
        # until the Composite (PR #5) lands.
        note = await self._notes.read(ref.filename)
        if note is None:
            raise KeyError(f"cannot promote: {ref.filename!r} not found")
        new_ref = ref.with_scope(to)
        # Persist the new scope on the row so subsequent reads agree.
        await self._conn.execute(
            "UPDATE notes SET scope = ? WHERE filename = ?",
            (to.value, ref.filename),
        )
        return new_ref

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
                backend="sqlite",
                location=self._dsn,
                metadata={"table": "stm_turns"},
            ),
            BackendInfo(
                layer=Layer.LTM,
                backend="sqlite",
                location=self._dsn,
                metadata={"table": "ltm_documents"},
            ),
            BackendInfo(
                layer=Layer.NOTES,
                backend="sqlite",
                location=self._dsn,
                metadata={"tables": ["notes", "note_tags", "note_links"]},
            ),
            BackendInfo(
                layer=Layer.INDEX,
                backend="sqlite",
                location=self._dsn,
                metadata={"derived_from": ["notes", "note_tags", "note_links"]},
            ),
        ]
        if self._vector is not None:
            layers.add(Layer.VECTOR)
            capabilities.add(Capability.REINDEX)
            backends.append(
                BackendInfo(
                    layer=Layer.VECTOR,
                    backend="sqlite",
                    location=self._dsn,
                    metadata={
                        "table": "vector_rows",
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
            config_schema=sql_provider_config_schema(),
            embedding=self._embedding,
            description=(
                "SQLite-backed memory provider. Schema mirrors the file "
                "provider: STM, LTM, Notes (with tags + links), Vector, "
                "and an SQL-derived Index. " + vector_note
            ),
            metadata={
                "dsn": self._dsn,
                "session_id": self._session_id,
                "timezone": str(self._tz),
            },
        )


# ── helpers ──────────────────────────────────────────────────────────


def _turn_to_text(turn: Turn) -> str:
    if isinstance(turn.content, str):
        return f"[{turn.role}] {turn.content}"
    return f"[{turn.role}] {turn.content!r}"
