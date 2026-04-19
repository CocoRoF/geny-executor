"""Vector plane for SQLMemoryProvider.

Vectors live in `vector_rows` as packed little-endian float32 BLOBs
alongside provenance metadata (scope/category/preview). Cosine
similarity is computed in pure Python — same calculation the file
provider uses — so the SQL provider needs no native extension.

A future backend swap to Postgres + pgvector would replace this store
without touching the descriptor or the public surface; the pgvector
arm is tracked in the Phase 2c follow-up notes and is not part of
this PR.
"""

from __future__ import annotations

import math
import struct
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple

from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.provider import (
    EmbeddingDescriptor,
    Layer,
    NoteRef,
    ReindexPlan,
    Scope,
)
from geny_executor.memory.providers.sql.connection import _SQLConnection
from geny_executor.stages.s02_context.types import MemoryChunk


class _SQLVectorStore:
    """`VectorHandle`-conformant vector store on SQL (SQLite or Postgres)."""

    def __init__(
        self,
        conn: _SQLConnection,
        *,
        client: EmbeddingClient,
        notes_text_lookup: Any = None,
        backend_name: str = "sqlite",
    ) -> None:
        self._conn = conn
        self._client = client
        self._notes_text_lookup = notes_text_lookup
        self._backend_name = backend_name

    # ── VectorHandle contract ───────────────────────────────────────

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._client.descriptor

    async def index(self, ref: NoteRef, text: str) -> int:
        vec = (await self._client.embed([text]))[0]
        self._validate_dim(vec)
        existed = await self._exists(ref.filename)
        await self._upsert_row(ref, text, vec)
        return 0 if existed else 1

    async def index_batch(self, items: Sequence[Tuple[NoteRef, str]]) -> int:
        if not items:
            return 0
        texts = [t for _, t in items]
        vectors = await self._client.embed(texts)
        added = 0
        for (ref, text), vec in zip(items, vectors):
            self._validate_dim(vec)
            existed = await self._exists(ref.filename)
            await self._upsert_row(ref, text, vec)
            if not existed:
                added += 1
        return added

    async def search(
        self,
        text: str,
        *,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[MemoryChunk]:
        if top_k <= 0 or not text:
            return []
        rows = await self._conn.fetchall(
            """
            SELECT filename, scope, category, backend, preview, dimension, vector_blob
              FROM vector_rows
            """
        )
        if not rows:
            return []
        query_vec = (await self._client.embed([text]))[0]
        self._validate_dim(query_vec)
        scored: List[Tuple[float, Any, Sequence[float]]] = []
        for row in rows:
            dim = int(row["dimension"])
            vec = _unpack_vector(bytes(row["vector_blob"]), dim)
            score = _cosine(query_vec, vec)
            if score >= threshold:
                scored.append((score, row, vec))
        scored.sort(key=lambda triple: -triple[0])
        out: List[MemoryChunk] = []
        for score, row, vec in scored[:top_k]:
            out.append(
                MemoryChunk(
                    key=str(row["filename"]),
                    content=str(row["preview"] or ""),
                    source="vector",
                    relevance_score=score,
                    metadata={
                        "filename": str(row["filename"]),
                        "scope": str(row["scope"]),
                        "category": _optional_str(row["category"]),
                        "dimension": len(vec),
                    },
                )
            )
        return out

    async def reindex(self, *, plan: Optional[ReindexPlan] = None) -> ReindexPlan:
        rows = await self._conn.fetchall(
            "SELECT filename, scope, category, backend FROM vector_rows"
        )
        await self._conn.execute("DELETE FROM vector_rows")
        total = 0
        if rows and self._notes_text_lookup is not None:
            texts: List[str] = []
            refs: List[NoteRef] = []
            for row in rows:
                text = await self._notes_text_lookup(str(row["filename"]))
                if not text:
                    continue
                refs.append(_ref_from_row(row))
                texts.append(text)
            if texts:
                vectors = await self._client.embed(texts)
                for ref, text, vec in zip(refs, texts, vectors):
                    self._validate_dim(vec)
                    await self._upsert_row(ref, text, vec)
                    total += 1
        reason = plan.reason if plan is not None else "manual reindex"
        metadata = dict(plan.metadata) if plan is not None else {}
        metadata["descriptor"] = {
            "provider": self.descriptor.provider,
            "model": self.descriptor.model,
            "dimension": self.descriptor.dimension,
            "metric": self.descriptor.metric,
        }
        metadata["rebuilt_rows"] = total
        return ReindexPlan(
            layer=Layer.VECTOR,
            reason=reason,
            chunks_to_reindex=total,
            requires_explicit_approval=False,
            metadata=metadata,
        )

    async def remove(self, ref: NoteRef) -> bool:
        _, count = await self._conn.execute_returning(
            "DELETE FROM vector_rows WHERE filename = ?",
            (ref.filename,),
        )
        return count > 0

    # ── snapshot helpers ────────────────────────────────────────────

    async def all_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM vector_rows ORDER BY filename ASC")
        return [dict(r) for r in rows]

    # ── internals ───────────────────────────────────────────────────

    async def _exists(self, filename: str) -> bool:
        row = await self._conn.fetchone(
            "SELECT 1 FROM vector_rows WHERE filename = ?",
            (filename,),
        )
        return row is not None

    async def _upsert_row(self, ref: NoteRef, text: str, vec: Sequence[float]) -> None:
        scope = ref.scope.value if isinstance(ref.scope, Scope) else str(ref.scope)
        blob = _pack_vector(vec)
        ts = datetime.now(timezone.utc).isoformat()
        preview = (text or "")[:400]
        # SQLite UPSERT keeps the surface dialect-portable.
        await self._conn.execute(
            """
            INSERT INTO vector_rows (
                filename, scope, category, backend, preview,
                dimension, vector_blob, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                scope = excluded.scope,
                category = excluded.category,
                backend = excluded.backend,
                preview = excluded.preview,
                dimension = excluded.dimension,
                vector_blob = excluded.vector_blob
            """,
            (
                ref.filename,
                scope,
                ref.category,
                ref.backend,
                preview,
                len(vec),
                blob,
                ts,
            ),
        )

    def _validate_dim(self, vec: Sequence[float]) -> None:
        expected = self.descriptor.dimension
        if expected and len(vec) != expected:
            raise ValueError(f"vector dimension mismatch: expected {expected}, got {len(vec)}")


# ── module helpers ──────────────────────────────────────────────────


def _ref_from_row(row: Any) -> NoteRef:
    scope_raw = str(row["scope"]) if row["scope"] is not None else Scope.SESSION.value
    try:
        scope = Scope(scope_raw)
    except ValueError:
        scope = Scope.SESSION
    return NoteRef(
        filename=str(row["filename"]),
        scope=scope,
        category=_optional_str(row["category"]),
        backend=str(row["backend"] or "sqlite"),
    )


def _pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(blob: bytes, dim: int) -> List[float]:
    if dim <= 0 or len(blob) != dim * 4:
        return []
    return list(struct.unpack(f"<{dim}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _optional_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw)
    return s if s else None


__all__ = ["_SQLVectorStore"]
