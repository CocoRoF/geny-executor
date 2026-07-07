"""Qdrant-backed ``VectorHandle`` — ANN vector search for knowledge vaults.

Where the built-in file/sql stores keep one vector per note (pure-Python
cosine, fine for a few hundred session notes), this store targets the
knowledge-repository shape:

* **N chunks per document** — ``index_document(ref, chunks)`` upserts one
  point per chunk with a payload (page/heading/source metadata) and
  replaces the document's previous points atomically-enough (delete by
  filter, then upsert).
* **Real ANN** — qdrant handles scale; searches return ``MemoryChunk``
  rows shaped exactly like the built-in stores so the Stage-2 retriever
  and host tools consume them unchanged.
* **Same auto-index seam** — ``index(ref, text)`` matches the protocol
  used by ``NotesStore.attach_vector_indexer``, so attaching this store
  to a provider gives every markdown note write a vector for free.

The client is created lazily so importing this module never requires
``qdrant-client`` (optional extra ``qdrant``). All public methods are
best-effort against transport errors *except* collection bootstrap
mismatches, which raise — silently writing 3072-dim vectors into a
1536-dim collection would corrupt search.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.provider import (
    EmbeddingDescriptor,
    Layer,
    NoteRef,
    ReindexPlan,
)
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)

#: Stable namespace for deterministic point ids — the same (collection,
#: filename, chunk) always maps to the same point, making upserts
#: idempotent across restarts.
_POINT_NAMESPACE = uuid.UUID("6f9d35b2-8a54-4f0e-9c39-2f6a1f6f0f3e")


@dataclass(slots=True)
class DocumentChunk:
    """One embeddable chunk of a source document.

    ``metadata`` rides into the qdrant payload verbatim (page numbers,
    heading path, source url/dsn, content hashes, …) and comes back on
    every search hit's ``MemoryChunk.metadata``.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _point_id(collection: str, filename: str, chunk_index: int) -> str:
    return str(
        uuid.uuid5(_POINT_NAMESPACE, f"{collection}:{filename}#{chunk_index}")
    )


class QdrantVectorStore:
    """``VectorHandle`` implementation on a qdrant collection.

    Args:
        url: qdrant HTTP endpoint (e.g. ``http://qdrant:6333``).
        collection: collection name — one per vault/tenant.
        client: embedding client (provides vectors + descriptor).
        api_key: optional qdrant API key.
        preview_chars: how much chunk text to keep in the payload for
            hit previews (full text is returned as ``MemoryChunk.content``).
    """

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        client: EmbeddingClient,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        preview_chars: int = 1200,
    ) -> None:
        if not url:
            raise ValueError("QdrantVectorStore requires a url")
        if not collection:
            raise ValueError("QdrantVectorStore requires a collection name")
        if client is None:
            raise ValueError("QdrantVectorStore requires an embedding client")
        self._url = url
        self._collection = collection
        self._client = client
        self._api_key = api_key
        self._timeout = timeout
        self._preview_chars = max(200, int(preview_chars))
        self._qdrant: Optional[Any] = None
        self._collection_ready = False

    # ── VectorHandle: descriptor ─────────────────────────────────────

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._client.descriptor

    # ── qdrant plumbing ──────────────────────────────────────────────

    def _get_qdrant(self) -> Any:
        if self._qdrant is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError as exc:  # pragma: no cover - env-specific
                raise RuntimeError(
                    "qdrant-client is not installed — "
                    "`pip install geny-executor[qdrant]`"
                ) from exc
            self._qdrant = AsyncQdrantClient(
                url=self._url, api_key=self._api_key, timeout=self._timeout,
            )
        return self._qdrant

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        from qdrant_client import models

        qdrant = self._get_qdrant()
        dim = int(self.descriptor.dimension)
        if await qdrant.collection_exists(self._collection):
            info = await qdrant.get_collection(self._collection)
            existing = info.config.params.vectors
            existing_dim = getattr(existing, "size", None)
            if existing_dim is not None and int(existing_dim) != dim:
                raise RuntimeError(
                    f"qdrant collection '{self._collection}' has dimension "
                    f"{existing_dim}, embedding model produces {dim} — "
                    "reindex into a fresh collection instead of mixing"
                )
        else:
            await qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE,
                ),
            )
        self._collection_ready = True

    @staticmethod
    def _ref_filter(filename: str) -> Any:
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="filename", match=models.MatchValue(value=filename),
                )
            ]
        )

    # ── VectorHandle: indexing ───────────────────────────────────────

    async def index(self, ref: NoteRef, text: str) -> int:
        """Single-chunk index — the ``attach_vector_indexer`` seam, so a
        plain note write gets one point (chunk 0) replacing prior ones."""
        return await self.index_document(ref, [DocumentChunk(text=text)])

    async def index_batch(self, items: Sequence[Tuple[NoteRef, str]]) -> int:
        total = 0
        for ref, text in items:
            total += await self.index(ref, text)
        return total

    async def index_document(
        self, ref: NoteRef, chunks: Sequence[DocumentChunk],
    ) -> int:
        """Replace *ref*'s points with one point per chunk. Returns the
        number of chunks indexed (0 on failure — never raises transport)."""
        rows = [c for c in chunks if (c.text or "").strip()]
        if not rows:
            return 0
        try:
            await self._ensure_collection()
            vectors = await self._client.embed([c.text for c in rows])
        except Exception:  # noqa: BLE001 — embedding/transport is best-effort
            logger.warning(
                "qdrant: embed failed for %s", ref.filename, exc_info=True,
            )
            return 0
        if len(vectors) != len(rows):
            logger.warning("qdrant: embedding count mismatch for %s", ref.filename)
            return 0

        from qdrant_client import models

        points = []
        for i, (chunk, vector) in enumerate(zip(rows, vectors)):
            payload: Dict[str, Any] = {
                "filename": ref.filename,
                "scope": getattr(ref.scope, "value", str(ref.scope)),
                "category": ref.category or "",
                "chunk_index": i,
                "chunk_count": len(rows),
                # Full chunk text — lossless document reassembly via
                # ``fetch_document``. ``preview`` stays a bounded copy for
                # search-hit display (and back-compat with pre-2.48 points
                # that carry only ``preview``).
                "text": chunk.text,
                "preview": chunk.text[: self._preview_chars],
                "content_sha1": hashlib.sha1(
                    chunk.text.encode("utf-8")
                ).hexdigest(),
            }
            for key, value in (chunk.metadata or {}).items():
                if key not in payload:
                    payload[key] = value
            points.append(
                models.PointStruct(
                    id=_point_id(self._collection, ref.filename, i),
                    vector=list(vector),
                    payload=payload,
                )
            )
        try:
            qdrant = self._get_qdrant()
            # Drop stale points beyond the new chunk count (shrinking doc),
            # then upsert — ids are deterministic so overlaps overwrite.
            await qdrant.delete(
                collection_name=self._collection,
                points_selector=self._ref_filter(ref.filename),
            )
            await qdrant.upsert(collection_name=self._collection, points=points)
        except Exception:  # noqa: BLE001
            logger.warning(
                "qdrant: upsert failed for %s", ref.filename, exc_info=True,
            )
            return 0
        return len(points)

    # ── VectorHandle: search / remove / reindex ──────────────────────

    async def search(
        self, text: str, *, top_k: int = 5, threshold: float = 0.0,
    ) -> List[MemoryChunk]:
        query = (text or "").strip()
        if not query:
            return []
        try:
            await self._ensure_collection()
            vectors = await self._client.embed([query])
            qdrant = self._get_qdrant()
            hits = await qdrant.query_points(
                collection_name=self._collection,
                query=list(vectors[0]),
                limit=max(1, int(top_k)),
                score_threshold=float(threshold) if threshold > 0 else None,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning("qdrant: search failed", exc_info=True)
            return []

        chunks: List[MemoryChunk] = []
        for point in getattr(hits, "points", []) or []:
            payload = dict(point.payload or {})
            preview = str(payload.get("preview", ""))
            filename = str(payload.get("filename", ""))
            chunk_index = payload.get("chunk_index", 0)
            chunks.append(
                MemoryChunk(
                    key=f"vector:{filename}#{chunk_index}",
                    content=preview,
                    source="vector",
                    relevance_score=float(point.score or 0.0),
                    metadata=payload,
                )
            )
        return chunks

    async def fetch_document(
        self, ref: NoteRef, *, max_chunks: int = 5000,
    ) -> List[MemoryChunk]:
        """Return ALL of *ref*'s chunks, ordered by ``chunk_index``.

        Reads by filter (no embedding — like ``remove``, it skips the
        dimension guard so a document embedded under another model is
        still fetchable). Each returned :class:`MemoryChunk` carries the
        FULL chunk text in ``content`` (falling back to the bounded
        ``preview`` for points indexed before 2.48). This is the
        reassembly primitive a host turns into a document-read tool: join
        the ordered ``content`` values to recover the document text.
        Empty list when the collection or document is absent.
        """
        try:
            qdrant = self._get_qdrant()
            if not await qdrant.collection_exists(self._collection):
                return []
        except Exception:  # noqa: BLE001
            logger.warning(
                "qdrant: fetch_document existence check failed for %s",
                ref.filename, exc_info=True,
            )
            return []

        points: List[Any] = []
        offset: Any = None
        try:
            while len(points) < max_chunks:
                batch, offset = await qdrant.scroll(
                    collection_name=self._collection,
                    scroll_filter=self._ref_filter(ref.filename),
                    limit=min(256, max_chunks - len(points)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points.extend(batch or [])
                if not offset or not batch:
                    break
        except Exception:  # noqa: BLE001
            logger.warning(
                "qdrant: fetch_document scroll failed for %s",
                ref.filename, exc_info=True,
            )
            return []

        rows = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            rows.append((int(payload.get("chunk_index", 0)), payload))
        rows.sort(key=lambda r: r[0])

        out: List[MemoryChunk] = []
        for idx, payload in rows:
            content = payload.get("text")
            if content is None:  # pre-2.48 point — only preview stored
                content = payload.get("preview", "")
            filename = str(payload.get("filename", ref.filename))
            out.append(
                MemoryChunk(
                    key=f"vector:{filename}#{idx}",
                    content=str(content),
                    source="vector",
                    relevance_score=1.0,
                    metadata=payload,
                )
            )
        return out

    async def remove(self, ref: NoteRef) -> bool:
        """Delete *ref*'s points by filter. Removal never embeds, so it
        deliberately skips the dimension guard (`_ensure_collection`) —
        cleaning vectors out of a collection built for a DIFFERENT model
        (e.g. after an embedding-model switch) must work; and a missing
        collection is simply "nothing to remove", not a reason to create
        one."""
        try:
            qdrant = self._get_qdrant()
            if not await qdrant.collection_exists(self._collection):
                return False
            await qdrant.delete(
                collection_name=self._collection,
                points_selector=self._ref_filter(ref.filename),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning("qdrant: remove failed for %s", ref.filename, exc_info=True)
            return False

    async def reindex(self, *, plan: Optional[ReindexPlan] = None) -> ReindexPlan:
        """Reindexing a knowledge collection is a host-driven pipeline
        (re-extract + re-chunk + re-embed); this handle only reports."""
        try:
            await self._ensure_collection()
            qdrant = self._get_qdrant()
            count = await qdrant.count(collection_name=self._collection)
            points = int(getattr(count, "count", 0))
        except Exception:  # noqa: BLE001
            points = 0
        return ReindexPlan(
            layer=Layer.VECTOR,
            reason="qdrant reindex is host-driven (re-extract + re-embed)",
            chunks_to_reindex=points,
            requires_explicit_approval=True,
            metadata={"collection": self._collection, "backend": "qdrant"},
        )

    async def close(self) -> None:
        if self._qdrant is not None:
            try:
                await self._qdrant.close()
            except Exception:  # noqa: BLE001
                pass
            self._qdrant = None
