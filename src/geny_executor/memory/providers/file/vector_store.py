"""Vector plane for FileMemoryProvider.

Stores dense vectors on disk in a compact binary file (`index.bin`)
plus a JSON metadata sidecar (`metadata.json`). Pure-Python — no numpy
or FAISS dependency — because:

- The file provider is targeted at single-session sessions where the
  note count is small (usually <500). O(N) cosine over a list is fast
  enough (<5 ms at 1k × 1024-dim vectors).
- Avoiding numpy keeps the dep surface minimal, matching Phase 1's
  zero-SDK-dep design for the core provider.
- When scale is needed, sub-PR 2c's `SQLMemoryProvider` plugs into
  sqlite-vss / pgvector, which are the right tools for that regime.

Format notes:
- `index.bin` — packed `float32` values, row-major: N rows × D dims.
- `metadata.json` — `{"dimension": D, "model": "<provider>/<model>",
  "rows": [{"filename", "ref": {...}, "preview"}, ...]}`.
- Row order in `index.bin` matches `rows` order in metadata.

Removing a row rewrites both files. For a file-backed session this
is acceptable; sub-PR 2c picks up a row-delete path for the SQL
backend.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.provider import (
    EmbeddingDescriptor,
    Layer,
    NoteRef,
    ReindexPlan,
    Scope,
)
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.stages.s02_context.types import MemoryChunk


class _FileVectorStore:
    """`VectorHandle`-conformant vector store on the filesystem.

    One instance per session. `index(ref, text)` embeds the text via
    the injected `EmbeddingClient` and appends the vector to the
    on-disk store. `search(text, top_k)` re-embeds the query and
    returns the top-k cosine-nearest chunks.

    Dimension is taken from the client's descriptor at construction.
    A client swap that produces a different dimension is detected by
    `compatibility_check()` in the provider layer; this store itself
    refuses mixed-dimension inserts.
    """

    def __init__(
        self,
        layout: DirectoryLayout,
        *,
        client: EmbeddingClient,
        notes_text_lookup: Any = None,
    ) -> None:
        self._layout = layout
        self._client = client
        self._notes_text_lookup = notes_text_lookup
        self._lock = asyncio.Lock()
        self._loaded = False
        self._vectors: List[List[float]] = []
        self._rows: List[Dict[str, Any]] = []

    # ── VectorHandle contract ───────────────────────────────────────

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._client.descriptor

    async def index(self, ref: NoteRef, text: str) -> int:
        async with self._lock:
            await self._ensure_loaded()
            vec = (await self._client.embed([text]))[0]
            self._validate_dim(vec)
            # Replace any existing row for the same filename
            removed = self._remove_by_filename(ref.filename)
            self._vectors.append(vec)
            self._rows.append(_row_for(ref, text))
            self._flush()
            return 1 if removed == 0 else 0

    async def index_batch(self, items: Sequence[Tuple[NoteRef, str]]) -> int:
        if not items:
            return 0
        async with self._lock:
            await self._ensure_loaded()
            texts = [text for _, text in items]
            vectors = await self._client.embed(texts)
            added = 0
            for (ref, text), vec in zip(items, vectors):
                self._validate_dim(vec)
                removed = self._remove_by_filename(ref.filename)
                self._vectors.append(vec)
                self._rows.append(_row_for(ref, text))
                if removed == 0:
                    added += 1
            self._flush()
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
        async with self._lock:
            await self._ensure_loaded()
            if not self._vectors:
                return []
            query_vec = (await self._client.embed([text]))[0]
            self._validate_dim(query_vec)
            scored: List[Tuple[float, int]] = []
            for i, vec in enumerate(self._vectors):
                score = _cosine(query_vec, vec)
                if score >= threshold:
                    scored.append((score, i))
            scored.sort(key=lambda pair: -pair[0])
            out: List[MemoryChunk] = []
            for score, idx in scored[:top_k]:
                row = self._rows[idx]
                out.append(
                    MemoryChunk(
                        key=row["filename"],
                        content=row.get("preview", ""),
                        source="vector",
                        relevance_score=score,
                        metadata={
                            "filename": row["filename"],
                            "scope": row.get("scope"),
                            "category": row.get("category"),
                            "dimension": len(vec),
                        },
                    )
                )
            return out

    async def reindex(self, *, plan: Optional[ReindexPlan] = None) -> ReindexPlan:
        """Rebuild every vector from source notes.

        If a plan is provided, we honour its `reason` in the returned
        receipt. Otherwise we infer a reason from the current state.
        """
        async with self._lock:
            await self._ensure_loaded()
            source = list(self._rows)  # snapshot rows before we wipe
            self._vectors = []
            self._rows = []
            total = 0
            if source and self._notes_text_lookup is not None:
                texts: List[str] = []
                refs: List[NoteRef] = []
                for row in source:
                    text = await self._notes_text_lookup(row["filename"])
                    if not text:
                        continue
                    ref = _ref_from_row(row)
                    refs.append(ref)
                    texts.append(text)
                if texts:
                    vectors = await self._client.embed(texts)
                    for ref, text, vec in zip(refs, texts, vectors):
                        self._validate_dim(vec)
                        self._vectors.append(vec)
                        self._rows.append(_row_for(ref, text))
                        total += 1
            self._flush()
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
        async with self._lock:
            await self._ensure_loaded()
            removed = self._remove_by_filename(ref.filename)
            if removed:
                self._flush()
            return removed > 0

    # ── internal ────────────────────────────────────────────────────

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        meta_path = self._layout.vector_metadata
        bin_path = self._layout.vector_index.with_suffix(".bin")
        self._vectors = []
        self._rows = []
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                rows = payload.get("rows", []) or []
                dim = int(payload.get("dimension", 0) or 0)
                if (
                    dim
                    and dim == self.descriptor.dimension
                    and isinstance(rows, list)
                    and bin_path.exists()
                ):
                    raw = bin_path.read_bytes()
                    count = len(rows)
                    expected = count * dim * 4
                    if len(raw) == expected:
                        flat = list(struct.unpack(f"<{count * dim}f", raw))
                        for i in range(count):
                            self._vectors.append(flat[i * dim : (i + 1) * dim])
                            self._rows.append(dict(rows[i]))
        self._loaded = True

    def _flush(self) -> None:
        dim = self.descriptor.dimension
        bin_path = self._layout.vector_index.with_suffix(".bin")
        meta_path = self._layout.vector_metadata
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        flat: List[float] = []
        for vec in self._vectors:
            flat.extend(vec)
        if flat:
            bin_path.write_bytes(struct.pack(f"<{len(flat)}f", *flat))
        elif bin_path.exists():
            bin_path.unlink()
        payload = {
            "dimension": dim,
            "model": f"{self.descriptor.provider}/{self.descriptor.model}",
            "metric": self.descriptor.metric,
            "rows": list(self._rows),
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_by_filename(self, filename: str) -> int:
        kept_vectors: List[List[float]] = []
        kept_rows: List[Dict[str, Any]] = []
        dropped = 0
        for vec, row in zip(self._vectors, self._rows):
            if row.get("filename") == filename:
                dropped += 1
                continue
            kept_vectors.append(vec)
            kept_rows.append(row)
        if dropped:
            self._vectors = kept_vectors
            self._rows = kept_rows
        return dropped

    def _validate_dim(self, vec: Sequence[float]) -> None:
        expected = self.descriptor.dimension
        if expected and len(vec) != expected:
            raise ValueError(f"vector dimension mismatch: expected {expected}, got {len(vec)}")


# ── helpers ──────────────────────────────────────────────────────────


def _row_for(ref: NoteRef, text: str) -> Dict[str, Any]:
    return {
        "filename": ref.filename,
        "scope": ref.scope.value if isinstance(ref.scope, Scope) else str(ref.scope),
        "category": ref.category,
        "backend": ref.backend,
        "preview": (text or "")[:400],
    }


def _ref_from_row(row: Dict[str, Any]) -> NoteRef:
    scope_raw = row.get("scope") or Scope.SESSION.value
    try:
        scope = Scope(scope_raw)
    except ValueError:
        scope = Scope.SESSION
    return NoteRef(
        filename=row["filename"],
        scope=scope,
        category=row.get("category"),
        backend=row.get("backend", "filesystem"),
    )


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


__all__ = ["_FileVectorStore"]
