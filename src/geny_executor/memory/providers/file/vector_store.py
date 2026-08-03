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

import hashlib

import json
import logging
import math
import os
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.embedding.client import EmbeddingClient, EmbeddingError
from geny_executor.memory.provider import (
    EmbeddingDescriptor,
    Layer,
    NoteRef,
    ReindexPlan,
    Scope,
)
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)

# Consecutive 'auth'-classified embedding failures before the store
# trips its breaker. Three, not one, because a proxy or gateway can
# emit a stray 401 during key rotation — but three in a row means the
# key is genuinely dead and every further call is paid log spam (the
# live prod incident retried + tracebacked on *every* note write).
AUTH_TRIP_THRESHOLD = 3


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

    Auth breaker (2.2.0, audit §2.6): after `AUTH_TRIP_THRESHOLD`
    consecutive 'auth'-classified embedding failures the store sets
    ``vector_disabled`` and degrades silently for the rest of the
    session — `index`/`index_batch` return 0, `search` returns [],
    `reindex` returns an empty receipt, all without touching the
    network. One warning is logged at trip time; recovery requires
    fixing the credentials and restarting (mirrors the MCP NEEDS_AUTH
    design — a dead key cannot heal mid-session, so retrying it on
    every note write only spams logs and burns latency).
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
        self._lock = LoopAgnosticLock()
        self._loaded = False
        self._vectors: List[List[float]] = []
        self._rows: List[Dict[str, Any]] = []
        # ── auth breaker state ──────────────────────────────────────
        self._consecutive_auth_failures = 0
        self._disabled = False
        self._disabled_reason: Optional[str] = None
        self._transient_failure_seen = False

    # ── VectorHandle contract ───────────────────────────────────────

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._client.descriptor

    @property
    def vector_disabled(self) -> bool:
        """True once the auth breaker has tripped for this session."""
        return self._disabled

    @property
    def disabled_reason(self) -> Optional[str]:
        """The failure message that tripped the breaker (None while live)."""
        return self._disabled_reason

    async def index(self, ref: NoteRef, text: str) -> int:
        if self._disabled:
            return 0
        async with self._lock:
            await self._ensure_loaded()
            vec = (await self._embed_guarded([text]))[0]
            self._validate_dim(vec)
            # Replace any existing row for the same filename
            removed = self._remove_by_filename(ref.filename)
            self._vectors.append(vec)
            self._rows.append(_row_for(ref, text))
            self._flush()
            return 1 if removed == 0 else 0

    async def index_batch(self, items: Sequence[Tuple[NoteRef, str]]) -> int:
        if not items or self._disabled:
            return 0
        async with self._lock:
            await self._ensure_loaded()
            # Idempotent backfill (2.64.3): index_batch is the session-resume
            # warm-up path — it used to re-embed EVERY note on every resume
            # (6k-note vault ≈ minutes of embedding HTTP + real money, every
            # 30-min idle-evict cycle). Rows persist a content sha, so only
            # new/changed notes reach the embedder.
            existing_sha = {
                row.get("filename"): row.get("sha")
                for row in self._rows
                if row.get("sha")
            }
            todo = [
                (ref, text)
                for ref, text in items
                if existing_sha.get(ref.filename) != _text_sha(text)
            ]
            if not todo:
                return 0
            texts = [text for _, text in todo]
            vectors = await self._embed_guarded(texts)
            added = 0
            for (ref, text), vec in zip(todo, vectors):
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
        if top_k <= 0 or not text or self._disabled:
            return []
        async with self._lock:
            await self._ensure_loaded()
            if not self._vectors:
                return []
            query_vec = (await self._embed_guarded([text]))[0]
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
        if self._disabled:
            # A full rebuild would re-fail once per source note with
            # the same dead key — return an honest empty receipt
            # instead of a burst of doomed API calls.
            return ReindexPlan(
                layer=Layer.VECTOR,
                reason="vector indexing disabled for this session (auth breaker tripped)",
                chunks_to_reindex=0,
                requires_explicit_approval=False,
                metadata={
                    "vector_disabled": True,
                    "disabled_reason": self._disabled_reason,
                },
            )
        async with self._lock:
            await self._ensure_loaded()
            source = list(self._rows)  # snapshot rows before we rebuild
            # Embed-then-swap (audit D6): build the new index into LOCALS
            # first, so a mid-rebuild embedding failure leaves the live
            # (and on-disk) index intact instead of emptying it.
            new_vectors: List[List[float]] = []
            new_rows: List[Dict[str, Any]] = []
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
                    vectors = await self._embed_guarded(texts)  # raises before swap
                    for ref, text, vec in zip(refs, texts, vectors):
                        self._validate_dim(vec)
                        new_vectors.append(list(vec))
                        new_rows.append(_row_for(ref, text))
                        total += 1
            self._vectors = new_vectors
            self._rows = new_rows
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

    async def _embed_guarded(self, texts: Sequence[str]) -> List[List[float]]:
        """Call ``client.embed`` with breaker bookkeeping.

        Successes reset the consecutive-auth counter. Failures are
        classified via ``EmbeddingError.category`` and re-raised — the
        caller's error handling is unchanged; this layer only decides
        whether the *next* call is allowed to happen at all, and owns
        the log-level policy so the notes store's `_safe_index` no
        longer needs per-write tracebacks:

          * 'auth'      — counts toward the trip. Individual failures
                          log at DEBUG; the trip itself logs the ONE
                          warning this session will ever see.
          * 'transient' — never trips. First occurrence logs a concise
                          WARNING, repeats log at DEBUG (retry-next-
                          time stays the policy, the operator already
                          knows).
          * 'quota'     — never trips (same logging as transient: a
                          429 storm is operationally identical).
          * 'unknown'   — never trips, resets the auth streak, and
                          stays silent here so the caller's
                          traceback-logging path keeps full fidelity
                          for genuinely unexpected failures.
        """
        try:
            vectors = await self._client.embed(texts)
        except EmbeddingError as exc:
            self._note_embed_failure(exc)
            raise
        self._consecutive_auth_failures = 0
        return vectors

    def _note_embed_failure(self, exc: EmbeddingError) -> None:
        category = getattr(exc, "category", "unknown")
        if category == "auth":
            self._consecutive_auth_failures += 1
            if self._consecutive_auth_failures >= AUTH_TRIP_THRESHOLD and not self._disabled:
                self._disabled = True
                self._disabled_reason = str(exc)
                logger.warning(
                    "vector indexing disabled for this session: %d consecutive "
                    "auth failures (%s); re-enable by fixing credentials and "
                    "restarting. Markdown writes continue unaffected.",
                    self._consecutive_auth_failures,
                    exc,
                )
            else:
                logger.debug(
                    "embedding auth failure %d/%d: %s",
                    self._consecutive_auth_failures,
                    AUTH_TRIP_THRESHOLD,
                    exc,
                )
            return
        self._consecutive_auth_failures = 0
        if category in ("transient", "quota"):
            if self._transient_failure_seen:
                logger.debug("embedding %s failure (will retry next call): %s", category, exc)
            else:
                self._transient_failure_seen = True
                logger.warning("embedding %s failure (will retry next call): %s", category, exc)

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
                    else:
                        # bin/meta out of sync (a crash between the two
                        # writes, pre-2.51 non-atomic flush). Don't drop the
                        # index silently — WARN so it's visible; the notes
                        # are authoritative and a reindex rebuilds vectors
                        # from them (audit D6).
                        logger.warning(
                            "vector index bin/meta size mismatch (%d bytes vs "
                            "%d expected for %d rows × %d dims) — vectors not "
                            "loaded; reindex from notes to rebuild.",
                            len(raw), expected, count, dim,
                        )
        self._loaded = True

    def _flush(self) -> None:
        # Atomic tmp+replace for BOTH files (audit D6): the pre-2.51 direct
        # writes could tear mid-write on a crash, and a half-written bin
        # against a full meta made ``_ensure_loaded`` silently drop the
        # whole index. os.replace is atomic per file, so neither file is
        # ever observed half-written; the bin is swapped in BEFORE the meta
        # so a crash between the two leaves the OLD (consistent) meta.
        dim = self.descriptor.dimension
        bin_path = self._layout.vector_index.with_suffix(".bin")
        meta_path = self._layout.vector_metadata
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        flat: List[float] = []
        for vec in self._vectors:
            flat.extend(vec)
        if flat:
            tmp_bin = bin_path.with_suffix(bin_path.suffix + ".tmp")
            tmp_bin.write_bytes(struct.pack(f"<{len(flat)}f", *flat))
            os.replace(tmp_bin, bin_path)
        elif bin_path.exists():
            bin_path.unlink()
        payload = {
            "dimension": dim,
            "model": f"{self.descriptor.provider}/{self.descriptor.model}",
            "metric": self.descriptor.metric,
            "rows": list(self._rows),
        }
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_meta.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_meta, meta_path)

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


def _text_sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _row_for(ref: NoteRef, text: str) -> Dict[str, Any]:
    return {
        "filename": ref.filename,
        "scope": ref.scope.value if isinstance(ref.scope, Scope) else str(ref.scope),
        "category": ref.category,
        "backend": ref.backend,
        "preview": (text or "")[:400],
        # Content fingerprint — lets index_batch (session-resume backfill)
        # skip re-embedding unchanged notes (2.64.3). Rows persisted before
        # this field simply lack it and re-embed once, then carry it.
        "sha": _text_sha(text),
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
