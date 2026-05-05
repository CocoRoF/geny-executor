"""``MemoryAwareRetriever`` — provider-driven Stage 2 retrieval.

Replaces the legacy ``GenyMemoryRetriever`` (host-manager duck-type)
with a generic implementation that talks to a ``MemoryProvider``
directly. All retrieval policy lives in ``MemoryHooks`` (host attaches
it via ``provider.set_hooks(hooks)`` and passes the same instance
into the retriever); the retriever itself never touches host code.

Layer order (mirrors plan §EXEC-1):

    L0  recent_turns      ← STMHandle.recent(n=hooks.recent_turns)
    L1  session_summary   ← STMHandle.read_summary()  (D1: written by stage 19 at session close)
    L1.5 pinned           ← NotesHandle.load_pinned(category=hooks.pin_category, max_chars=…)
    L1.7 vault_map        ← IndexHandle.render_vault_map(category_descriptions=hooks.vault_descriptions)
    L2  ltm_main          ← LTMHandle.read_main()
    L3  vector            ← VectorHandle.search(query, top_k=hooks.max_results)
    L4  keyword           ← NotesHandle.search(query, …) + LTMHandle.search(query)
    L5  backlink          ← NotesHandle.read(filename) + IndexHandle.graph()
    L6  curated           ← provider.curated().notes().search(query, …)

The retriever is **stateless w.r.t. host objects** — every host policy
input arrives through ``hooks`` (a ``MemoryHooks`` instance shared
with the provider). ``llm_gate`` is a free callable not part of the
hooks bag because it is per-turn and not policy.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import MemoryHooks, MemoryProvider
from geny_executor.stages.s02_context.interface import MemoryRetriever
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)


def _layer_cap(hooks: MemoryHooks, layer: str) -> int:
    """Resolve the per-layer character cap from the hooks bag."""
    ratio = hooks.layer_budget_ratio.get(layer, 0.0)
    return max(0, int(hooks.max_inject_chars * float(ratio)))


class MemoryAwareRetriever(MemoryRetriever):
    """Provider-driven 6-layer memory retriever for Stage 2.

    Args:
        provider: Live ``MemoryProvider`` (typically a
            ``CompositeMemoryProvider``). All reads route through it.
        hooks: ``MemoryHooks`` carrying the retrieval policy. Should
            be the same instance the provider was attached with via
            ``provider.set_hooks(hooks)`` — keeps every layer (stage
            2 retrieval, stage 18 record, archivers) on the same
            policy view.
        llm_gate: Optional async callable that decides whether memory
            is needed at all for this turn. Signature: ``async (query)
            -> bool``. When ``False``, the retriever returns an empty
            list. When ``None``, memory is always retrieved.
    """

    def __init__(
        self,
        provider: MemoryProvider,
        *,
        hooks: Optional[MemoryHooks] = None,
        llm_gate: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> None:
        if provider is None:
            raise ValueError("MemoryAwareRetriever requires a non-None provider")
        self._provider = provider
        self._hooks = hooks or MemoryHooks()
        self._llm_gate = llm_gate

    @property
    def name(self) -> str:
        return "memory_aware"

    @property
    def description(self) -> str:
        return "Provider-driven 6-layer memory retrieval (STM / LTM / Notes / Vector / Index)"

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        hooks = self._hooks
        if not query or not query.strip():
            self._emit_empty(state, query, reason="empty_query")
            return []

        search_query = query[: hooks.search_chars].strip()

        if self._llm_gate is not None:
            try:
                if not await self._llm_gate(search_query):
                    self._emit_empty(state, query, reason="llm_gate_skip")
                    return []
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory_aware: llm_gate failed (%s); proceeding", exc)

        chunks: List[MemoryChunk] = []
        total = 0
        budget = hooks.max_inject_chars
        breakdown: Dict[str, int] = {}

        def _record(layer: str, before: int) -> None:
            breakdown[layer] = sum(1 for c in chunks if (c.metadata or {}).get("layer") == layer)
            del before  # parameter present for symmetry only

        # ── L0: recent STM tail ─────────────────────────────────────
        if hooks.recent_turns > 0:
            before = total
            total = await self._load_recent_turns(chunks, total, budget, hooks)
            _record("recent_turns", before)

        # ── L1: session summary (D1: stage 19 writes at session close) ──
        before = total
        total = await self._load_session_summary(chunks, total, budget, hooks)
        _record("session_summary", before)

        # ── L1.5: pinned facts (always-inject, host-policy category) ──
        before = total
        total = await self._load_pinned_facts(chunks, total, budget, hooks)
        _record("pinned", before)

        # ── L1.7: vault map (lightweight directory hint) ────────────
        if hooks.slim_mode or hooks.always_render_vault_map:
            before = total
            total = await self._load_vault_map(chunks, total, budget, hooks)
            _record("vault_map", before)

        if hooks.slim_mode:
            self._emit_breakdown(state, query, breakdown, total, len(chunks), slim=True)
            return chunks

        # ── L2: LTM main body ───────────────────────────────────────
        before = total
        total = await self._load_ltm_main(chunks, total, budget, hooks)
        _record("ltm_main", before)

        # ── L3: vector semantic search ──────────────────────────────
        if hooks.enable_vector_search:
            before = total
            total = await self._load_vector(chunks, search_query, total, budget, hooks)
            _record("vector", before)

        # ── L4: keyword search (notes + LTM) ────────────────────────
        before = total
        total = await self._load_keyword(chunks, search_query, total, budget, hooks)
        _record("keyword", before)

        # ── L5: backlink expansion (graph-driven) ───────────────────
        before = total
        total = await self._load_backlinks(chunks, total, budget, hooks)
        _record("backlink", before)

        # ── L6: curated knowledge (cross-scope) ─────────────────────
        before = total
        total = await self._load_curated(chunks, search_query, total, budget, hooks)
        _record("curated", before)

        self._emit_breakdown(state, query, breakdown, total, len(chunks), slim=False)
        if not chunks:
            self._emit_empty(state, query, reason="no_layers_matched")
        logger.info(
            "memory_aware: %d chunks (%d chars) for session %s",
            len(chunks),
            total,
            state.session_id,
        )
        return chunks

    # ── L0 ──────────────────────────────────────────────────────────

    async def _load_recent_turns(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        try:
            stm = self._provider.stm()
            turns = await stm.recent(n=hooks.recent_turns)
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: stm.recent failed", exc_info=True)
            return total
        if not turns:
            return total

        lines: List[str] = []
        for t in turns:
            content = getattr(t, "content", "") or ""
            if isinstance(content, list):
                # text-only flatten
                pieces = [
                    str(b.get("text", ""))
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(p for p in pieces if p)
            if not content or not str(content).strip():
                continue
            role = getattr(t, "role", "user") or "user"
            lines.append(f"[{role}] {str(content).strip()}")
        if not lines:
            return total

        body = "\n".join(lines)
        cap = _layer_cap(hooks, "recent_turns")
        if cap and len(body) > cap:
            body = body[-cap:]  # keep most-recent tail
        if total + len(body) > budget:
            return total
        chunks.append(
            MemoryChunk(
                key="recent_turns",
                content=body,
                source="short_term",
                relevance_score=1.0,
                metadata={"layer": "recent_turns", "turns": len(lines)},
            )
        )
        return total + len(body)

    # ── L1 ──────────────────────────────────────────────────────────

    async def _load_session_summary(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        """Read the host-managed `transcripts/summary.md`.

        D1 decision: stage 19 writes this at session close. Outside a
        session-close run the file may not exist — the call is then a
        silent no-op via the protocol's optional ``read_summary``.
        """
        try:
            stm = self._provider.stm()
            reader = getattr(stm, "read_summary", None)
            if reader is None:
                return total
            summary = await reader()
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: stm.read_summary failed", exc_info=True)
            return total
        if not summary:
            return total
        cap = _layer_cap(hooks, "session_summary") or budget
        body = summary[:cap]
        if total + len(body) > budget:
            return total
        chunks.append(
            MemoryChunk(
                key="session_summary",
                content=body,
                source="short_term",
                relevance_score=1.0,
                metadata={"layer": "session_summary"},
            )
        )
        return total + len(body)

    # ── L1.5 ────────────────────────────────────────────────────────

    async def _load_pinned_facts(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        cap = _layer_cap(hooks, "pinned")
        if cap <= 0:
            return total
        try:
            notes = self._provider.notes()
            content = await notes.load_pinned(category=hooks.pin_category, max_chars=cap)
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: notes.load_pinned failed", exc_info=True)
            return total
        if not content or not str(content).strip():
            return total
        body = str(content)
        if total + len(body) > budget:
            return total
        chunks.append(
            MemoryChunk(
                key="pinned_facts",
                content=body,
                source="pinned",
                relevance_score=2.0,
                metadata={
                    "layer": "pinned",
                    "host_layer": hooks.pin_category,
                    "char_count": len(body),
                },
            )
        )
        return total + len(body)

    # ── L1.7 ────────────────────────────────────────────────────────

    async def _load_vault_map(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        try:
            idx = self._provider.index()
            rendered = await idx.render_vault_map(
                category_descriptions=hooks.vault_descriptions or None,
            )
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: index.render_vault_map failed", exc_info=True)
            return total
        if not rendered:
            return total
        cap = hooks.vault_map_max_chars or _layer_cap(hooks, "vault_map") or budget
        body = rendered[:cap]
        if total + len(body) > budget:
            return total
        chunks.append(
            MemoryChunk(
                key="vault_map",
                content=body,
                source="vault_map",
                relevance_score=1.0,
                metadata={"layer": "vault_map"},
            )
        )
        return total + len(body)

    # ── L2 ──────────────────────────────────────────────────────────

    async def _load_ltm_main(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        try:
            ltm = self._provider.ltm()
            body = await ltm.read_main()
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: ltm.read_main failed", exc_info=True)
            return total
        if not body or not str(body).strip():
            return total
        cap = _layer_cap(hooks, "ltm_main") or budget
        text = str(body)[:cap]
        if total + len(text) > budget:
            return total
        chunks.append(
            MemoryChunk(
                key="MEMORY.md",
                content=text,
                source="long_term",
                relevance_score=1.0,
                metadata={"layer": "ltm_main", "char_count": len(text)},
            )
        )
        return total + len(text)

    # ── L3 ──────────────────────────────────────────────────────────

    async def _load_vector(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        if budget - total <= 200:
            return total
        try:
            vec = self._provider.vector()
            if vec is None:
                return total
            hits = await vec.search(query, top_k=hooks.max_results)
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: vector.search failed", exc_info=True)
            return total
        if not hits:
            return total
        already = {c.key for c in chunks}
        for h in hits:
            text = h.content or ""
            if not text or h.key in already:
                continue
            if total + len(text) > budget:
                break
            chunks.append(
                MemoryChunk(
                    key=h.key,
                    content=text,
                    source="vector",
                    relevance_score=h.relevance_score,
                    metadata={"layer": "vector", **(h.metadata or {})},
                )
            )
            total += len(text)
            already.add(h.key)
        return total

    # ── L4 ──────────────────────────────────────────────────────────

    async def _load_keyword(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        if budget - total <= 200:
            return total

        # NotesHandle.search and LTMHandle.search both return
        # ``List[MemoryChunk]`` per the protocol. Boost fields live on
        # the chunk's ``metadata``.
        results: List[MemoryChunk] = []
        try:
            notes = self._provider.notes()
            results.extend(list(await notes.search(query, limit=hooks.max_results)))
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: notes.search failed", exc_info=True)
        try:
            ltm = self._provider.ltm()
            ltm_hits = await ltm.search(query, limit=hooks.max_results)
            results.extend(list(ltm_hits or []))
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: ltm.search failed", exc_info=True)

        if not results:
            return total

        query_words = {w for w in query.lower().split() if w}
        for r in results:
            meta = r.metadata or {}
            importance = str(meta.get("importance", "medium")).lower()
            base = float(r.relevance_score or 0.0)
            base *= float(hooks.importance_boost.get(importance, 1.0))

            tags = meta.get("tags") or []
            if tags and query_words:
                tag_words = {str(t).lower() for t in tags}
                base *= 1.0 + 0.3 * len(query_words & tag_words)

            if hooks.category_boosts:
                cat = meta.get("category")
                if isinstance(cat, str):
                    boost = hooks.category_boosts.get(cat)
                    if boost is not None:
                        base *= float(boost)
            r.relevance_score = base

        results.sort(key=lambda c: c.relevance_score, reverse=True)

        already = {c.key for c in chunks}
        for r in results:
            text = r.content or ""
            if not text:
                continue
            if r.key in already:
                continue
            if total + len(text) > budget:
                break
            chunks.append(
                MemoryChunk(
                    key=r.key,
                    content=text,
                    source=r.source or "keyword",
                    relevance_score=r.relevance_score,
                    metadata={"layer": "keyword", **(r.metadata or {})},
                )
            )
            total += len(text)
            already.add(r.key)
        return total

    # ── L5 ──────────────────────────────────────────────────────────

    async def _load_backlinks(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        if budget - total <= 200 or not chunks:
            return total
        try:
            notes = self._provider.notes()
            graph = await notes.graph()
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: notes.graph failed", exc_info=True)
            return total
        edges = getattr(graph, "edges", None) or []
        if not edges:
            return total
        # Build adjacency from edges (tuples of (src, tgt))
        adj: Dict[str, List[str]] = {}
        for e in edges:
            try:
                src, tgt = e[0], e[1]
            except (TypeError, IndexError, KeyError):
                continue
            adj.setdefault(str(src), []).append(str(tgt))

        already = {c.key for c in chunks}
        seeds = [c.key for c in list(chunks)]
        for seed in seeds:
            for tgt in adj.get(seed, []):
                if tgt in already:
                    continue
                try:
                    note = await notes.read(tgt)
                except Exception:  # noqa: BLE001
                    continue
                if note is None:
                    continue
                body = (getattr(note, "body", "") or "")[:800]
                if not body:
                    continue
                if total + len(body) > budget:
                    return total
                chunks.append(
                    MemoryChunk(
                        key=tgt,
                        content=body,
                        source="backlink",
                        relevance_score=0.5,
                        metadata={"layer": "backlink", "linked_from": seed},
                    )
                )
                total += len(body)
                already.add(tgt)
        return total

    # ── L6 ──────────────────────────────────────────────────────────

    async def _load_curated(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
        hooks: MemoryHooks,
    ) -> int:
        if budget - total <= 200:
            return total
        try:
            curated = self._provider.curated()
            if curated is None:
                return total
            curated_notes = curated.notes()
            hits = await curated_notes.search(query, limit=hooks.max_results)
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: curated search failed", exc_info=True)
            return total
        if not hits:
            return total
        already = {c.key for c in chunks}
        for h in hits:
            text = h.content or ""
            if not text or h.key in already:
                continue
            if total + len(text) > budget:
                break
            chunks.append(
                MemoryChunk(
                    key=h.key,
                    content=text,
                    source="curated",
                    relevance_score=h.relevance_score,
                    metadata={"layer": "curated", **(h.metadata or {})},
                )
            )
            total += len(text)
            already.add(h.key)
        return total

    # ── observability ───────────────────────────────────────────────

    def _emit_breakdown(
        self,
        state: PipelineState,
        query: str,
        breakdown: Dict[str, int],
        total_chars: int,
        chunk_count: int,
        *,
        slim: bool,
    ) -> None:
        try:
            state.add_event(
                "memory.retrieve_breakdown",
                {
                    "query_preview": str(query)[:120],
                    "layers": dict(breakdown),
                    "total_chars": int(total_chars),
                    "chunk_count": int(chunk_count),
                    "slim_mode": bool(slim),
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: breakdown emit failed", exc_info=True)

    def _emit_empty(self, state: PipelineState, query: str, *, reason: str) -> None:
        try:
            state.add_event(
                "memory.retrieved_empty",
                {
                    "query_preview": str(query)[:120],
                    "reason": reason,
                    "session_id": getattr(state, "session_id", ""),
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("memory_aware: empty emit failed", exc_info=True)


__all__ = ["MemoryAwareRetriever"]
