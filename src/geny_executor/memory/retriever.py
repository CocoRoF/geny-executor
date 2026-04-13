"""GenyMemoryRetriever — Geny-compatible 5-layer memory retrieval.

Implements the MemoryRetriever interface (S02 Context) using Geny's
SessionMemoryManager. Replicates the logic from Geny's MemoryInjectNode:

1. Session summary (short-term)
2. MEMORY.md (long-term persistent notes)
3. FAISS vector semantic search (if enabled)
4. Keyword-based memory recall (with importance weighting)
5. Backlink context (linked notes)
6. Curated Knowledge (optional)

The memory_manager is injected as a duck-typed dependency — any object
that satisfies the expected interface works (Geny's SessionMemoryManager
is the canonical implementation).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.interface import MemoryRetriever
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)

# Importance boost factors (mirrors Geny's MemoryInjectNode)
_IMPORTANCE_BOOST = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


class GenyMemoryRetriever(MemoryRetriever):
    """5-layer memory retrieval compatible with Geny's SessionMemoryManager.

    Args:
        memory_manager: Geny's SessionMemoryManager (or duck-typed equivalent).
        enable_vector_search: Enable FAISS vector semantic search.
        max_results: Maximum memory chunks per search type.
        max_inject_chars: Total character budget for memory context.
        search_chars: Character limit of input used for search query.
        llm_gate: Optional async callable that decides if memory is needed.
            Signature: ``async (query: str) -> bool``.
            When None, memory is always retrieved.
        curated_knowledge_manager: Optional CuratedKnowledgeManager for
            curated knowledge injection.
    """

    def __init__(
        self,
        memory_manager: Any,
        *,
        enable_vector_search: bool = True,
        max_results: int = 5,
        max_inject_chars: int = 10000,
        search_chars: int = 500,
        llm_gate: Optional[Callable[[str], Awaitable[bool]]] = None,
        curated_knowledge_manager: Any = None,
    ):
        self._mgr = memory_manager
        self._enable_vector = enable_vector_search
        self._max_results = max_results
        self._max_inject = max_inject_chars
        self._search_chars = search_chars
        self._llm_gate = llm_gate
        self._curated = curated_knowledge_manager

    @property
    def name(self) -> str:
        return "geny_memory"

    @property
    def description(self) -> str:
        return "Geny-compatible 5-layer memory retrieval"

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        if not self._mgr or not query.strip():
            return []

        search_query = query[: self._search_chars].strip()

        # LLM gate: let caller decide if memory is needed
        if self._llm_gate is not None:
            try:
                needs_memory = await self._llm_gate(search_query)
                if not needs_memory:
                    logger.debug("geny_retriever: LLM gate skipped memory retrieval")
                    return []
            except Exception as exc:
                logger.warning("geny_retriever: LLM gate failed (%s), proceeding", exc)

        chunks: List[MemoryChunk] = []
        total_chars = 0
        budget = self._max_inject

        # 1. Session summary (short-term latest state)
        total_chars = self._load_session_summary(chunks, total_chars, budget)

        # 2. MEMORY.md (persistent long-term notes)
        total_chars = self._load_main_memory(chunks, total_chars, budget)

        # 3. FAISS vector semantic search
        if self._enable_vector:
            total_chars = await self._load_vector_memory(chunks, search_query, total_chars, budget)

        # 4. Keyword-based recall with importance weighting
        total_chars = self._load_keyword_memory(chunks, search_query, total_chars, budget)

        # 5. Backlink context
        total_chars = self._load_backlink_context(chunks, total_chars, budget)

        # 6. Curated Knowledge (optional)
        if self._curated:
            total_chars = self._load_curated_knowledge(chunks, search_query, total_chars, budget)

        logger.info(
            "geny_retriever: loaded %d chunks (%d chars) for session %s",
            len(chunks),
            total_chars,
            state.session_id,
        )
        return chunks

    # ── Layer 1: Session Summary ─────────────────────────────────────

    def _load_session_summary(self, chunks: List[MemoryChunk], total: int, budget: int) -> int:
        try:
            stm = getattr(self._mgr, "short_term", None)
            if stm is None:
                return total
            summary = stm.get_summary()
            if summary and (total + len(summary)) <= budget:
                chunks.append(
                    MemoryChunk(
                        key="session_summary",
                        content=summary,
                        source="short_term",
                        relevance_score=1.0,
                        metadata={"layer": "session_summary"},
                    )
                )
                return total + len(summary)
        except Exception:
            logger.debug("geny_retriever: session summary load failed", exc_info=True)
        return total

    # ── Layer 2: MEMORY.md ───────────────────────────────────────────

    def _load_main_memory(self, chunks: List[MemoryChunk], total: int, budget: int) -> int:
        try:
            ltm = getattr(self._mgr, "long_term", None)
            if ltm is None:
                return total
            main_mem = ltm.load_main()
            if main_mem and main_mem.content and (total + main_mem.char_count) <= budget:
                chunks.append(
                    MemoryChunk(
                        key=main_mem.filename or "MEMORY.md",
                        content=main_mem.content,
                        source="long_term",
                        relevance_score=1.0,
                        metadata={
                            "layer": "long_term_main",
                            "char_count": main_mem.char_count,
                        },
                    )
                )
                return total + main_mem.char_count
        except Exception:
            logger.debug("geny_retriever: MEMORY.md load failed", exc_info=True)
        return total

    # ── Layer 3: FAISS Vector Search ─────────────────────────────────

    async def _load_vector_memory(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
    ) -> int:
        try:
            vmm = getattr(self._mgr, "vector_memory", None)
            if vmm is None or not getattr(vmm, "enabled", False):
                return total

            v_results = await vmm.search(query, top_k=self._max_results)
            if not v_results:
                return total

            for vr in v_results:
                text = getattr(vr, "text", "")
                source_file = getattr(vr, "source_file", "vector")
                score = getattr(vr, "score", 0.0)
                chunk_len = len(text)

                if (total + chunk_len) > budget:
                    break

                chunks.append(
                    MemoryChunk(
                        key=source_file,
                        content=text,
                        source="vector",
                        relevance_score=score,
                        metadata={
                            "layer": "vector",
                            "chunk_index": getattr(vr, "chunk_index", 0),
                        },
                    )
                )
                total += chunk_len

        except Exception:
            logger.debug("geny_retriever: vector search failed", exc_info=True)
        return total

    # ── Layer 4: Keyword Search + Importance Weighting ───────────────

    def _load_keyword_memory(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
    ) -> int:
        remaining = budget - total
        if remaining <= 200:
            return total

        try:
            results = self._mgr.search(query, max_results=self._max_results)
            if not results:
                return total

            # Apply importance boost (mirrors Geny's MemoryInjectNode)
            query_words = set(query.lower().split())
            for r in results:
                entry = getattr(r, "entry", r)
                importance = getattr(entry, "importance", "medium") or "medium"
                r.score *= _IMPORTANCE_BOOST.get(importance, 1.0)

                tags = getattr(entry, "tags", None) or []
                if tags and query_words:
                    tag_words = {t.lower() for t in tags}
                    overlap = len(query_words & tag_words)
                    r.score *= 1.0 + 0.3 * overlap

            results.sort(key=lambda r: r.score, reverse=True)

            already_loaded = {c.key for c in chunks}

            for r in results:
                entry = getattr(r, "entry", r)
                filename = getattr(entry, "filename", None) or "keyword_result"
                if filename in already_loaded:
                    continue

                snippet = getattr(r, "snippet", "") or getattr(entry, "content", "")
                chunk_len = len(snippet)

                if (total + chunk_len) > budget:
                    break

                chunks.append(
                    MemoryChunk(
                        key=filename,
                        content=snippet,
                        source=getattr(entry, "source", {}).value
                        if hasattr(getattr(entry, "source", None), "value")
                        else str(getattr(entry, "source", "keyword")),
                        relevance_score=r.score,
                        metadata={
                            "layer": "keyword",
                            "importance": getattr(entry, "importance", "medium"),
                        },
                    )
                )
                total += chunk_len
                already_loaded.add(filename)

        except Exception:
            logger.debug("geny_retriever: keyword search failed", exc_info=True)
        return total

    # ── Layer 5: Backlink Context ────────────────────────────────────

    def _load_backlink_context(self, chunks: List[MemoryChunk], total: int, budget: int) -> int:
        remaining = budget - total
        if remaining <= 200 or not chunks:
            return total

        try:
            read_note = getattr(self._mgr, "read_note", None)
            if read_note is None:
                return total

            already_loaded = {c.key for c in chunks}

            for chunk in list(chunks):  # iterate copy
                note = read_note(chunk.key)
                if note is None:
                    continue

                meta = note.get("metadata") or {}
                for linked_fn in meta.get("links_to", []):
                    if linked_fn in already_loaded:
                        continue

                    linked_note = read_note(linked_fn)
                    if linked_note is None:
                        continue

                    body = (linked_note.get("body") or "")[:800]
                    if not body:
                        continue

                    if (total + len(body)) > budget:
                        return total

                    chunks.append(
                        MemoryChunk(
                            key=linked_fn,
                            content=body,
                            source="backlink",
                            relevance_score=0.5,
                            metadata={
                                "layer": "backlink",
                                "linked_from": chunk.key,
                            },
                        )
                    )
                    total += len(body)
                    already_loaded.add(linked_fn)

        except Exception:
            logger.debug("geny_retriever: backlink load failed", exc_info=True)
        return total

    # ── Layer 6: Curated Knowledge ───────────────────────────────────

    def _load_curated_knowledge(
        self,
        chunks: List[MemoryChunk],
        query: str,
        total: int,
        budget: int,
    ) -> int:
        remaining = budget - total
        if remaining <= 200 or self._curated is None:
            return total

        try:
            ck_text = self._curated.inject_context(query, max_chars=remaining)
            if ck_text:
                chunks.append(
                    MemoryChunk(
                        key="curated_knowledge",
                        content=ck_text,
                        source="curated_knowledge",
                        relevance_score=0.8,
                        metadata={"layer": "curated_knowledge"},
                    )
                )
                total += len(ck_text)
        except Exception:
            logger.debug("geny_retriever: curated knowledge failed", exc_info=True)
        return total
