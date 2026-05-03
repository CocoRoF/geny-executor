"""GenyMemoryRetriever — Geny-compatible memory retrieval.

Implements the MemoryRetriever interface (S02 Context) using Geny's
SessionMemoryManager (or any duck-typed equivalent). Layer order:

0. Recent STM tail (always, capped budget share).
1. Session summary (short-term).
1.5. Pinned facts — host-supplied "always-inject" content
   (duck-typed ``mgr.load_pinned(max_chars)``). Empty when the host
   does not implement it; non-empty when the host promotes
   high-importance insights to a pinned surface.
1.7. Vault map — host-supplied ``mgr.index_manager.render_vault_map()``
   short index. Always injected when available (Memory v2 PR 12 cycle:
   slim_mode is no longer the only switch — the lite map is cheap
   enough to ship with every retrieve so the agent always has a
   directory hint).
2. MEMORY.md (long-term main).
3. Vector semantic search (if enabled).
4. Keyword-based memory recall (with importance weighting).
5. Backlink context (linked notes).
6. Curated Knowledge (optional).

The memory_manager is injected as a duck-typed dependency — any object
that satisfies the expected interface works (Geny's SessionMemoryManager
is the canonical implementation). Hosts that do not implement the
optional surfaces (``load_pinned``, ``index_manager.render_vault_map``,
``vector_memory``, ``read_note``, etc.) silently fall through; this
keeps the retriever a *general* component and lets each host opt into
the calls it can actually serve.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

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
        recent_turns: Size of the L0 tail (STM transcript window) that
            is always injected before semantic/keyword layers run. Set
            to 0 to disable. Useful for trigger-style turns (idle
            reflection, sub-worker auto-reports) whose query text has
            no lexical overlap with the prior conversation — they would
            otherwise miss the last few turns entirely.
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
        recent_turns: int = 6,
        # Memory v2 PR 10 — slim mode (plan §5.2). When True, the
        # retriever returns ONLY the lightweight layers (recent
        # turns, session summary, pinned facts, vault map). Heavy
        # layers (MEMORY.md body, vector top-k, keyword recall,
        # backlinks, curated) are reserved for the agent's tool
        # calls (memory_search / memory_read). Default is False so
        # existing callers preserve legacy 6-layer behaviour until
        # they explicitly opt in.
        slim_mode: bool = False,
        # Memory v2 PR 12 — pinned-facts layer. ``pin_budget_ratio``
        # is the share of ``max_inject_chars`` reserved for the
        # host-supplied pinned surface (``mgr.load_pinned(max_chars)``).
        # Defaults to 0.30 — i.e. up to 30% of the budget is
        # reserved for "must-always-be-known" facts the host has
        # promoted (e.g. user preferences, persona-defining facts).
        # Set to 0.0 to disable; the layer also no-ops automatically
        # when the host does not implement ``load_pinned``.
        pin_budget_ratio: float = 0.30,
        # ``category_boosts`` — multiplicative score boost applied
        # to L4 keyword-search results based on the entry's
        # category. Pure policy: callers wire whatever weights they
        # want (e.g. ``{"insights": 1.2, "projects": 1.2}``).
        # Empty dict → no boost (legacy behaviour).
        category_boosts: Optional[Dict[str, float]] = None,
        # ``always_render_vault_map`` — when True, the small vault
        # map (host's ``index_manager.render_vault_map()``) is
        # injected even when ``slim_mode`` is False. Caps at
        # ``vault_map_max_chars``. Default True — the map is small
        # (~500 chars) and gives the agent a "what's in memory"
        # signal regardless of search hits.
        always_render_vault_map: bool = True,
        vault_map_max_chars: int = 500,
    ):
        self._mgr = memory_manager
        self._enable_vector = enable_vector_search
        self._max_results = max_results
        self._max_inject = max_inject_chars
        self._search_chars = search_chars
        self._llm_gate = llm_gate
        self._curated = curated_knowledge_manager
        self._recent_turns = recent_turns
        self._slim_mode = slim_mode
        self._pin_budget_ratio = max(0.0, min(0.7, float(pin_budget_ratio)))
        self._category_boosts: Dict[str, float] = dict(category_boosts or {})
        self._always_vault_map = bool(always_render_vault_map)
        self._vault_map_max = max(0, int(vault_map_max_chars))

    @property
    def name(self) -> str:
        return "geny_memory"

    @property
    def description(self) -> str:
        return "Geny-compatible 5-layer memory retrieval"

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        if not self._mgr or not query.strip():
            self._emit_empty(state, query, reason="no_manager_or_query")
            return []

        search_query = query[: self._search_chars].strip()

        # LLM gate: let caller decide if memory is needed
        if self._llm_gate is not None:
            try:
                needs_memory = await self._llm_gate(search_query)
                if not needs_memory:
                    logger.debug("geny_retriever: LLM gate skipped memory retrieval")
                    self._emit_empty(state, query, reason="llm_gate_skip")
                    return []
            except Exception as exc:
                logger.warning("geny_retriever: LLM gate failed (%s), proceeding", exc)

        chunks: List[MemoryChunk] = []
        total_chars = 0
        budget = self._max_inject
        # Per-layer chunk counts for the breakdown event. Indexed by
        # layer name so PR 12 observability gives operators a
        # readable distribution.
        breakdown: Dict[str, int] = {}

        def _record(layer: str) -> None:
            """Snapshot ``len(chunks)`` after a layer call so the
            breakdown reflects how many chunks each layer added.
            """
            breakdown[layer] = sum(
                1 for c in chunks
                if (c.metadata or {}).get("layer") == layer
                or c.source == layer
            )

        # 0. Recent turns (tail of the STM transcript). Always injected
        #    before semantic/keyword layers run so trigger-style queries
        #    (idle reflection, sub-worker auto-reports) still see the
        #    last few conversation turns even when the query text has
        #    zero lexical overlap with the prior dialogue.
        if self._recent_turns > 0:
            total_chars = self._load_recent_turns(chunks, total_chars, budget)
            _record("recent_turns")

        # 1. Session summary (short-term latest state)
        total_chars = self._load_session_summary(chunks, total_chars, budget)
        _record("session_summary")

        # 1.5. Pinned facts — host's ``load_pinned`` duck-type. Runs
        #      regardless of slim_mode because pinned content is the
        #      "must-always-be-known" surface (plan §3 T1 tier).
        total_chars = self._load_pinned_facts(chunks, total_chars, budget)
        _record("pinned")

        # 1.7. Vault map — small directory hint. In slim_mode this is
        #      the only structural cue the agent gets. Outside slim
        #      mode it's still injected when ``always_render_vault_map``
        #      so the agent always knows what categories exist.
        if self._slim_mode or self._always_vault_map:
            total_chars = self._load_vault_map(chunks, total_chars, budget)
            _record("vault_map")

        if self._slim_mode:
            # Memory v2 PR 10 — slim path: stop here and let the
            # agent's progressive disclosure tools (memory_search /
            # memory_read) do the rest. Plan §5.2.
            self._emit_breakdown(state, query, breakdown, total_chars, len(chunks))
            logger.info(
                "geny_retriever: slim mode loaded %d chunks (%d chars) for session %s",
                len(chunks),
                total_chars,
                state.session_id,
            )
            return chunks

        # 2. MEMORY.md (persistent long-term notes)
        total_chars = self._load_main_memory(chunks, total_chars, budget)
        _record("long_term_main")

        # 3. FAISS vector semantic search
        if self._enable_vector:
            total_chars = await self._load_vector_memory(chunks, search_query, total_chars, budget)
            _record("vector")

        # 4. Keyword-based recall with importance weighting
        total_chars = self._load_keyword_memory(chunks, search_query, total_chars, budget)
        _record("keyword")

        # 5. Backlink context
        total_chars = self._load_backlink_context(chunks, total_chars, budget)
        _record("backlink")

        # 6. Curated Knowledge (optional)
        if self._curated:
            total_chars = self._load_curated_knowledge(chunks, search_query, total_chars, budget)
            _record("curated_knowledge")

        self._emit_breakdown(state, query, breakdown, total_chars, len(chunks))
        if not chunks:
            self._emit_empty(state, query, reason="no_layers_matched")

        logger.info(
            "geny_retriever: loaded %d chunks (%d chars) for session %s",
            len(chunks),
            total_chars,
            state.session_id,
        )
        return chunks

    # ── Layer 0: Recent STM turns (tail) ─────────────────────────────

    def _load_recent_turns(
        self,
        chunks: List[MemoryChunk],
        total: int,
        budget: int,
    ) -> int:
        """Inject the last N STM messages verbatim as a L0 chunk.

        Bypasses semantic/keyword matching entirely — the goal is to
        make sure trigger-driven turns (idle reflection, sub-worker
        auto-reports) can always see the most recent conversation
        regardless of lexical overlap with their query text.

        Duck-typed: the memory manager must expose
        ``short_term.get_recent(n)`` returning an iterable of entries
        with ``.content`` and (optionally) ``.metadata["role"]``. Any
        missing attribute quietly disables this layer.

        Budget policy: capped at 40% of total so other layers still fit.
        Messages are kept tail-first — we want the most recent turns.
        """
        try:
            stm = getattr(self._mgr, "short_term", None)
            if stm is None:
                return total
            get_recent = getattr(stm, "get_recent", None)
            if get_recent is None:
                return total

            recent = get_recent(self._recent_turns)
            if not recent:
                return total

            lines: List[str] = []
            for entry in recent:
                role = "user"
                meta = getattr(entry, "metadata", None)
                if meta and isinstance(meta, dict):
                    role = meta.get("role", role) or role
                # Some implementations expose role directly on the entry.
                role = getattr(entry, "role", role) or role

                content = getattr(entry, "content", "") or ""
                if not content.strip():
                    continue
                lines.append(f"[{role}] {content.strip()}")

            if not lines:
                return total

            body = "\n".join(lines)
            max_body = min(len(body), int(budget * 0.4))
            if max_body < len(body):
                body = body[-max_body:]  # keep the most recent tail

            chunk_len = len(body)
            if (total + chunk_len) > budget:
                return total

            chunks.append(
                MemoryChunk(
                    key="recent_turns",
                    content=body,
                    source="short_term",
                    relevance_score=1.0,
                    metadata={
                        "layer": "recent_turns",
                        "turns": len(lines),
                    },
                )
            )
            return total + chunk_len

        except Exception:
            logger.debug("geny_retriever: recent turns load failed", exc_info=True)
            return total

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

    # ── Layer 1.5: Pinned Facts (always-inject T1 surface) ──────────

    def _load_pinned_facts(self, chunks: List[MemoryChunk], total: int, budget: int) -> int:
        """Inject the host's "always-pinned" facts.

        Duck-typed: the memory manager may expose
        ``load_pinned(max_chars: int) -> Optional[object]``. The
        returned object is expected to have ``.content`` (str) and
        either ``.char_count`` (int) or be measurable via ``len()``.
        Hosts that do not implement the method silently no-op so the
        retriever stays generic.

        Budget policy: capped at ``pin_budget_ratio`` of the total
        char budget. This is the reserved surface for content the
        host has decided must be in the prompt regardless of query.
        """
        if self._pin_budget_ratio <= 0.0:
            return total
        try:
            loader = getattr(self._mgr, "load_pinned", None)
            if loader is None:
                return total
            cap = max(0, int(budget * self._pin_budget_ratio))
            if cap <= 0:
                return total
            pinned = loader(max_chars=cap)
            if not pinned:
                return total
            content = getattr(pinned, "content", None)
            if not isinstance(content, str) or not content.strip():
                return total
            char_count = getattr(pinned, "char_count", None)
            if not isinstance(char_count, int) or char_count <= 0:
                char_count = len(content)
            if (total + char_count) > budget:
                return total
            chunks.append(
                MemoryChunk(
                    key="pinned_facts",
                    content=content,
                    source="pinned",
                    relevance_score=2.0,
                    metadata={
                        "layer": "pinned",
                        "char_count": char_count,
                        "host_layer": getattr(pinned, "source", "pinned"),
                    },
                )
            )
            return total + char_count
        except Exception:
            logger.debug("geny_retriever: pinned facts load failed", exc_info=True)
            return total

    # ── Layer 1.7: Vault Map (slim mode + always-on) ─────────────────

    def _load_vault_map(self, chunks: List[MemoryChunk], total: int, budget: int) -> int:
        """Inject the rendered vault map (~500 chars) so the agent
        knows *where* to look without seeing the bodies. PR 9 + PR 10.

        Duck-typed: accepts a memory_manager that exposes
        ``index_manager.render_vault_map()``. Silent skip when the
        index manager is absent (no LTM yet) or when render fails.
        """
        try:
            idx_mgr = getattr(self._mgr, "index_manager", None)
            if idx_mgr is None:
                return total
            render = getattr(idx_mgr, "render_vault_map", None)
            if render is None:
                return total
            rendered = render() or ""
            if not rendered:
                return total
            # Trim to ``vault_map_max_chars`` so the directory hint
            # does not eat the whole budget. The map is a structural
            # cue, not a body.
            if self._vault_map_max and len(rendered) > self._vault_map_max:
                rendered = rendered[: self._vault_map_max]
            if total + len(rendered) > budget:
                return total
            chunks.append(
                MemoryChunk(
                    key="vault_map",
                    content=rendered,
                    source="vault_map",
                    relevance_score=1.0,
                    metadata={"layer": "vault_map"},
                )
            )
            return total + len(rendered)
        except Exception:
            logger.debug("geny_retriever: vault_map load failed", exc_info=True)
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

                # Category boost — pure policy via the host-supplied
                # ``category_boosts`` mapping. Defaults to no boost
                # so legacy behaviour is preserved.
                if self._category_boosts:
                    category = (
                        getattr(entry, "category", None)
                        or (getattr(entry, "metadata", {}) or {}).get("category")
                    )
                    if isinstance(category, str):
                        boost = self._category_boosts.get(category)
                        if boost is not None:
                            r.score *= float(boost)

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

    # ── Observability helpers ────────────────────────────────────────

    def _emit_breakdown(
        self,
        state: PipelineState,
        query: str,
        breakdown: Dict[str, int],
        total_chars: int,
        chunk_count: int,
    ) -> None:
        """Emit ``memory.retrieve_breakdown`` so operators can see
        which layers contributed chunks. Pure observability — no
        behavioural effect.
        """
        try:
            state.add_event(
                "memory.retrieve_breakdown",
                {
                    "query_preview": str(query)[:120],
                    "layers": dict(breakdown),
                    "total_chars": int(total_chars),
                    "chunk_count": int(chunk_count),
                    "slim_mode": bool(self._slim_mode),
                },
            )
        except Exception:
            logger.debug("geny_retriever: breakdown emit failed", exc_info=True)

    def _emit_empty(self, state: PipelineState, query: str, *, reason: str) -> None:
        """Emit ``memory.retrieved_empty`` when the retriever returns
        no chunks. Lets the host raise an alert / surface a metric
        when the pinned + search layers all whiff.
        """
        try:
            state.add_event(
                "memory.retrieved_empty",
                {
                    "query_preview": str(query)[:120],
                    "reason": reason,
                    "session_id": getattr(state, "session_id", ""),
                },
            )
        except Exception:
            logger.debug("geny_retriever: empty emit failed", exc_info=True)
