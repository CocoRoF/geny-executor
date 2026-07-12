"""Stage 2: Context — concrete stage implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.compaction import reconcile_recorded_index, run_compaction
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens
from geny_executor.memory.provider import (
    MemoryEvent,
    MemoryProvider,
    RetrievalQuery,
)
from geny_executor.stages.s02_context.interface import (
    ContextStrategy,
    HistoryCompactor,
    MemoryRetriever,
)
from geny_executor.stages.s02_context.artifact.default.strategies import (
    HybridStrategy,
    ProgressiveDisclosureStrategy,
    SimpleLoadStrategy,
)
from geny_executor.stages.s02_context.artifact.default.compactors import (
    LLMSummaryCompactor,
    SlidingWindowCompactor,
    SummaryCompactor,
    TruncateCompactor,
)
from geny_executor.stages.s02_context.artifact.default.retrievers import (
    NullRetriever,
    StaticRetriever,
)

logger = logging.getLogger(__name__)


class _CompactionShadow:
    """Minimal state stand-in for background compaction (TTFT program).

    ``LLMSummaryCompactor.compact`` reads ``messages`` / ``model`` /
    ``llm_client`` and assigns ``messages``; events it emits are
    collected here and replayed onto the real state when the result is
    applied, so observability is preserved turn-shifted.
    """

    def __init__(self, state: PipelineState):
        self.messages: List[Dict[str, Any]] = list(state.messages)
        self.model = getattr(state, "model", "")
        self.llm_client = getattr(state, "llm_client", None)
        self.context_window_budget = getattr(state, "context_window_budget", 200_000)
        self.shared: Dict[str, Any] = {}
        self.events: List[tuple] = []

    def add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.events.append((event_type, dict(data or {})))


class ContextStage(Stage[Any, Any]):
    """Stage 2: Context.

    Dual abstraction:
      - Level 2 context_strategy: how to collect context
      - Level 2 compactor: how to compress when over budget
      - Level 2 retriever: how to fetch memory

    Phase 1+ also accepts an optional :class:`MemoryProvider`. When
    set, the unified `provider.retrieve(RetrievalQuery)` is invoked
    *in addition to* the legacy retriever. Provider chunks are merged
    after legacy retriever output, deduplicated by `key`. The result
    is rendered into `state.metadata["memory_context"]` (string form
    suitable for prompt injection).
    """

    def __init__(
        self,
        strategy: Optional[ContextStrategy] = None,
        compactor: Optional[HistoryCompactor] = None,
        retriever: Optional[MemoryRetriever] = None,
        *,
        stateless: bool = False,
        provider: Optional[MemoryProvider] = None,
        retrieval_timeout_s: float = 10.0,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "strategy": StrategySlot(
                name="strategy",
                strategy=strategy or SimpleLoadStrategy(),
                registry={
                    "simple_load": SimpleLoadStrategy,
                    "hybrid": HybridStrategy,
                    "progressive_disclosure": ProgressiveDisclosureStrategy,
                },
                description="Context collection strategy",
            ),
            "compactor": StrategySlot(
                name="compactor",
                strategy=compactor or TruncateCompactor(),
                registry={
                    "truncate": TruncateCompactor,
                    "summary": SummaryCompactor,
                    "llm_summary": LLMSummaryCompactor,
                    "sliding_window": SlidingWindowCompactor,
                },
                description="History compaction strategy",
            ),
            "retriever": StrategySlot(
                name="retriever",
                strategy=retriever or NullRetriever(),
                registry={
                    "null": NullRetriever,
                    "static": StaticRetriever,
                },
                description="Memory retrieval strategy",
            ),
        }
        self._stateless = stateless
        self._provider = provider
        self._retrieval_timeout_s = max(0.0, float(retrieval_timeout_s))
        # In-flight background compaction (TTFT program, finding B3):
        # {"task": asyncio.Task[_CompactionShadow], "len": int, "tail_id": int}
        self._bg_compaction: Optional[Dict[str, Any]] = None

    @property
    def provider(self) -> Optional[MemoryProvider]:
        return self._provider

    @provider.setter
    def provider(self, value: Optional[MemoryProvider]) -> None:
        self._provider = value

    @property
    def _strategy(self) -> ContextStrategy:
        return self._slots["strategy"].strategy  # type: ignore[return-value]

    @property
    def _compactor(self) -> HistoryCompactor:
        return self._slots["compactor"].strategy  # type: ignore[return-value]

    @property
    def _retriever(self) -> MemoryRetriever:
        return self._slots["retriever"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "context"

    @property
    def order(self) -> int:
        return 2

    @property
    def category(self) -> str:
        return "ingress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="context",
            fields=[
                ConfigField(
                    name="stateless",
                    type="boolean",
                    label="Stateless",
                    description="Bypass context assembly (no conversation history).",
                    default=False,
                    ui_widget="toggle",
                ),
                ConfigField(
                    name="retrieval_timeout_s",
                    type="number",
                    label="Retrieval timeout (s)",
                    description=(
                        "Upper bound on per-turn memory retrieval. A slow "
                        "vector store / embedding endpoint degrades to a "
                        "memory-less turn instead of stalling the first "
                        "token. 0 disables the bound."
                    ),
                    default=10.0,
                    min_value=0,
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "stateless": self._stateless,
            "retrieval_timeout_s": self._retrieval_timeout_s,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "stateless" in config:
            self._stateless = bool(config["stateless"])
        if "retrieval_timeout_s" in config:
            try:
                self._retrieval_timeout_s = max(0.0, float(config["retrieval_timeout_s"]))
            except (TypeError, ValueError):
                pass

    def should_bypass(self, state: PipelineState) -> bool:
        return self._stateless

    async def _retrieve_memory(self, query: str, state: PipelineState) -> List[Any]:
        """Run the legacy retriever and the provider retrieval CONCURRENTLY,
        bounded by ``retrieval_timeout_s``.

        TTFT program (2026-07-12 audit, finding B1): both paths are
        independent reads that used to run back-to-back in front of the
        first API call. On timeout the turn proceeds WITHOUT memory —
        a degraded answer beats a stalled first token; the event trail
        records the skip. Real retrieval errors still propagate exactly
        as before.
        """
        use_provider = self._provider is not None and bool(query)

        async def _both():
            if not use_provider:
                return await self._retriever.retrieve(query, state), None
            return await asyncio.gather(
                self._retriever.retrieve(query, state),
                self._provider.retrieve(RetrievalQuery(text=query)),
            )

        timeout = self._retrieval_timeout_s or None
        try:
            retrieved, provider_result = await asyncio.wait_for(_both(), timeout=timeout)
        except asyncio.TimeoutError:
            state.add_event(
                "context.retrieval_timeout",
                {"timeout_s": self._retrieval_timeout_s},
            )
            logger.warning(
                "context: memory retrieval exceeded %.1fs — proceeding without memory",
                self._retrieval_timeout_s,
            )
            return []

        chunks = list(retrieved)
        if provider_result is not None:
            seen_keys = {c.key for c in chunks}
            for c in provider_result.chunks:
                if c.key not in seen_keys:
                    chunks.append(c)
                    seen_keys.add(c.key)
            state.add_event(MemoryEvent.CONTEXT_BUILT.value, provider_result.to_event())
        return chunks

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Build context via strategy
        await self._strategy.build_context(state)

        # Retrieve memory — extract query from the last user message, not final_text
        # (final_text is only populated after Stage 9 Parse, not available here)
        query = ""
        for msg in reversed(state.messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        if isinstance(query, list):
            # Extract text from content blocks (could be multimodal)
            query = " ".join(
                b.get("text", "") for b in query if isinstance(b, dict) and b.get("type") == "text"
            )
        query = str(query)

        # Clear last turn's retrieved memory BEFORE this turn's retrieval
        # (audit C1). ``state.metadata`` is sticky, and the injection below
        # only WRITES these keys when chunks come back — so a retrieval
        # that times out or returns nothing would leave the previous
        # turn's situational memory presented as if it were current.
        if state.iteration == 0:
            state.metadata.pop("memory_context", None)
            state.metadata.pop("memory_pinned", None)

        # TTFT program (finding B2): retrieval results are only injected
        # into the prompt at iteration 0 (below) — later tool-loop
        # iterations re-paid the embedding + vector round-trips for
        # results that were thrown away. Skip retrieval entirely there.
        chunks: List[Any] = []
        if state.iteration == 0:
            chunks = await self._retrieve_memory(query, state)

        if chunks:
            # Deduplicate by key
            seen = {ref.get("key") for ref in state.memory_refs}
            for chunk in chunks:
                if chunk.key not in seen:
                    state.memory_refs.append(
                        {
                            "key": chunk.key,
                            "source": chunk.source,
                            "content_length": len(chunk.content),
                            "relevance": chunk.relevance_score,
                        }
                    )
                    seen.add(chunk.key)

            # Split pinned chunks (always-inject T1 surface) from the
            # rest (per-turn retrieval). The system prompt builder
            # (``MemoryContextBlock``) renders them as two distinct
            # sections so the agent can tell what's permanent from
            # what's situational.
            pinned_chunks = [
                c
                for c in chunks
                if c.source == "pinned" or (c.metadata or {}).get("layer") == "pinned"
            ]
            other_chunks = [c for c in chunks if c not in pinned_chunks]

            if state.messages and state.iteration == 0:
                if pinned_chunks:
                    # Pinned chunks usually carry pre-rendered prose;
                    # join with blank lines instead of the bullet
                    # form used for search results.
                    state.metadata["memory_pinned"] = "\n\n".join(c.content for c in pinned_chunks)
                if other_chunks:
                    memory_text = "\n".join(
                        f"- [{c.source}] {c.key}: {c.content}" for c in other_chunks
                    )
                    state.metadata["memory_context"] = memory_text

        # Proactive compaction: when the projected next-call context
        # (system + messages + tools) crosses 80% of the window, compact
        # so the Stage 4 token-budget guard's 95% safety net rarely has
        # to. Both stages use ``estimate_prompt_tokens`` so compaction
        # measurably lowers the same number the guard checks.
        #
        # TTFT program (finding B3): an LLM-backed compactor is a whole
        # second model round-trip that used to run synchronously in
        # front of the first token. Now: a finished background summary
        # is applied first (cheap list swap); if still over 80% but
        # under the 90% hard line, the summary is computed in the
        # BACKGROUND (overlapping this turn's generation) and applied
        # at the next turn's Stage 2. Past 90% — or for cheap non-LLM
        # compactors — compaction stays synchronous as the safety net.
        if await self._apply_bg_compaction(state):
            state.shared.pop("_prompt_tokens_memo", None)
        estimated_tokens = estimate_prompt_tokens(state)
        budget = state.context_window_budget
        if estimated_tokens > budget * 0.8:
            defer_to_background = (
                isinstance(self._compactor, LLMSummaryCompactor)
                and estimated_tokens <= budget * 0.9
            )
            if defer_to_background:
                self._schedule_bg_compaction(state)
            else:
                await run_compaction(
                    state, self._compactor, trigger="proactive", provider=self._provider
                )

        state.add_event(
            "context.built",
            {
                "message_count": len(state.messages),
                "memory_refs": len(state.memory_refs),
                "estimated_tokens": estimated_tokens,
            },
        )

        return input

    # ── background compaction (TTFT program, finding B3) ─────────────

    def _schedule_bg_compaction(self, state: PipelineState) -> None:
        """Kick off the LLM summary on a message SNAPSHOT, off the hot path.

        History is append-only between turns, so a summary computed over
        messages[0:N] stays applicable as long as that prefix survives;
        the apply step verifies it (length + tail identity) and discards
        the result if a synchronous guard compaction rewrote history in
        the meantime. At most one background run is in flight per stage.
        """
        if self._bg_compaction is not None:
            return
        snapshot_len = len(state.messages)
        if snapshot_len == 0:
            return
        shadow = _CompactionShadow(state)
        tail_id = id(state.messages[snapshot_len - 1])

        async def _run() -> _CompactionShadow:
            await self._compactor.compact(shadow)
            return shadow

        task = asyncio.create_task(_run())
        # Surface failures in the log instead of "exception never retrieved".
        task.add_done_callback(
            lambda t: t.cancelled()
            or t.exception() is None
            or logger.warning("background compaction failed: %s", t.exception())
        )
        self._bg_compaction = {"task": task, "len": snapshot_len, "tail_id": tail_id}
        state.add_event(
            "context.compaction_scheduled",
            {
                "compactor": str(getattr(self._compactor, "name", "") or type(self._compactor).__name__),
                "snapshot_messages": snapshot_len,
            },
        )

    async def _apply_bg_compaction(self, state: PipelineState) -> bool:
        """Swap in a finished background summary; True when history changed."""
        info = self._bg_compaction
        if info is None:
            return False
        task: asyncio.Task = info["task"]
        if not task.done():
            return False
        self._bg_compaction = None
        if task.cancelled() or task.exception() is not None:
            return False  # already logged by the done-callback
        shadow: _CompactionShadow = task.result()

        n = int(info["len"])
        msgs = state.messages
        if len(msgs) < n or n == 0 or id(msgs[n - 1]) != info["tail_id"]:
            # Prefix rewritten since the snapshot (e.g. the Stage 4 guard
            # compacted synchronously) — the summary no longer matches.
            return False
        if len(shadow.messages) >= n:
            return False  # compactor was a no-op (below its keep threshold)

        before = len(msgs)
        replaced = before - (len(shadow.messages) + (before - n))
        before_list = list(msgs)
        state.messages = list(shadow.messages) + msgs[n:]
        # Keep Stage-18's STM watermark valid across the background swap
        # (audit D3) — same contract as run_compaction's synchronous path.
        reconcile_recorded_index(before_list, list(state.messages), state.metadata)
        for event_type, data in shadow.events:
            state.add_event(event_type, data)
        compactor_name = str(
            getattr(self._compactor, "name", "") or type(self._compactor).__name__
        )
        state.add_event(
            "context.compacted",
            {
                "strategy": compactor_name,
                "trigger": "background",
                "messages_before": before,
                "messages_after": len(state.messages),
                "saved_tokens_estimate": 0,
            },
        )

        # Persist the snapshot — same contract as run_compaction().
        if (
            replaced > 0
            and self._provider is not None
            and not getattr(self._compactor, "persists_own_compaction", False)
            and hasattr(self._provider, "record_compaction")
        ):
            summary_head = ""
            if state.messages and isinstance(state.messages[0], dict):
                head_content = state.messages[0].get("content", "")
                summary_head = head_content if isinstance(head_content, str) else ""
            try:
                await self._provider.record_compaction(
                    summary_head,
                    replaced_count=replaced,
                    strategy=compactor_name,
                    saved_tokens=0,
                    session_id=getattr(state, "session_id", "") or "",
                    trigger="background",
                )
            except Exception as exc:  # noqa: BLE001 — best effort
                state.add_event(
                    "context.compaction_record_failed",
                    {"compactor": compactor_name, "error": str(exc)},
                )
        return True
