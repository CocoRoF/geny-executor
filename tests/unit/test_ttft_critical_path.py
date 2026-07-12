"""TTFT program (2.50.0) — pre-call critical-path fixes, group B.

B1  retrieval parallelized (retriever ∥ provider; composite layers
    gathered; query-embedding LRU) and bounded by retrieval_timeout_s.
B2  retrieval skipped on tool-loop iterations ≥ 1 — results were only
    ever injected at iteration 0.
B3  LLM compaction runs in the BACKGROUND between turns instead of as
    a second synchronous model call in front of the first token.
B4  estimate_prompt_tokens memoized per state fingerprint.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens
from geny_executor.memory.embedding.client import QueryEmbedLRU
from geny_executor.stages.s02_context import ContextStage
from geny_executor.stages.s02_context.artifact.default.compactors import (
    LLMSummaryCompactor,
)
from geny_executor.stages.s02_context.types import MemoryChunk


class _CountingRetriever:
    """Retriever double recording call count and overlap."""

    name = "counting"
    description = "test double"

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return [
                MemoryChunk(key="r1", content="from retriever", source="test", relevance_score=1.0)
            ]
        finally:
            self.active -= 1


def _user_state(iteration: int = 0) -> PipelineState:
    state = PipelineState()
    state.messages.append({"role": "user", "content": "질문입니다"})
    state.iteration = iteration
    return state


class TestIterationGate:
    @pytest.mark.asyncio
    async def test_retrieval_runs_at_iteration_zero(self):
        retriever = _CountingRetriever()
        stage = ContextStage(retriever=retriever)
        await stage.execute("in", _user_state(iteration=0))
        assert retriever.calls == 1

    @pytest.mark.asyncio
    async def test_retrieval_skipped_on_later_iterations(self):
        """Tool-loop iterations re-paid embedding+vector round-trips for
        results that were discarded (injection is iteration-0-only)."""
        retriever = _CountingRetriever()
        stage = ContextStage(retriever=retriever)
        await stage.execute("in", _user_state(iteration=1))
        await stage.execute("in", _user_state(iteration=3))
        assert retriever.calls == 0


class TestRetrievalParallelismAndTimeout:
    @pytest.mark.asyncio
    async def test_retriever_and_provider_overlap(self):
        """Legacy retriever and provider retrieval must run concurrently,
        not back-to-back: with two 50ms paths the pair finishes well
        under 2×50ms."""

        class _SlowProvider:
            async def retrieve(self, rq):
                await asyncio.sleep(0.05)

                class _R:
                    chunks = [
                        MemoryChunk(key="p1", content="from provider", source="test",
                                    relevance_score=0.5)
                    ]

                    @staticmethod
                    def to_event():
                        return {"chunks": 1}

                return _R()

        retriever = _CountingRetriever(delay=0.05)
        stage = ContextStage(retriever=retriever, provider=_SlowProvider())

        state = _user_state()
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        await stage.execute("in", state)
        elapsed = loop.time() - t0

        assert elapsed < 0.09, f"paths ran serially ({elapsed*1000:.0f}ms)"
        keys = {r["key"] for r in state.memory_refs}
        assert {"r1", "p1"} <= keys

    @pytest.mark.asyncio
    async def test_timeout_degrades_to_memoryless_turn(self):
        class _HangingRetriever(_CountingRetriever):
            def __init__(self):
                super().__init__(delay=30.0)

        stage = ContextStage(retriever=_HangingRetriever(), retrieval_timeout_s=0.05)
        state = _user_state()
        await asyncio.wait_for(stage.execute("in", state), timeout=5)
        assert state.memory_refs == []
        assert any(e["type"] == "context.retrieval_timeout" for e in state.events)


class TestQueryEmbedLRU:
    def test_hit_miss_and_eviction(self):
        lru = QueryEmbedLRU(maxsize=2)
        assert lru.get("a") is None
        lru.put("a", [1.0]), lru.put("b", [2.0])
        assert lru.get("a") == [1.0]  # refreshes 'a'
        lru.put("c", [3.0])  # evicts LRU 'b'
        assert lru.get("b") is None
        assert lru.get("a") == [1.0] and lru.get("c") == [3.0]


class TestEstimateMemo:
    def test_same_state_returns_memo_and_mutation_invalidates(self):
        state = PipelineState()
        state.messages.append({"role": "user", "content": "x" * 400})
        first = estimate_prompt_tokens(state)
        memo = state.shared.get("_prompt_tokens_memo")
        assert memo is not None and memo[1] == first
        assert estimate_prompt_tokens(state) == first

        state.messages.append({"role": "assistant", "content": "y" * 400})
        second = estimate_prompt_tokens(state)
        assert second > first


class _InstantLLMClient:
    provider = "fake"

    async def create_message(self, *, model_config: Any, messages, purpose: str = ""):
        class _Resp:
            text = "요약: 이전 대화 내용."

        return _Resp()


def _llm_compactor() -> LLMSummaryCompactor:
    from geny_executor.core.config import ModelConfig

    return LLMSummaryCompactor(
        keep_recent=2,
        resolve_cfg=lambda s: ModelConfig(model="fake", max_tokens=256),
        has_override=lambda: True,
    )


class TestBackgroundCompaction:
    @pytest.mark.asyncio
    async def test_llm_compaction_deferred_then_applied_next_turn(self):
        """80–90% zone: turn N schedules the summary in the background
        (no second model call in front of the first token); turn N+1
        applies the finished swap."""
        stage = ContextStage(compactor=_llm_compactor())
        state = PipelineState()
        state.llm_client = _InstantLLMClient()
        state.context_window_budget = 1000
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            state.messages.append({"role": role, "content": f"글" * 280})

        await stage.execute("in", state)  # ~85% → schedule, not block
        assert any(e["type"] == "context.compaction_scheduled" for e in state.events)
        assert len(state.messages) == 12  # untouched this turn

        await asyncio.sleep(0)  # let the background task run
        state.iteration = 0
        await stage.execute("in", state)  # next turn applies the swap

        compacted = [e for e in state.events if e["type"] == "context.compacted"]
        assert compacted and compacted[-1]["data"]["trigger"] == "background"
        assert len(state.messages) < 12
        head = state.messages[0]["content"]
        assert "요약" in head

    @pytest.mark.asyncio
    async def test_danger_zone_still_compacts_synchronously(self):
        """Past 90% the safety net kicks in immediately."""
        stage = ContextStage(compactor=_llm_compactor())
        state = PipelineState()
        state.llm_client = _InstantLLMClient()
        state.context_window_budget = 1000
        for i in range(12):
            state.messages.append({"role": "user", "content": "글" * 400})

        await stage.execute("in", state)
        assert len(state.messages) < 12  # compacted THIS turn
        assert not any(e["type"] == "context.compaction_scheduled" for e in state.events)

    @pytest.mark.asyncio
    async def test_stale_background_result_discarded_when_prefix_rewritten(self):
        stage = ContextStage(compactor=_llm_compactor())
        state = PipelineState()
        state.llm_client = _InstantLLMClient()
        state.context_window_budget = 1000
        for i in range(12):
            state.messages.append({"role": "user", "content": "글" * 280})

        await stage.execute("in", state)
        assert stage._bg_compaction is not None
        await asyncio.sleep(0)

        # Simulate a guard compaction rewriting history before next turn.
        state.messages = [{"role": "user", "content": "전혀 다른 히스토리"}]
        await stage.execute("in", state)

        assert not any(
            e["type"] == "context.compacted" and e["data"].get("trigger") == "background"
            for e in state.events
        )
