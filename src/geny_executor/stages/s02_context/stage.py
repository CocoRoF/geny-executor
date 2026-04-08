"""Stage 2: Context — collects history, memory, references."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.strategies import ContextStrategy, SimpleLoadStrategy
from geny_executor.stages.s02_context.compactors import HistoryCompactor, TruncateCompactor
from geny_executor.stages.s02_context.retrievers import MemoryRetriever, NullRetriever


class ContextStage(Stage[Any, Any]):
    """Stage 2: Context.

    Dual abstraction:
      - Level 2 context_strategy: how to collect context
      - Level 2 compactor: how to compress when over budget
      - Level 2 retriever: how to fetch memory
    """

    def __init__(
        self,
        strategy: Optional[ContextStrategy] = None,
        compactor: Optional[HistoryCompactor] = None,
        retriever: Optional[MemoryRetriever] = None,
        *,
        stateless: bool = False,
    ):
        self._strategy = strategy or SimpleLoadStrategy()
        self._compactor = compactor or TruncateCompactor()
        self._retriever = retriever or NullRetriever()
        self._stateless = stateless

    @property
    def name(self) -> str:
        return "context"

    @property
    def order(self) -> int:
        return 2

    @property
    def category(self) -> str:
        return "ingress"

    def should_bypass(self, state: PipelineState) -> bool:
        return self._stateless

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

        chunks = await self._retriever.retrieve(str(query), state)
        if chunks:
            # Deduplicate by key
            seen = {ref.get("key") for ref in state.memory_refs}
            for chunk in chunks:
                if chunk.key not in seen:
                    state.memory_refs.append({
                        "key": chunk.key,
                        "source": chunk.source,
                        "content_length": len(chunk.content),
                        "relevance": chunk.relevance_score,
                    })
                    seen.add(chunk.key)

            # Inject memory into system prompt or as user message
            memory_text = "\n".join(
                f"- [{c.source}] {c.key}: {c.content}" for c in chunks
            )
            if state.messages and state.iteration == 0:
                # First iteration: inject as context
                state.metadata["memory_context"] = memory_text

        # Compact if needed (rough estimate: 4 chars per token)
        estimated_tokens = sum(
            len(str(m.get("content", ""))) // 4 for m in state.messages
        )
        if estimated_tokens > state.context_window_budget * 0.8:
            await self._compactor.compact(state)
            state.add_event("context.compacted", {
                "strategy": type(self._compactor).__name__,
            })

        state.add_event("context.built", {
            "message_count": len(state.messages),
            "memory_refs": len(state.memory_refs),
            "estimated_tokens": estimated_tokens,
        })

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="strategy",
                current_impl=type(self._strategy).__name__,
                available_impls=[
                    "SimpleLoadStrategy",
                    "HybridStrategy",
                    "ProgressiveDisclosureStrategy",
                ],
            ),
            StrategyInfo(
                slot_name="compactor",
                current_impl=type(self._compactor).__name__,
                available_impls=[
                    "TruncateCompactor",
                    "SummaryCompactor",
                    "SlidingWindowCompactor",
                ],
            ),
            StrategyInfo(
                slot_name="retriever",
                current_impl=type(self._retriever).__name__,
                available_impls=[
                    "NullRetriever",
                    "StaticRetriever",
                ],
            ),
        ]
