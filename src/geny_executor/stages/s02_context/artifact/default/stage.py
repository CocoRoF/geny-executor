"""Stage 2: Context — concrete stage implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import (
    Layer,
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
    SlidingWindowCompactor,
    SummaryCompactor,
    TruncateCompactor,
)
from geny_executor.stages.s02_context.artifact.default.retrievers import (
    NullRetriever,
    StaticRetriever,
)


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
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {"stateless": self._stateless}

    def update_config(self, config: Dict[str, Any]) -> None:
        if "stateless" in config:
            self._stateless = bool(config["stateless"])

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

        chunks = list(await self._retriever.retrieve(str(query), state))

        # Provider-driven retrieval (Phase 1+). Runs in addition to the
        # legacy retriever so users mid-migration keep both paths working.
        if self._provider is not None and query:
            rq = RetrievalQuery(text=str(query))
            result = await self._provider.retrieve(rq)
            seen_keys = {c.key for c in chunks}
            for c in result.chunks:
                if c.key not in seen_keys:
                    chunks.append(c)
                    seen_keys.add(c.key)
            state.add_event(MemoryEvent.CONTEXT_BUILT.value, result.to_event())

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

            # Inject memory into system prompt or as user message
            memory_text = "\n".join(f"- [{c.source}] {c.key}: {c.content}" for c in chunks)
            if state.messages and state.iteration == 0:
                # First iteration: inject as context
                state.metadata["memory_context"] = memory_text

        # Compact if needed (rough estimate: 4 chars per token)
        estimated_tokens = sum(len(str(m.get("content", ""))) // 4 for m in state.messages)
        if estimated_tokens > state.context_window_budget * 0.8:
            await self._compactor.compact(state)
            state.add_event(
                "context.compacted",
                {
                    "strategy": type(self._compactor).__name__,
                },
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
