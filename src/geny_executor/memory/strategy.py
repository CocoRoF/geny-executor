"""GenyMemoryStrategy — Geny-compatible memory update after execution.

Implements the MemoryUpdateStrategy interface (S15 Memory) using Geny's
SessionMemoryManager. Replicates logic from:

1. TranscriptRecordNode — records execution to short-term memory
2. MemoryReflectNode — LLM-powered insight extraction + structured note saving

The reflection step is optional and requires an ``llm_reflect`` callable.
When not provided, only transcript recording occurs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_memory.interface import MemoryUpdateStrategy

logger = logging.getLogger(__name__)


class GenyMemoryStrategy(MemoryUpdateStrategy):
    """Geny-compatible memory update strategy.

    Args:
        memory_manager: Geny's SessionMemoryManager (or duck-typed equivalent).
        enable_reflection: Whether to run LLM-powered insight extraction.
        llm_reflect: Async callable for LLM reflection.
            Signature: ``async (input_text: str, output_text: str) -> List[InsightDict]``
            Each InsightDict has: title, content, category, tags, importance.
            When None and enable_reflection=True, only a ``needs_reflection``
            flag is set in state.metadata.
        max_insights: Maximum insights to extract per execution.
        auto_promote_importance: Importance levels that trigger auto-promotion
            to curated knowledge (e.g. {"high", "critical"}).
        curated_knowledge_manager: Optional manager for auto-promotion.
    """

    def __init__(
        self,
        memory_manager: Any,
        *,
        enable_reflection: bool = True,
        llm_reflect: Optional[
            Callable[[str, str], Awaitable[List[Dict[str, Any]]]]
        ] = None,
        max_insights: int = 3,
        auto_promote_importance: Optional[set] = None,
        curated_knowledge_manager: Any = None,
    ):
        self._mgr = memory_manager
        self._enable_reflection = enable_reflection
        self._llm_reflect = llm_reflect
        self._max_insights = max_insights
        self._auto_promote = auto_promote_importance or {"high", "critical"}
        self._curated = curated_knowledge_manager

    @property
    def name(self) -> str:
        return "geny_memory"

    @property
    def description(self) -> str:
        return "Geny-compatible memory update with transcript recording and optional LLM reflection"

    async def update(self, state: PipelineState) -> None:
        if not self._mgr:
            return

        # Step 1: Record conversation to short-term memory (transcript)
        self._record_transcript(state)

        # Step 2: Record execution result to long-term memory
        self._record_execution_result(state)

        # Step 3: LLM reflection (optional)
        if self._enable_reflection:
            await self._reflect(state)

    # ── Step 1: Transcript Recording ─────────────────────────────────

    def _record_transcript(self, state: PipelineState) -> None:
        """Record the latest user+assistant messages to short-term memory.

        Records BOTH the user input and assistant output to STM,
        matching Geny's TranscriptRecordNode + PostModel behavior.
        Only records messages from the current turn (not re-recording old ones).
        """
        try:
            record = getattr(self._mgr, "record_message", None)
            if record is None:
                return

            # Determine which messages are "new" in this turn.
            # On first iteration, record the user input.
            # On every iteration, record the latest assistant output.
            recorded_count = state.metadata.get("_stm_recorded_count", 0)
            new_messages = state.messages[recorded_count:]

            for msg in new_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role not in ("user", "assistant"):
                    continue

                # Extract text from content blocks if needed
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = "\n".join(text_parts)

                if content:
                    record(role, content[:5000])

            # Track how many messages we've recorded to avoid re-recording
            state.metadata["_stm_recorded_count"] = len(state.messages)

        except Exception:
            logger.debug("geny_strategy: transcript recording failed", exc_info=True)

    # ── Step 2: Execution Result Recording ───────────────────────────

    def _record_execution_result(self, state: PipelineState) -> None:
        """Record execution summary to long-term memory (dated entry)."""
        if not state.final_text or state.loop_decision == "error":
            return

        try:
            remember_dated = getattr(self._mgr, "remember_dated", None)
            if remember_dated is None:
                return

            # Only record for non-trivial executions (multi-turn or with tools)
            if state.iteration < 1 and not state.tool_results:
                return

            # Build a concise execution record
            input_text = ""
            for msg in state.messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        input_text = content[:300]
                    break

            summary = (
                f"**Input:** {input_text}\n"
                f"**Output:** {state.final_text[:800]}\n"
                f"**Iterations:** {state.iteration} | "
                f"**Cost:** ${state.total_cost_usd:.4f}"
            )

            remember_dated(summary)

        except Exception:
            logger.debug("geny_strategy: execution recording failed", exc_info=True)

    # ── Step 3: LLM Reflection ───────────────────────────────────────

    async def _reflect(self, state: PipelineState) -> None:
        """Extract reusable insights from execution via LLM."""
        if self._llm_reflect is None:
            # No LLM callable provided — just set a flag
            state.metadata["needs_reflection"] = True
            state.add_event("memory.reflection_queued", {
                "message_count": len(state.messages),
                "iteration": state.iteration,
            })
            return

        # Extract input and output
        input_text = ""
        for msg in state.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    input_text = content
                break

        output_text = state.final_text
        if not input_text.strip() or not output_text.strip():
            return

        try:
            insights = await self._llm_reflect(
                input_text[:2000], output_text[:3000]
            )
            if not insights:
                return

            write_note = getattr(self._mgr, "write_note", None)
            if write_note is None:
                return

            saved = 0
            for item in insights[: self._max_insights]:
                try:
                    filename = write_note(
                        title=item.get("title", "Insight"),
                        content=item.get("content", ""),
                        category=item.get("category", "insights"),
                        tags=item.get("tags", []),
                        importance=item.get("importance", "medium"),
                        source="reflection",
                    )
                    if filename:
                        saved += 1

                        # Auto-promote high-importance insights
                        importance = item.get("importance", "medium")
                        if importance in self._auto_promote and self._curated:
                            try:
                                self._curated.write_note(
                                    title=item.get("title", "Insight"),
                                    content=item.get("content", ""),
                                    category=item.get("category", "insights"),
                                    tags=item.get("tags", []) + ["auto-promoted"],
                                    importance=importance,
                                    source="promoted",
                                )
                            except Exception:
                                pass

                except Exception:
                    logger.debug(
                        "geny_strategy: failed to save insight '%s'",
                        item.get("title", "?"),
                        exc_info=True,
                    )

            if saved:
                state.add_event("memory.insights_saved", {"count": saved})
                logger.info("geny_strategy: saved %d insights", saved)

        except Exception:
            logger.warning("geny_strategy: LLM reflection failed", exc_info=True)
