"""GenyMemoryStrategy — Geny-compatible memory update after execution.

Implements the MemoryUpdateStrategy interface (S15 Memory) using Geny's
SessionMemoryManager. Replicates logic from:

1. TranscriptRecordNode — records execution to short-term memory
2. MemoryReflectNode — LLM-powered insight extraction + structured note saving

Reflection has three resolution modes:

1. ``llm_reflect`` callback (legacy Geny plumbing) — used when provided.
2. Native path via ``ReflectionResolver`` + ``state.llm_client`` —
   runs when no callback is set AND the hosting stage has an explicit
   model override AND ``state.llm_client`` is non-None.
3. Deferred flag — sets ``state.metadata['needs_reflection'] = True``
   and emits ``memory.reflection_queued`` (pre-cycle behavior).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s18_memory.interface import MemoryUpdateStrategy

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResolver:
    """Glue between the strategy and the hosting Memory stage.

    Lets the strategy obtain the stage's :class:`ModelConfig` and the
    shared :class:`BaseClient` without taking a hard dependency on the
    stage class.

    Attributes:
        resolve_cfg: Callable taking ``state`` and returning the effective
            :class:`ModelConfig`. Typically
            ``lambda s: stage.resolve_model_config(s)``.
        has_override: Callable returning True iff the stage has an explicit
            ``model_override`` set. Used to gate the native path.
        client_getter: Callable taking ``state`` and returning the
            :class:`BaseClient`. Defaults to ``state.llm_client``.
    """

    resolve_cfg: Callable[[PipelineState], Any]
    has_override: Callable[[], bool]
    client_getter: Callable[[PipelineState], Optional[Any]] = field(
        default=lambda s: getattr(s, "llm_client", None)
    )


class GenyMemoryStrategy(MemoryUpdateStrategy):
    """Geny-compatible memory update strategy.

    Args:
        memory_manager: Geny's SessionMemoryManager (or duck-typed equivalent).
        enable_reflection: Whether to run LLM-powered insight extraction.
        llm_reflect: Async callable for LLM reflection (legacy path).
            Signature: ``async (input_text: str, output_text: str) -> List[InsightDict]``
            Each InsightDict has: title, content, category, tags, importance.
        max_insights: Maximum insights to extract per execution.
        auto_promote_importance: Importance levels that trigger auto-promotion
            to curated knowledge (e.g. {"high", "critical"}).
        curated_knowledge_manager: Optional manager for auto-promotion.
        resolver: :class:`ReflectionResolver` enabling the native LLM path
            when no callback is provided.
    """

    #: Importance ranking — lower index = higher priority. Used by the
    #: ``min_insight_importance`` gate so callers can write
    #: ``min_insight_importance="high"`` and reject low/medium without
    #: writing comparison code in the host.
    _IMPORTANCE_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(
        self,
        memory_manager: Any,
        *,
        enable_reflection: bool = True,
        llm_reflect: Optional[Callable[[str, str], Awaitable[List[Dict[str, Any]]]]] = None,
        max_insights: int = 3,
        auto_promote_importance: Optional[set] = None,
        curated_knowledge_manager: Any = None,
        resolver: Optional[ReflectionResolver] = None,
        min_insight_importance: str = "high",
    ):
        self._mgr = memory_manager
        self._enable_reflection = enable_reflection
        self._llm_reflect = llm_reflect
        self._max_insights = max_insights
        self._auto_promote = auto_promote_importance or {"high", "critical"}
        self._curated = curated_knowledge_manager
        self._resolver = resolver
        # Quality gate. Default ``high`` — only high/critical insights
        # land on disk. Operators wanting the historical permissive
        # behaviour pass ``min_insight_importance="low"``. The gate
        # rejects below-threshold reflections silently — the LLM can
        # still emit them; they just don't pollute insights/.
        self._min_insight_rank = self._IMPORTANCE_RANK.get(
            min_insight_importance.lower(),
            1,
        )

    @property
    def name(self) -> str:
        return "geny_memory"

    @property
    def description(self) -> str:
        return "Geny-compatible memory update with transcript recording and optional LLM reflection"

    async def update(self, state: PipelineState) -> None:
        if not self._mgr:
            return

        self._record_transcript(state)
        self._record_execution_result(state)

        if self._enable_reflection:
            await self._reflect(state)

    # ── Step 1: Transcript Recording ─────────────────────────────────

    def _record_transcript(self, state: PipelineState) -> None:
        try:
            record = getattr(self._mgr, "record_message", None)
            if record is None:
                return

            recorded_count = state.metadata.get("_stm_recorded_count", 0)
            new_messages = state.messages[recorded_count:]

            for msg in new_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role not in ("user", "assistant"):
                    continue

                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    content = "\n".join(text_parts)

                if content:
                    record(role, content[:5000])

            state.metadata["_stm_recorded_count"] = len(state.messages)

        except Exception:
            logger.debug("geny_strategy: transcript recording failed", exc_info=True)

    # ── Step 2: Execution Result Recording ───────────────────────────

    def _record_execution_result(self, state: PipelineState) -> None:
        if not state.final_text or state.loop_decision == "error":
            return

        try:
            remember_dated = getattr(self._mgr, "remember_dated", None)
            if remember_dated is None:
                return

            if state.iteration < 1 and not state.tool_results:
                return

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
        """Extract reusable insights from execution.

        Resolution order:
            1. If ``llm_reflect`` callback was provided, use it (legacy path).
            2. Else if a :class:`ReflectionResolver` was provided AND the
               hosting stage has an explicit override AND
               ``state.llm_client`` is available → run a native reflection
               call.
            3. Else set ``state.metadata['needs_reflection'] = True`` and
               emit ``memory.reflection_queued`` (pre-cycle behavior).
        """
        input_text = self._extract_user_input(state)
        output_text = state.final_text or ""

        if self._llm_reflect is not None:
            if not input_text.strip() or not output_text.strip():
                return
            try:
                insights = await self._llm_reflect(input_text[:2000], output_text[:3000])
            except Exception:
                logger.warning("geny_strategy: callback reflection failed", exc_info=True)
                return
            await self._save_insights(state, insights)
            return

        if self._resolver is None or not self._resolver.has_override():
            state.metadata["needs_reflection"] = True
            state.add_event(
                "memory.reflection_queued",
                {
                    "message_count": len(state.messages),
                    "iteration": state.iteration,
                },
            )
            return

        client = self._resolver.client_getter(state)
        if client is None:
            state.metadata["needs_reflection"] = True
            state.add_event("memory.reflection_queued", {"reason": "no_llm_client"})
            return

        if not input_text.strip() or not output_text.strip():
            return

        prompt = self._build_reflection_prompt(input_text, output_text)
        cfg = self._resolver.resolve_cfg(state)
        try:
            resp = await client.create_message(
                model_config=cfg,
                messages=[{"role": "user", "content": prompt}],
                purpose="s15.reflect",
            )
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            if not data.get("should_save"):
                state.add_event(
                    "memory.reflection.native",
                    {
                        "saved": 0,
                        "model": cfg.model,
                        "provider": getattr(client, "provider", ""),
                    },
                )
                return
            insights = data.get("learned") or []
        except Exception as exc:
            state.add_event(
                "memory.reflection.llm_failed",
                {"error": str(exc), "source": "native"},
            )
            return

        await self._save_insights(state, insights, source_label="reflection_native")
        state.add_event(
            "memory.reflection.native",
            {
                "saved": min(len(insights), self._max_insights),
                "model": cfg.model,
                "provider": getattr(client, "provider", ""),
            },
        )

    def _extract_user_input(self, state: PipelineState) -> str:
        for msg in state.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _build_reflection_prompt(self, input_text: str, output_text: str) -> str:
        # Reflection prompt — high bar by design. The host filters
        # the result by ``min_insight_importance`` (default "high")
        # so anything that scrapes through with importance="medium"
        # is dropped silently. The wording here aims to make the
        # LLM rarely emit anything at all — most turns produce no
        # genuine factual learning, and the strategy must reflect
        # that.
        return (
            "Analyze the following execution. Extract ONLY genuine, "
            "factual knowledge worth remembering across future sessions — "
            "concrete user preferences, project facts, decisions with "
            "reasoning, or domain knowledge that wasn't already obvious.\n\n"
            f"<input>\n{input_text[:2000]}\n</input>\n\n"
            f"<output>\n{output_text[:3000]}\n</output>\n\n"
            "REJECT (do not save) anything in these categories:\n"
            "  - Behavioral / communication patterns "
            "    (e.g. 'use friendly tone', 'greet warmly')\n"
            "  - Generic best practices applicable to any agent\n"
            "  - Restating what the user just said\n"
            "  - Per-turn tactics ('I confirmed before delegating')\n"
            "  - Anything already captured in entities/<counterpart>.md\n"
            "    (interaction stats — that's automatic)\n"
            "\n"
            "ACCEPT only when the turn produced one of these:\n"
            "  - A user-stated fact about themselves, their goals, "
            "    constraints, or environment\n"
            "  - A project decision with non-obvious rationale\n"
            "  - A non-trivial technical finding (a working approach, "
            "    a gotcha, a constraint)\n"
            "\n"
            "Importance scale (be conservative):\n"
            "  - critical: the agent will fail without this fact\n"
            "  - high:     the agent's quality drops noticeably without it\n"
            "  - medium:   nice to know — DO NOT EMIT, host drops these\n"
            "  - low:      drop\n"
            "\n"
            "Most turns produce zero insights. Empty output is correct.\n"
            "\n"
            "Respond with JSON only:\n"
            "{\n"
            '  "learned": [\n'
            "    {\n"
            '      "title": "concise title (3-10 words)",\n'
            '      "content": "what was learned (1-3 sentences)",\n'
            '      "category": "topics|insights|entities|projects",\n'
            '      "tags": ["tag1", "tag2"],\n'
            '      "importance": "high|critical"\n'
            "    }\n"
            "  ],\n"
            '  "should_save": true\n'
            "}\n\n"
            "When in doubt, return:\n"
            '{"learned": [], "should_save": false}'
        )

    async def _save_insights(
        self,
        state: PipelineState,
        insights: List[Dict[str, Any]],
        *,
        source_label: str = "reflection",
    ) -> None:
        if not insights:
            return
        write_note = getattr(self._mgr, "write_note", None)
        if write_note is None:
            return
        # Quality gate. Drop anything below ``min_insight_importance``
        # before slicing to ``max_insights`` — otherwise a noisy LLM
        # batch of medium-importance items would crowd out the
        # high-importance ones.
        gated: List[Dict[str, Any]] = []
        dropped_below_threshold = 0
        for item in insights:
            importance = (item.get("importance") or "medium").lower()
            rank = self._IMPORTANCE_RANK.get(importance, 99)
            if rank > self._min_insight_rank:
                dropped_below_threshold += 1
                continue
            gated.append(item)
        if dropped_below_threshold:
            logger.debug(
                "geny_strategy: dropped %d insights below importance gate",
                dropped_below_threshold,
            )
        if not gated:
            state.add_event(
                "memory.insights_gated",
                {"dropped": dropped_below_threshold, "threshold_rank": self._min_insight_rank},
            )
            return
        saved = 0
        for item in gated[: self._max_insights]:
            try:
                filename = write_note(
                    title=item.get("title", "Insight"),
                    content=item.get("content", ""),
                    category=item.get("category", "insights"),
                    tags=item.get("tags", []),
                    importance=item.get("importance", "medium"),
                    source=source_label,
                )
                if filename:
                    saved += 1
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
