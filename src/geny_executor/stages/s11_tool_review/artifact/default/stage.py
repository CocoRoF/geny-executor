"""Stage 11: Tool Review — chain implementation (S9b.1).

Sub-phase 9a shipped this stage as a pass-through scaffold.
Sub-phase 9b (S9b.1) replaces the scaffold body with a real
:class:`SlotChain` of :class:`Reviewer` strategies. The default
chain is::

    Schema → Sensitive → Destructive → Network → Size

Each reviewer runs in declared order and is failure-isolated by the
stage; an exception only sidelines that reviewer for the turn (the
flag list still survives). The merged flag list lives at
``state.shared['tool_review_flags']`` so downstream stages
(typically Stage 14 Evaluate) can act on it.

The stage clears the running flag list at the start of every
``execute`` call so flags don't bleed across loop iterations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from geny_executor.core.slot import SlotChain
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review.artifact.default.reviewers import (
    DestructiveResultReviewer,
    NetworkAuditReviewer,
    SchemaReviewer,
    SensitivePatternReviewer,
    SizeReviewer,
)
from geny_executor.stages.s11_tool_review.interface import (
    Reviewer,
    append_flags,
    reset_flags,
)

logger = logging.getLogger(__name__)


_DEFAULT_CHAIN: List[Reviewer] = [
    SchemaReviewer(),
    SensitivePatternReviewer(),
    DestructiveResultReviewer(),
    NetworkAuditReviewer(),
    SizeReviewer(),
]


class ToolReviewStage(Stage[Any, Any]):
    """Stage 11: Tool Review.

    Walks an ordered chain of :class:`Reviewer` strategies and
    accumulates :class:`ToolReviewFlag` records on
    ``state.shared['tool_review_flags']``.
    """

    def __init__(self, reviewers: List[Reviewer] | None = None):
        chain_items: List[Reviewer] = (
            list(reviewers) if reviewers is not None else list(_DEFAULT_CHAIN)
        )
        self._chain = SlotChain(
            name="reviewers",
            items=chain_items,
            registry={
                "schema": SchemaReviewer,
                "sensitive": SensitivePatternReviewer,
                "destructive": DestructiveResultReviewer,
                "network": NetworkAuditReviewer,
                "size": SizeReviewer,
            },
            description="Ordered chain of tool-call reviewers",
        )

    @property
    def name(self) -> str:
        return "tool_review"

    @property
    def order(self) -> int:
        return 11

    @property
    def category(self) -> str:
        return "review"

    def get_strategy_slots(self) -> Dict[str, Any]:
        return {}

    def get_strategy_chains(self) -> Dict[str, SlotChain]:
        return {"reviewers": self._chain}

    def should_bypass(self, state: PipelineState) -> bool:
        # No tool calls and no tool results → nothing to review.
        return not state.pending_tool_calls and not state.tool_results

    async def execute(self, input: Any, state: PipelineState) -> Any:
        reset_flags(state)
        tool_calls = list(state.pending_tool_calls)
        tool_results = list(state.tool_results)

        for reviewer in self._chain.items:
            try:
                flags = await reviewer.review(tool_calls, tool_results, state)
            except Exception as exc:  # noqa: BLE001 — chain-wide failure isolation
                logger.warning(
                    "Tool reviewer %s raised %s; skipping its flags this turn",
                    reviewer.name,
                    exc,
                )
                state.add_event(
                    "tool_review.reviewer_error",
                    {"reviewer": reviewer.name, "error": str(exc)},
                )
                continue
            if not flags:
                continue
            append_flags(state, flags)
            for flag in flags:
                state.add_event("tool_review.flag", flag.to_dict())

        flag_count = len(state.shared.get("tool_review_flags") or [])
        state.add_event(
            "tool_review.completed",
            {
                "reviewers": [r.name for r in self._chain.items],
                "flags": flag_count,
                "tool_calls": len(tool_calls),
                "tool_results": len(tool_results),
            },
        )
        return input
