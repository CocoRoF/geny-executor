"""Stage 15: HITL — interface definitions (S9b.3).

A :class:`Requester` resolves a single :class:`HITLRequest` into an
:class:`HITLDecision`. Hosts implement this for their UI/CLI/Slack
channel — the executor is intentionally agnostic about how approval
is collected.

A :class:`TimeoutPolicy` decides what to do when the requester does
not return inside the configured wall-clock budget. Built-in policies:
indefinite (no timeout), auto-approve, auto-reject.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl.types import HITLDecision, HITLRequest


# State keys used by Stage 15.
HITL_REQUEST_KEY = "hitl_request"
HITL_HISTORY_KEY = "hitl_history"
HITL_LAST_DECISION_KEY = "hitl_last_decision"


class Requester(Strategy):
    """Resolve a single HITL request into a decision."""

    @abstractmethod
    async def request(self, request: HITLRequest, state: PipelineState) -> Optional[HITLDecision]:
        """Return the human's decision, or None if no decision yet."""
        ...


class TimeoutPolicy(Strategy):
    """Decide what verdict applies when the requester times out."""

    @abstractmethod
    def on_timeout(self, request: HITLRequest, state: PipelineState) -> HITLDecision: ...

    @property
    @abstractmethod
    def timeout_seconds(self) -> Optional[float]:
        """``None`` means "wait forever" — disables the timeout entirely."""
        ...


__all__ = [
    "HITL_HISTORY_KEY",
    "HITL_LAST_DECISION_KEY",
    "HITL_REQUEST_KEY",
    "Requester",
    "TimeoutPolicy",
]
