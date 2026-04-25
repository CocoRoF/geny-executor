"""Stage 15: HITL — real implementation (S9b.3).

If ``state.shared['hitl_request']`` is populated (by host code, by
Stage 11 Tool Review, or by an earlier loop iteration), this stage
hands the request to the configured :class:`Requester` and waits for
an :class:`HITLDecision`. The wait is optionally bounded by the
:class:`TimeoutPolicy`'s ``timeout_seconds``; on timeout the policy
decides the verdict.

The decision is appended to ``state.shared['hitl_history']`` (audit
trail) and exposed at ``state.shared['hitl_last_decision']``. The
request key is consumed (set to ``None``) so the gate doesn't
re-fire on subsequent iterations.

When the decision is :attr:`HITLDecision.REJECT` or
:attr:`HITLDecision.CANCEL`, the stage marks the loop with
``state.loop_decision = "escalate"`` (cancel) or ``"complete"``
(reject) and sets ``state.completion_signal`` so downstream
observers see *why* the run terminated.

Pipelines that don't populate the request key see no behaviour
change — :meth:`should_bypass` returns True so the stage is a no-op
unless a host opts in.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl.artifact.default.requesters import (
    CallbackRequester,
    NullRequester,
)
from geny_executor.stages.s15_hitl.artifact.default.timeouts import (
    AutoApproveTimeout,
    AutoRejectTimeout,
    IndefiniteTimeout,
)
from geny_executor.stages.s15_hitl.interface import (
    HITL_HISTORY_KEY,
    HITL_LAST_DECISION_KEY,
    HITL_REQUEST_KEY,
    Requester,
    TimeoutPolicy,
)
from geny_executor.stages.s15_hitl.types import (
    HITLDecision,
    HITLEntry,
    HITLRequest,
    coerce_request,
)

logger = logging.getLogger(__name__)


class HITLStage(Stage[Any, Any]):
    """Stage 15: Human-in-the-loop gate.

    Two strategy slots:

    * ``requester`` — how decisions are sourced (default
      :class:`NullRequester` always-approve, so existing pipelines
      see no behaviour change).
    * ``timeout`` — what verdict applies when the requester takes too
      long (default :class:`IndefiniteTimeout`).
    """

    def __init__(
        self,
        requester: Optional[Requester] = None,
        timeout: Optional[TimeoutPolicy] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "requester": StrategySlot(
                name="requester",
                strategy=requester or NullRequester(),
                registry={
                    "null": NullRequester,
                    "callback": CallbackRequester,
                },
                description="Resolves a HITL request into a decision",
            ),
            "timeout": StrategySlot(
                name="timeout",
                strategy=timeout or IndefiniteTimeout(),
                registry={
                    "indefinite": IndefiniteTimeout,
                    "auto_approve": AutoApproveTimeout,
                    "auto_reject": AutoRejectTimeout,
                },
                description="What verdict applies when the requester times out",
            ),
        }

    @property
    def name(self) -> str:
        return "hitl"

    @property
    def order(self) -> int:
        return 15

    @property
    def category(self) -> str:
        return "gate"

    @property
    def _requester(self) -> Requester:
        return self._slots["requester"].strategy  # type: ignore[return-value]

    @property
    def _timeout(self) -> TimeoutPolicy:
        return self._slots["timeout"].strategy  # type: ignore[return-value]

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def should_bypass(self, state: PipelineState) -> bool:
        # No pending request → nothing to gate on.
        return state.shared.get(HITL_REQUEST_KEY) is None

    async def execute(self, input: Any, state: PipelineState) -> Any:
        request = coerce_request(state.shared.get(HITL_REQUEST_KEY))
        if request is None:
            return input

        # Consume the request key up front so we don't loop on it if
        # the requester fails fast.
        state.shared[HITL_REQUEST_KEY] = None
        state.add_event("hitl.request", request.to_dict())

        decision = await self._await_decision(request, state)
        state.shared[HITL_LAST_DECISION_KEY] = decision.value

        history: List[Any] = state.shared.setdefault(HITL_HISTORY_KEY, [])
        entry = HITLEntry(request=request, decision=decision)
        history.append(entry.to_dict())

        state.add_event(
            "hitl.decision",
            {"token": request.token, "decision": decision.value},
        )

        if decision == HITLDecision.REJECT:
            state.loop_decision = "complete"
            state.completion_signal = "HITL_REJECTED"
            state.completion_detail = request.reason or "human rejected"
        elif decision == HITLDecision.CANCEL:
            state.loop_decision = "escalate"
            state.completion_signal = "HITL_CANCELLED"
            state.completion_detail = request.reason or "human cancelled"

        return input

    async def _await_decision(self, request: HITLRequest, state: PipelineState) -> HITLDecision:
        timeout = self._timeout.timeout_seconds
        try:
            if timeout is None:
                decision = await self._requester.request(request, state)
            else:
                decision = await asyncio.wait_for(
                    self._requester.request(request, state), timeout=timeout
                )
        except asyncio.TimeoutError:
            verdict = self._timeout.on_timeout(request, state)
            state.add_event(
                "hitl.timeout",
                {
                    "token": request.token,
                    "timeout_seconds": timeout,
                    "verdict": verdict.value,
                },
            )
            return verdict
        except Exception as exc:  # noqa: BLE001 — never block the loop on requester bugs
            logger.warning(
                "HITL requester %s raised %s; cancelling request",
                self._requester.name,
                exc,
            )
            state.add_event(
                "hitl.requester_error",
                {"requester": self._requester.name, "error": str(exc)},
            )
            return HITLDecision.CANCEL

        if decision is None:
            verdict = self._timeout.on_timeout(request, state)
            state.add_event(
                "hitl.no_decision",
                {"token": request.token, "verdict": verdict.value},
            )
            return verdict
        return decision
