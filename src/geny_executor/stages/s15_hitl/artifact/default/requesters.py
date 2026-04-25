"""Default HITL requesters for Stage 15 (S9b.3)."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl.interface import Requester
from geny_executor.stages.s15_hitl.types import HITLDecision, HITLRequest


class NullRequester(Requester):
    """Always-approve. Used as the safe default so the gate never blocks
    pipelines that don't need real human approval."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def description(self) -> str:
        return "Always approves — no human in the loop"

    async def request(self, request: HITLRequest, state: PipelineState) -> Optional[HITLDecision]:
        return HITLDecision.APPROVE


CallbackFn = Callable[[HITLRequest, PipelineState], Awaitable[Optional[HITLDecision]]]


class CallbackRequester(Requester):
    """Delegates to a host-supplied async callable.

    The callable receives ``(request, state)`` and returns an
    :class:`HITLDecision` (or ``None`` to fall through to the timeout
    policy). This is the bridge hosts wire to their UI / CLI / Slack
    channel — Stage 15 doesn't care how the answer is sourced.
    """

    def __init__(self, callback: Optional[CallbackFn] = None) -> None:
        self._callback = callback

    @property
    def name(self) -> str:
        return "callback"

    @property
    def description(self) -> str:
        return "Delegates to a host-supplied async callable"

    def configure(self, config: dict) -> None:
        callback = config.get("callback")
        if callback is not None:
            self._callback = callback

    async def request(self, request: HITLRequest, state: PipelineState) -> Optional[HITLDecision]:
        if self._callback is None:
            return None
        return await self._callback(request, state)


__all__ = [
    "CallbackFn",
    "CallbackRequester",
    "NullRequester",
]
