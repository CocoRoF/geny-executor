"""Default HITL requesters for Stage 15 (S9b.3)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl.interface import Requester
from geny_executor.stages.s15_hitl.types import HITLDecision, HITLRequest

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


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


class PipelineResumeRequester(Requester):
    """Requester that pauses the pipeline until ``Pipeline.resume`` fires (S9c.1).

    On every request the requester:

    1. creates an :class:`asyncio.Future` on the running event loop,
    2. registers it on the host :class:`Pipeline` under the request's
       ``token`` (via ``pipeline._pending_hitl``),
    3. awaits the future,
    4. removes the registration in a ``finally`` so a cancelled run
       cannot leak Future objects.

    External code (typically a websocket handler or HTTP endpoint
    receiving the human's verdict) then calls
    ``pipeline.resume(token, decision)`` to satisfy the future. The
    pipeline is taken by reference at construction time so it stays
    alive for the duration of the run; pipelines released after the
    run completes will GC normally.
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "pipeline_resume"

    @property
    def description(self) -> str:
        return "Awaits Pipeline.resume(token, decision) — for cross-request HITL"

    async def request(self, request: HITLRequest, state: PipelineState) -> Optional[HITLDecision]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pipeline._pending_hitl[request.token] = future
        try:
            return await future
        finally:
            # Drop the registration whether resolved, cancelled, or
            # exceptioned out — never leak entries on the pipeline.
            self._pipeline._pending_hitl.pop(request.token, None)


__all__ = [
    "CallbackFn",
    "CallbackRequester",
    "NullRequester",
    "PipelineResumeRequester",
]
