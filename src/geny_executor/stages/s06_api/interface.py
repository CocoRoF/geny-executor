"""Stage 6: API — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from geny_executor.core.errors import ErrorCategory
from geny_executor.core.stage import Strategy
from geny_executor.stages.s06_api.types import APIRequest, APIResponse

if TYPE_CHECKING:
    from geny_executor.core.config import ModelConfig
    from geny_executor.core.state import PipelineState
    from geny_executor.llm_client import BaseClient

#: One retry-wrapped client call. Built by ``APIStage.execute`` as a
#: closure over the resolved client/model/stream mode; the argument is
#: an optional list of *loop-local* messages appended after
#: ``state.messages`` for this call only (the internal loop's pending
#: tool exchanges — ``state.messages`` itself is not mutated mid-loop).
#: Each invocation emits its own ``api.request`` / ``api.response``
#: pair, so every inner call of an agentic loop is individually visible
#: to hosts exactly like the single-call path always was.
ToolLoopCall = Callable[[Optional[List[Dict[str, Any]]]], Awaitable[APIResponse]]


class APIProvider(Strategy):
    """Base interface for making API calls."""

    @abstractmethod
    async def create_message(self, request: APIRequest) -> APIResponse:
        """Create a message (non-streaming)."""
        ...

    async def create_message_stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        """Create a message with streaming. Default: falls back to non-streaming."""
        response = await self.create_message(request)
        yield {"type": "message_complete", "response": response}


class RetryStrategy(Strategy):
    """Base interface for retry logic."""

    @abstractmethod
    def should_retry(self, category: ErrorCategory, attempt: int) -> bool:
        """Whether to retry given the error category and attempt number."""
        ...

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds before next retry."""
        ...

    @property
    def max_retries(self) -> int:
        return 0


class ToolLoopStrategy(Strategy):
    """Decide WHERE the agentic tool loop runs for this stage's calls.

    Two execution shapes exist for "the model wants tools" (2.3.0):

    - **pipeline** — the stage makes exactly one client call and returns
      the response verbatim, tool_use blocks included. Stage 9 parses
      them, Stage 10 dispatches, Stage 16 loops the whole pipeline.
      Full per-round-trip stage control (guards, token tracking,
      review, evaluation) at the cost of re-running every stage per
      tool round-trip.
    - **internal** — the strategy resolves tool calls *inside* Stage 6
      (call → dispatch → call …) and returns only the final response,
      mirroring how the ``claude_code_cli`` backend's subprocess loop
      already behaves (see ``StreamJsonAccumulator.finalize``).

    The strategy never talks to the client directly — it drives the
    stage-built :data:`ToolLoopCall` closure so every call shares the
    stage's retry strategy, timeout plumbing, stream handling and
    ``api.request``/``api.response``/``api.error`` event contract.
    """

    @abstractmethod
    async def run(
        self,
        *,
        call: ToolLoopCall,
        client: "BaseClient",
        state: "PipelineState",
    ) -> APIResponse:
        """Produce the response Stage 6 hands to the rest of the pipeline.

        Implementations that resolve tool exchanges internally must
        record those intermediate messages onto ``state.messages`` (in
        order) before returning — the stage appends only the FINAL
        assistant content, exactly as it does on the single-call path.
        """
        ...


class ModelRouter(Strategy):
    """Decide which model to use for a given API call.

    Implementations inspect the resolved :class:`ModelConfig` together
    with the live :class:`PipelineState` and either return a *new* config
    (overriding the default for this call) or ``None`` to keep what the
    pipeline already chose.

    The router runs *after* ``Stage.resolve_model_config(state)`` so the
    decision is based on the same baseline that would have been used.
    Returning a new ``ModelConfig`` only affects this single
    invocation — the underlying state is not mutated.
    """

    @abstractmethod
    def route(self, cfg: "ModelConfig", state: "PipelineState") -> Optional["ModelConfig"]:
        """Return an overridden model config, or ``None`` to keep ``cfg``."""
        ...
