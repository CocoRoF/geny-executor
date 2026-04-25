"""Stage 6: API — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, TYPE_CHECKING

from geny_executor.core.errors import ErrorCategory
from geny_executor.core.stage import Strategy
from geny_executor.stages.s06_api.types import APIRequest, APIResponse

if TYPE_CHECKING:
    from geny_executor.core.config import ModelConfig
    from geny_executor.core.state import PipelineState


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
