"""Stage 15: Memory — update and persist memory."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_memory.strategies import (
    AppendOnlyStrategy,
    MemoryUpdateStrategy,
    NoMemoryStrategy,
)
from geny_executor.stages.s15_memory.persistence import (
    ConversationPersistence,
)

logger = logging.getLogger(__name__)


class MemoryStage(Stage[Any, Any]):
    """Stage 15: Memory.

    Dual abstraction:
      - Level 2 strategy: what to do with conversation data
      - Level 2 persistence: where to store it
    """

    def __init__(
        self,
        strategy: Optional[MemoryUpdateStrategy] = None,
        persistence: Optional[ConversationPersistence] = None,
        *,
        stateless: bool = False,
    ):
        self._strategy = strategy or AppendOnlyStrategy()
        self._persistence = persistence
        self._stateless = stateless

    @property
    def name(self) -> str:
        return "memory"

    @property
    def order(self) -> int:
        return 15

    @property
    def category(self) -> str:
        return "egress"

    def should_bypass(self, state: PipelineState) -> bool:
        return self._stateless or isinstance(self._strategy, NoMemoryStrategy)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Run memory update strategy
        await self._strategy.update(state)

        # Persist if configured
        if self._persistence and not state.session_id:
            logger.warning(
                "Memory persistence configured but session_id is empty — skipping persist"
            )
        if self._persistence and state.session_id:
            await self._persistence.save(state.session_id, state.messages)
            state.add_event(
                "memory.persisted",
                {
                    "session_id": state.session_id,
                    "message_count": len(state.messages),
                    "persistence": type(self._persistence).__name__,
                },
            )

        state.add_event(
            "memory.updated",
            {
                "strategy": type(self._strategy).__name__,
            },
        )

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="strategy",
                current_impl=type(self._strategy).__name__,
                available_impls=[
                    "AppendOnlyStrategy",
                    "NoMemoryStrategy",
                    "ReflectiveStrategy",
                ],
            ),
            StrategyInfo(
                slot_name="persistence",
                current_impl=(type(self._persistence).__name__ if self._persistence else "None"),
                available_impls=[
                    "InMemoryPersistence",
                    "FilePersistence",
                ],
            ),
        ]
