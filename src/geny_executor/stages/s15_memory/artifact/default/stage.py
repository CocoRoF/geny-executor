"""Default implementation of Stage 15: Memory."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_memory.interface import (
    ConversationPersistence,
    MemoryUpdateStrategy,
)
from geny_executor.stages.s15_memory.artifact.default.persistence import (
    FilePersistence,
    InMemoryPersistence,
    NullPersistence,
)
from geny_executor.stages.s15_memory.artifact.default.strategies import (
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
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
        persistence_path: str = "",
    ):
        self._slots: Dict[str, StrategySlot] = {
            "strategy": StrategySlot(
                name="strategy",
                strategy=strategy or AppendOnlyStrategy(),
                registry={
                    "append_only": AppendOnlyStrategy,
                    "no_memory": NoMemoryStrategy,
                    "reflective": ReflectiveStrategy,
                },
                description="Memory update strategy",
            ),
            "persistence": StrategySlot(
                name="persistence",
                strategy=persistence or NullPersistence(),
                registry={
                    "null": NullPersistence,
                    "in_memory": InMemoryPersistence,
                    "file": FilePersistence,
                },
                description="Conversation persistence backend",
            ),
        }
        self._stateless = stateless
        self._persistence_path = str(persistence_path)
        if self._persistence_path and isinstance(self._persistence, NullPersistence):
            self._slots["persistence"].strategy = FilePersistence(base_dir=self._persistence_path)

    @property
    def _strategy(self) -> MemoryUpdateStrategy:
        return self._slots["strategy"].strategy  # type: ignore[return-value]

    @property
    def _persistence(self) -> ConversationPersistence:
        return self._slots["persistence"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "memory"

    @property
    def order(self) -> int:
        return 15

    @property
    def category(self) -> str:
        return "egress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="memory",
            fields=[
                ConfigField(
                    name="stateless",
                    type="boolean",
                    label="Stateless",
                    description="Skip persistence and memory update (ephemeral sessions).",
                    default=False,
                    ui_widget="toggle",
                ),
                ConfigField(
                    name="persistence_path",
                    type="string",
                    label="Persistence Path",
                    description="Directory used by FilePersistence when set.",
                    default="",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "stateless": self._stateless,
            "persistence_path": self._persistence_path,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "stateless" in config:
            self._stateless = bool(config["stateless"])
        if "persistence_path" in config:
            path = str(config["persistence_path"])
            self._persistence_path = path
            if path:
                self._slots["persistence"].strategy = FilePersistence(base_dir=path)
            elif not path and isinstance(self._persistence, FilePersistence):
                self._slots["persistence"].strategy = NullPersistence()

    def should_bypass(self, state: PipelineState) -> bool:
        return self._stateless or isinstance(self._strategy, NoMemoryStrategy)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        await self._strategy.update(state)

        persistence = self._persistence
        persistence_active = not isinstance(persistence, NullPersistence)

        if persistence_active and not state.session_id:
            logger.warning(
                "Memory persistence configured but session_id is empty — skipping persist"
            )
        if persistence_active and state.session_id:
            await persistence.save(state.session_id, state.messages)
            state.add_event(
                "memory.persisted",
                {
                    "session_id": state.session_id,
                    "message_count": len(state.messages),
                    "persistence": type(persistence).__name__,
                },
            )

        state.add_event("memory.updated", {"strategy": type(self._strategy).__name__})
        return input
