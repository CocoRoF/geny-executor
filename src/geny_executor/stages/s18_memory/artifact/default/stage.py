"""Default implementation of Stage 15: Memory."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import (
    ExecutionSummary,
    MemoryEvent,
    MemoryHooks,
    MemoryProvider,
    ReflectionContext,
    Turn,
)
from geny_executor.stages.s18_memory._dehydrate import (
    dehydrate_message,
    dehydrate_messages,
)
from geny_executor.stages.s18_memory.interface import (
    ConversationPersistence,
    MemoryUpdateStrategy,
)
from geny_executor.stages.s18_memory.artifact.default.persistence import (
    FilePersistence,
    InMemoryPersistence,
    NullPersistence,
)
from geny_executor.stages.s18_memory.artifact.default.strategies import (
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
    StructuredReflectiveStrategy,
)

logger = logging.getLogger(__name__)

_STATE_LAST_RECORDED = "memory.last_recorded_idx"
_TERMINAL_DECISIONS = frozenset({"complete", "error", "escalate"})


class MemoryStage(Stage[Any, Any]):
    """Stage 15: Memory.

    Dual abstraction:
      - Level 2 strategy: what to do with conversation data
      - Level 2 persistence: where to store it

    Phase 1+ also accepts an optional :class:`MemoryProvider` which
    handles the unified 4-axis memory contract. When set, the stage
    drives `provider.record_turn` per appended message,
    `provider.record_execution` on terminal states, and
    `provider.reflect`/`promote` per the supplied :class:`MemoryHooks`.
    The legacy strategy/persistence slots continue to run in parallel
    for back-compat — they may be set to no-ops when a provider is
    supplied.
    """

    def __init__(
        self,
        strategy: Optional[MemoryUpdateStrategy] = None,
        persistence: Optional[ConversationPersistence] = None,
        *,
        stateless: bool = False,
        persistence_path: str = "",
        provider: Optional[MemoryProvider] = None,
        hooks: Optional[MemoryHooks] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "strategy": StrategySlot(
                name="strategy",
                strategy=strategy or AppendOnlyStrategy(),
                registry={
                    "append_only": AppendOnlyStrategy,
                    "no_memory": NoMemoryStrategy,
                    "reflective": ReflectiveStrategy,
                    "structured_reflective": StructuredReflectiveStrategy,
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
        self._provider = provider
        self._hooks = hooks or MemoryHooks()
        if self._persistence_path and isinstance(self._persistence, NullPersistence):
            self._slots["persistence"].strategy = FilePersistence(base_dir=self._persistence_path)

    @property
    def provider(self) -> Optional[MemoryProvider]:
        return self._provider

    @provider.setter
    def provider(self, value: Optional[MemoryProvider]) -> None:
        self._provider = value

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

        if self._provider is not None:
            await self._drive_provider(state)

        persistence = self._persistence
        persistence_active = not isinstance(persistence, NullPersistence)

        if persistence_active and not state.session_id:
            logger.warning(
                "Memory persistence configured but session_id is empty — skipping persist"
            )
        if persistence_active and state.session_id:
            # Strip multimodal raw payloads (base64 image bytes etc.) before
            # writing to disk — see ``_dehydrate`` for the schema.
            await persistence.save(state.session_id, dehydrate_messages(state.messages))
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

    async def _drive_provider(self, state: PipelineState) -> None:
        provider = self._provider
        if provider is None:
            return

        # Incrementally record any newly-appended messages as STM turns.
        last_idx = int(state.metadata.get(_STATE_LAST_RECORDED, 0))
        new_msgs = state.messages[last_idx:]
        for msg in new_msgs:
            # STM also stores dehydrated copies — base64 payloads stay only
            # in the live ``state.messages`` for the current pipeline run.
            turn = Turn.from_state_message(dehydrate_message(msg))
            await provider.record_turn(turn)
            state.add_event(
                MemoryEvent.TURN_RECORDED.value,
                {"role": turn.role, "bytes": turn.bytes},
            )
        if new_msgs:
            state.metadata[_STATE_LAST_RECORDED] = len(state.messages)

        is_terminal = state.loop_decision in _TERMINAL_DECISIONS
        if is_terminal and self._hooks.should_record_execution(state):
            summary = ExecutionSummary.from_state(state)
            receipt = await provider.record_execution(summary)
            state.add_event(MemoryEvent.EXECUTION_RECORDED.value, receipt.to_event())

        if is_terminal and self._hooks.should_reflect(state):
            ctx = ReflectionContext.from_state(state)
            insights = await provider.reflect(ctx)
            for insight in insights:
                state.add_event(MemoryEvent.INSIGHT.value, insight.to_event())
                if self._hooks.should_auto_promote(insight):
                    if insight.ref is None:
                        # Reflection didn't materialise the note; skip
                        # promotion silently — providers that want
                        # auto-write should populate `insight.ref`.
                        continue
                    from geny_executor.memory.provider import Scope as _Scope

                    new_ref = await provider.promote(insight.ref, _Scope.USER)
                    state.add_event(
                        MemoryEvent.PROMOTED.value,
                        {
                            "ref": new_ref.to_dict(),
                            "from_scope": insight.ref.scope.value,
                            "to_scope": new_ref.scope.value,
                        },
                    )
