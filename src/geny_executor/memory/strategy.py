"""``ProviderDrivenStrategy`` — provider-driven Stage 18 strategy.

Replaces the legacy ``GenyMemoryStrategy`` (host-manager duck-type)
with a generic implementation that talks to a ``MemoryProvider``
directly. The strategy is intentionally minimal — it only wires
``provider.record_turn`` for newly-appended messages. Heavy lifting
(record_execution / reflect / promote) is handled by ``MemoryStage``
itself (see ``stages/s18_memory/artifact/default/stage.py``); the
strategy's job is to make sure every user/assistant message lands in
STM via the provider before that stage's terminal logic runs.

The legacy ``GenyMemoryStrategy`` and its ``ReflectionResolver``
helper are deleted in this cycle (D5 — "즉시 폐기"). Hosts that need
custom reflection plumbing should implement a callback through
``MemoryHooks.should_reflect`` + ``state.metadata['needs_reflection']``
and run their own reflection job out-of-band, OR use the executor's
native reflection path via the ``MemoryStage`` provider hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import MemoryProvider, Turn
from geny_executor.stages.s18_memory.interface import MemoryUpdateStrategy

logger = logging.getLogger(__name__)


_RECORDED_KEY = "memory.provider_strategy_recorded_idx"


class ProviderDrivenStrategy(MemoryUpdateStrategy):
    """Per-turn provider drive strategy.

    For every newly-appended message in ``state.messages`` since the
    last invocation, calls ``provider.stm().append(Turn.from_state_message(msg))``.
    This is a thin wrapper because ``MemoryStage._drive_provider``
    already handles the same duty when ``provider`` is attached to
    the stage; ``ProviderDrivenStrategy`` exists so the strategy slot
    of ``MemoryStage`` is non-empty even when the host wants the
    provider to be the sole STM authority. With this strategy + a
    provider attached, the legacy ``AppendOnlyStrategy`` /
    ``ReflectiveStrategy`` paths can stay unused.

    Construction takes only the provider — every piece of policy
    (should_reflect, should_auto_promote, importance gate) lives on
    ``provider.set_hooks(MemoryHooks(...))`` and is consumed by
    ``MemoryStage`` directly.
    """

    def __init__(self, provider: Optional[MemoryProvider] = None) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "provider_driven"

    @property
    def description(self) -> str:
        return "Drive provider.record_turn per appended message; let MemoryStage own reflection / promotion."

    def attach_provider(self, provider: MemoryProvider) -> None:
        """Late-bind the provider after construction.

        ``default_manifest`` constructs the strategy before the host
        has built its provider; the host then calls this method (or
        ``MemoryStage.provider = ...``) once the provider exists.
        """
        self._provider = provider

    async def update(self, state: PipelineState) -> None:
        provider = self._provider
        if provider is None:
            return
        last_recorded = int(state.metadata.get(_RECORDED_KEY, 0))
        new_msgs = state.messages[last_recorded:]
        if not new_msgs:
            return

        recorded = 0
        for msg in new_msgs:
            try:
                turn = Turn.from_state_message(msg)
            except Exception:  # noqa: BLE001
                logger.debug("provider_driven: Turn.from_state_message failed", exc_info=True)
                continue
            try:
                await provider.record_turn(turn)
                recorded += 1
            except Exception:  # noqa: BLE001
                logger.debug("provider_driven: record_turn failed", exc_info=True)
                continue

        state.metadata[_RECORDED_KEY] = len(state.messages)
        if recorded:
            try:
                state.add_event(
                    "memory.provider_recorded",
                    {"count": recorded, "total_messages": len(state.messages)},
                )
            except Exception:  # noqa: BLE001
                pass


__all__ = ["ProviderDrivenStrategy"]
