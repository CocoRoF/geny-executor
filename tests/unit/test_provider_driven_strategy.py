"""``ProviderDrivenStrategy`` — Stage 18 strategy tests.

Verifies the per-turn provider drive: every newly-appended message is
forwarded to ``provider.record_turn``. The strategy is intentionally
minimal — reflection / promotion live on ``MemoryStage`` itself.
"""

from __future__ import annotations

import asyncio

from geny_executor.core.state import PipelineState
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.strategy import ProviderDrivenStrategy


def _run(coro):
    return asyncio.run(coro)


def test_strategy_idle_when_no_provider() -> None:
    s = ProviderDrivenStrategy()
    state = PipelineState()
    state.messages = [{"role": "user", "content": "hi"}]
    _run(s.update(state))  # no exception


def test_strategy_records_only_new_messages() -> None:
    p = EphemeralMemoryProvider()
    s = ProviderDrivenStrategy(p)
    state = PipelineState()
    state.session_id = "sess"

    async def go() -> int:
        await p.initialize()
        state.messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        await s.update(state)

        # Append more, run again — only the new ones should be recorded.
        state.messages.append({"role": "user", "content": "second"})
        await s.update(state)

        turns = await p.stm().recent(n=10)
        return len(turns)

    assert _run(go()) == 3


def test_strategy_late_attach_provider() -> None:
    s = ProviderDrivenStrategy()
    state = PipelineState()
    state.messages = [{"role": "user", "content": "hi"}]
    _run(s.update(state))  # no provider — no-op

    p = EphemeralMemoryProvider()
    s.attach_provider(p)

    async def go() -> int:
        await p.initialize()
        await s.update(state)
        turns = await p.stm().recent(n=10)
        return len(turns)

    # Earlier attempt did not record (no provider). Late attach + re-update
    # records every message because the recorded marker only advances when
    # a provider is present.
    assert _run(go()) == 1


def test_strategy_emits_event_when_recorded() -> None:
    p = EphemeralMemoryProvider()
    s = ProviderDrivenStrategy(p)
    state = PipelineState()
    state.session_id = "sess"
    state.messages = [{"role": "user", "content": "hi"}]

    async def go() -> bool:
        await p.initialize()
        await s.update(state)
        return any(e.get("type") == "memory.provider_recorded" for e in state.events)

    assert _run(go()) is True
