"""Stage 18 records each message into short-term memory exactly once — and
keeps recording after a compaction.

Two recorders share Stage 18: the ``ProviderDrivenStrategy`` in its strategy
slot, and ``MemoryStage._drive_provider`` whenever the stage holds the
provider. They kept two watermarks. With both active (a host that attached
the provider to the stage, as Geny does once the pipeline exists) every
message went into STM twice. And compaction translated only the stage's
key, so a strategy-only host's watermark pointed past the end of a
shortened history and nothing new was recorded until the list regrew.

Also pinned: the turn replay finds its provider where hosts put it — on the
retriever — instead of only in the stage slot hosts leave empty.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from geny_executor.core.compaction import run_compaction
from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Turn
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever
from geny_executor.memory.strategy import STM_RECORDED_KEY, ProviderDrivenStrategy
from geny_executor.stages.s02_context.artifact.default.compactors import TruncateCompactor
from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
from geny_executor.stages.s18_memory.artifact.default.stage import MemoryStage


def _run(coro):
    return asyncio.run(coro)


async def _provider() -> EphemeralMemoryProvider:
    provider = EphemeralMemoryProvider()
    await provider.initialize()
    return provider


async def _stm(provider) -> List[Any]:
    return list(await provider.stm().recent(n=10_000))


def _msg(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


class TestOnceEach:
    def test_strategy_and_stage_together_record_once(self) -> None:
        async def go():
            provider = await _provider()
            stage = MemoryStage(strategy=ProviderDrivenStrategy(provider))
            stage.provider = provider
            state = PipelineState(session_id="s")
            state.messages = [_msg("user", "hi"), _msg("assistant", "hello")]
            state.loop_decision = "continue"
            await stage.execute(None, state)
            state.messages.append(_msg("user", "again"))
            await stage.execute(None, state)
            return await _stm(provider)

        rows = _run(go())
        assert [r.content[0]["text"] for r in rows] == ["hi", "hello", "again"]

    def test_images_are_recorded_dehydrated(self) -> None:
        async def go():
            provider = await _provider()
            state = PipelineState(session_id="s")
            state.messages = [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "A" * 50_000}},
                {"type": "text", "text": "look"},
            ]}]
            await ProviderDrivenStrategy(provider).update(state)
            return state, await _stm(provider)

        state, rows = _run(go())
        assert rows[0].content[0]["source"]["data"] is None
        assert state.messages[0]["content"][0]["source"]["data"] == "A" * 50_000


class TestAfterACompaction:
    def test_new_messages_are_still_recorded(self) -> None:
        async def go():
            provider = await _provider()
            strategy = ProviderDrivenStrategy(provider)
            state = PipelineState(session_id="s")
            state.messages = [_msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(30)]
            await strategy.update(state)
            assert state.metadata[STM_RECORDED_KEY] == 30
            await run_compaction(state, TruncateCompactor(keep_last=6), trigger="guard")
            assert len(state.messages) <= 7
            state.messages.append(_msg("user", "after compaction"))
            await strategy.update(state)
            return await _stm(provider)

        rows = _run(go())
        texts = [r.content[0]["text"] for r in rows]
        assert texts.count("after compaction") == 1
        assert len(texts) == 31, "a message was recorded twice, or not at all"


class TestTheReplayFindsItsProvider:
    def test_through_the_retriever(self) -> None:
        """How hosts wire memory: ``attach_runtime(memory_retriever=...)``
        and ``from_manifest`` both leave the stage's own provider unset."""

        async def go():
            provider = await _provider()
            now = datetime.now(timezone.utc)
            for role, text in (("user", "earlier"), ("assistant", "EARLIER-ANSWER")):
                await provider.stm().append(
                    Turn(role=role, content=[{"type": "text", "text": text}], timestamp=now)
                )
            stage = ContextStage(retriever=MemoryAwareRetriever(provider))
            assert stage.provider is None
            state = PipelineState(session_id="s")
            state.messages = [{"role": "user", "content": "now"}]
            await stage.execute(None, state)
            return state

        state = _run(go())
        assert any("EARLIER-ANSWER" in str(m["content"]) for m in state.messages[:-1])


class TestTheProvidersPolicyIsTheStagesPolicy:
    """``provider.set_hooks`` is where a host puts its memory policy, and
    Stage 18 ignored it: it gated on a fresh ``MemoryHooks()`` of its own.
    Geny said "never file executions" (it archives them itself) and every
    turn was still filed — as an ``insights`` note, a category retrieval
    boosts."""

    async def _terminal_turn(self, stage, provider):
        state = PipelineState(session_id="s")
        state.messages = [_msg("user", "q"), _msg("assistant", "a")]
        state.final_text = "a"
        state.loop_decision = "complete"
        await stage.execute(None, state)
        return [e["type"] for e in state.events]

    def test_the_host_saying_never_is_obeyed(self) -> None:
        from geny_executor.memory.provider import MemoryEvent, MemoryHooks

        async def go():
            provider = await _provider()
            provider.set_hooks(MemoryHooks(should_record_execution=lambda s: False))
            stage = MemoryStage(strategy=ProviderDrivenStrategy(provider))
            stage.provider = provider
            return await self._terminal_turn(stage, provider)

        assert MemoryEvent.EXECUTION_RECORDED.value not in _run(go())

    def test_hooks_given_to_the_stage_still_win(self) -> None:
        from geny_executor.memory.provider import MemoryEvent, MemoryHooks

        async def go():
            provider = await _provider()
            provider.set_hooks(MemoryHooks(should_record_execution=lambda s: False))
            stage = MemoryStage(
                strategy=ProviderDrivenStrategy(provider),
                hooks=MemoryHooks(should_record_execution=lambda s: True),
            )
            stage.provider = provider
            return await self._terminal_turn(stage, provider)

        assert MemoryEvent.EXECUTION_RECORDED.value in _run(go())
