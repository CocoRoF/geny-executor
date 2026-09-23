"""The previous turns, put back in front of this one — the pipeline side.

``short_term_window`` decides WHAT the window is; these tests pin WHERE it
goes and what must not happen around it: it is built once per turn, never on
top of history the host already carries, never recorded to STM a second
time, never shown to the model a second time through the retriever, and
never sent as tool blocks to a request that defines no tools.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List


from geny_executor.core.state import PipelineState
from geny_executor.llm_client.base import ModelConfig
from geny_executor.llm_client.openai import OpenAIClient
from geny_executor.memory.provider import MemoryHooks, NoteDraft, Turn
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever
from geny_executor.memory.strategy import ProviderDrivenStrategy
from geny_executor.stages.s02_context.artifact.default.replay import (
    NoReplay,
    TurnWindowReplay,
    WINDOW_METADATA_KEY,
)
from geny_executor.stages.s02_context.artifact.default.stage import ContextStage


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc)


async def _record(provider, rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        await provider.stm().append(Turn(role=row["role"], content=row["content"], timestamp=_now()))


def _tool_turn(n: int) -> List[Dict[str, Any]]:
    uid = f"call-{n}"
    return [
        {"role": "user", "content": [{"type": "text", "text": f"task {n}"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": uid, "name": "Read", "input": {"file_path": f"/w/f{n}.txt"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": uid, "content": f"EVIDENCE-{n}"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": f"finished {n}"}]},
    ]


async def _provider_with(turns: int) -> EphemeralMemoryProvider:
    provider = EphemeralMemoryProvider()
    await provider.initialize()
    for n in range(turns):
        await _record(provider, _tool_turn(n))
    return provider


def _fresh_state(text: str = "what next?") -> PipelineState:
    state = PipelineState(session_id="s1")
    state.messages = [{"role": "user", "content": text}]
    return state


def _flat(messages: List[Dict[str, Any]]) -> str:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
            continue
        for b in content or []:
            out.append(str(b.get("text") or b.get("content") or b.get("name") or ""))
    return "\n".join(out)


class TestTheReplayGoesInFront:
    def test_a_fresh_turn_gets_its_past(self) -> None:
        async def go():
            provider = await _provider_with(6)
            stage = ContextStage(provider=provider)
            state = _fresh_state()
            await stage.execute(None, state)
            return state

        state = _run(go())
        assert state.messages[-1] == {"role": "user", "content": "what next?"}
        body = _flat(state.messages)
        assert "EVIDENCE-5" in body and "EVIDENCE-4" in body, "the nearest turns lost their tools"
        assert "EVIDENCE-3" not in body and "finished 3" in body
        assert "task 0" not in body, "a sixth turn back was replayed"
        meta = state.metadata[WINDOW_METADATA_KEY]
        assert (meta["turns"], meta["full"], meta["dialogue"]) == (5, 2, 3)
        events = [e for e in state.events if e["type"] == "context.short_term_window"]
        assert len(events) == 1 and events[0]["data"]["messages"] == meta["messages"]

    def test_the_budget_follows_the_route(self) -> None:
        """A 32k model gets a small window; nothing is sized in characters."""

        async def go(window: int):
            provider = await _provider_with(6)
            state = _fresh_state()
            state.context_window_budget = window
            state.max_tokens = 8_192
            await ContextStage(provider=provider).execute(None, state)
            return state.metadata[WINDOW_METADATA_KEY]["budget"]

        assert _run(go(32_768)) == 3_686
        assert _run(go(1_000_000)) == 148_771

    def test_the_share_is_a_setting_that_does_something(self) -> None:
        async def go(share: float):
            provider = await _provider_with(6)
            stage = ContextStage(provider=provider)
            stage.set_strategy("replay", "turn_window", {"window_share": share})
            state = _fresh_state()
            state.context_window_budget, state.max_tokens = 200_000, 32_000
            await stage.execute(None, state)
            return state.metadata[WINDOW_METADATA_KEY]["budget"]

        assert _run(go(0.15)) == 25_200
        assert _run(go(0.05)) == 8_400

    def test_only_once_per_turn(self) -> None:
        """Later iterations already carry it; rebuilding would move the
        prompt-cache prefix and double the history."""

        async def go():
            provider = await _provider_with(3)
            stage = ContextStage(provider=provider)
            state = _fresh_state()
            await stage.execute(None, state)
            first = list(state.messages)
            state.iteration = 1
            await stage.execute(None, state)
            return first, state.messages

        first, after = _run(go())
        assert after == first

    def test_history_the_host_already_carries_is_not_doubled(self) -> None:
        async def go():
            provider = await _provider_with(3)
            state = _fresh_state()
            state.messages = [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "now"},
            ]
            await ContextStage(provider=provider).execute(None, state)
            return state

        state = _run(go())
        assert len(state.messages) == 3
        assert WINDOW_METADATA_KEY not in state.metadata

    def test_no_provider_no_replay(self) -> None:
        state = _fresh_state()
        _run(ContextStage().execute(None, state))
        assert len(state.messages) == 1

    def test_the_slot_can_be_turned_off(self) -> None:
        async def go():
            provider = await _provider_with(3)
            state = _fresh_state()
            await ContextStage(provider=provider, replay=NoReplay()).execute(None, state)
            return state

        assert len(_run(go()).messages) == 1

    def test_a_question_the_host_recorded_first_is_not_its_own_history(self) -> None:
        async def go():
            provider = await _provider_with(2)
            await _record(provider, [{"role": "user", "content": [{"type": "text", "text": "what next?"}]}])
            state = _fresh_state("what next?")
            await ContextStage(provider=provider).execute(None, state)
            return state

        state = _run(go())
        assert _flat(state.messages).count("what next?") == 1

    def test_the_slot_is_configured_like_any_other(self) -> None:
        async def go():
            provider = await _provider_with(6)
            stage = ContextStage(provider=provider)
            stage.set_strategy("replay", "turn_window", {"full_turns": 1, "dialogue_turns": 1})
            state = _fresh_state()
            await stage.execute(None, state)
            return state.metadata[WINDOW_METADATA_KEY]

        meta = _run(go())
        assert (meta["full"], meta["dialogue"]) == (1, 1)
        schema = TurnWindowReplay.config_schema()
        assert {f.name for f in schema.fields} == {"full_turns", "dialogue_turns", "window_share"}


class TestNotRecordedTwice:
    def test_stage_18_records_only_this_turn(self) -> None:
        """Replayed messages came FROM short-term memory. Recording them
        again would double every turn, every turn."""

        async def go():
            provider = await _provider_with(3)
            before = len(await provider.stm().recent(n=1_000))
            state = _fresh_state()
            await ContextStage(provider=provider).execute(None, state)
            state.messages.append({"role": "assistant", "content": [{"type": "text", "text": "ok"}]})
            await ProviderDrivenStrategy(provider).update(state)
            after = len(await provider.stm().recent(n=1_000))
            return before, after

        before, after = _run(go())
        assert after - before == 2


class TestNotShownTwice:
    """The retriever's L0 carried the same turns as flattened text in the
    system prompt; conversation-record notes carried them again through
    search, under "Relevant Knowledge"."""

    async def _retrieve(self, *, replaying: bool):
        provider = await _provider_with(3)
        await provider.notes().write(NoteDraft(
            title="rollup", body="task 2 finished 2 Read inv", category="conversations",
        ))
        await provider.notes().write(NoteDraft(
            title="fact", body="task 2 relates to the inventory file", category="topics",
        ))
        hooks = MemoryHooks(
            recent_turns=6,
            transcript_categories=("conversations",),
            enable_vector_search=False,
        )
        state = _fresh_state("task 2")
        if replaying:
            state.metadata[WINDOW_METADATA_KEY] = {"turns": 3}
        return await MemoryAwareRetriever(provider, hooks=hooks).retrieve("task 2", state)

    def test_while_replaying_the_recent_layer_stands_down(self) -> None:
        chunks = _run(self._retrieve(replaying=True))
        layers = {c.metadata.get("layer") for c in chunks}
        assert "recent_turns" not in layers

    def test_while_replaying_conversation_records_stay_out_of_search(self) -> None:
        chunks = _run(self._retrieve(replaying=True))
        categories = {c.metadata.get("category") for c in chunks}
        assert "conversations" not in categories
        assert "topics" in categories, "real knowledge was dropped with the transcripts"

    def test_without_a_replay_nothing_changes(self) -> None:
        chunks = _run(self._retrieve(replaying=False))
        layers = {c.metadata.get("layer") for c in chunks}
        categories = {c.metadata.get("category") for c in chunks}
        assert "recent_turns" in layers and "conversations" in categories


class TestNoToolsNoToolBlocks:
    """Anthropic rejects tool blocks on a request that defines no tools, and
    the replay carries the calls an earlier hop made."""

    HISTORY = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "Read", "input": {"file_path": "/w/inv.txt"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "42 apples", "is_error": False},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "42 apples"}]},
        {"role": "user", "content": "and now?"},
    ]

    def _request(self, tools):
        client = OpenAIClient(api_key="sk-test")
        return client._build_request(
            model_config=ModelConfig(model="gpt-4o"),
            messages=self.HISTORY,
            system="",
            tools=tools,
            tool_choice=None,
            stream=False,
        )

    def test_they_become_prose(self) -> None:
        request = self._request(None)
        types = {b.get("type") for m in request.messages if isinstance(m["content"], list)
                 for b in m["content"]}
        assert types == {"text"}
        body = _flat(request.messages)
        assert "[called Read" in body and "inv.txt" in body
        assert "[Read returned: 42 apples]" in body

    def test_the_canonical_history_keeps_its_blocks(self) -> None:
        self._request(None)
        assert self.HISTORY[1]["content"][0]["type"] == "tool_use"

    def test_with_tools_nothing_changes(self) -> None:
        tools = [{"name": "Read", "description": "r", "input_schema": {"type": "object"}}]
        request = self._request(tools)
        assert request.messages[1]["content"][0]["type"] == "tool_use"

    def test_a_failure_says_so(self) -> None:
        history = [dict(m) for m in self.HISTORY]
        history[2] = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "no such file", "is_error": True},
        ]}
        client = OpenAIClient(api_key="sk-test")
        request = client._build_request(
            model_config=ModelConfig(model="gpt-4o"), messages=history, system="",
            tools=None, tool_choice=None, stream=False,
        )
        assert "[Read failed: no such file]" in _flat(request.messages)
