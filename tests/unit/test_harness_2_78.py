"""2.78.0 — cache, cost, and the agent's own turns.

Each test pins one defect found by the 2026-09-29 harness audit:

* the replay window slid one turn every turn, so its first message changed
  every time and no prompt cache ever matched it across turns;
* turns the agent started on its own (wake-ups, screen glances) took the
  replay's conversation slots and pushed the last real exchange out;
* "silent" meant one thing to the replay and another to the host;
* a turn's stale tool output was resent in full on every call until
  compaction at 80% of the window — which on a large window never came;
* a turn's usage said nothing about whether the cache was working, and
  OpenAI's cached tokens were counted twice.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.core.usage import turn_usage_summary
from geny_executor.memory.provider import Turn
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.short_term_window import (
    WindowConfig,
    build_window,
    is_silence_text,
)
from geny_executor.stages.s02_context.artifact.default.replay import (
    STABLE_PREFIX_KEY,
    TurnWindowReplay,
)
from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
from geny_executor.stages.s05_cache.artifact.default.strategies import AggressiveCacheStrategy
from geny_executor.stages.s16_loop.turn_budget import turn_input_tokens

_T0 = datetime(2026, 9, 29, tzinfo=timezone.utc)


def _rows(spec: List[tuple]) -> List[Turn]:
    """``(question, answer, internal?)`` → STM rows, one minute apart."""
    rows = []
    for n, (q, a, internal) in enumerate(spec):
        ts = _T0 + timedelta(minutes=n)
        rows.append(Turn(role="user", content=q, timestamp=ts, direction="internal" if internal else "inbound"))
        rows.append(Turn(role="assistant", content=a, timestamp=ts))
    return rows


def _texts(messages) -> List[str]:
    out = []
    for m in messages:
        c = m["content"]
        out.append(c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict)))
    return out


def test_silence_is_one_rule():
    assert is_silence_text("[SILENT]")
    assert is_silence_text("[SILENT].")
    assert is_silence_text("[neutral] [SILENT]")
    assert not is_silence_text("[SILENT] but I have something to say")
    assert not is_silence_text("I'm here")


def test_the_agents_own_turns_do_not_push_the_conversation_out():
    spec = []
    for i in range(6):
        spec.append((f"user question {i}", f"answer {i}", False))
        spec.append((f"[THINKING_TRIGGER] wake {i}", f"musing {i}", True))
    window = build_window(_rows(spec), WindowConfig(full_turns=2, dialogue_turns=3, max_tokens=50_000))
    joined = " ".join(_texts(window.messages))
    # The last five conversation turns, all of them ...
    for i in range(1, 6):
        assert f"user question {i}" in joined
    assert "user question 0" not in joined
    # ... and only the latest thing it said on its own.
    assert "musing 5" in joined and "musing 4" not in joined
    assert window.autonomous == 1


def test_a_quiet_conversation_leaves_room_for_the_agents_turns():
    spec = [("hello", "hi", False)] + [(f"[THINKING_TRIGGER] {i}", f"musing {i}", True) for i in range(6)]
    window = build_window(_rows(spec), WindowConfig(full_turns=2, dialogue_turns=3, max_tokens=50_000))
    assert window.turns == 5  # one conversation turn + four of its own


def test_the_window_holds_its_start_so_its_older_part_is_the_same_text():
    spec = [(f"q{i}", f"a{i}", False) for i in range(6)]
    cfg = WindowConfig(full_turns=2, dialogue_turns=3, max_tokens=50_000)
    first = build_window(_rows(spec), cfg)
    assert first.anchor and first.stable_messages == 6  # three dialogue turns

    spec.append(("q6", "a6", False))
    cfg.anchor = first.anchor
    second = build_window(_rows(spec), cfg)
    # Same start; the older part renders identically, so a cache matches it.
    assert second.messages[: first.stable_messages] == first.messages[: first.stable_messages]
    assert second.turns == 6

    # Past turns + sticky_turns it moves on.
    for i in range(7, 10):
        spec.append((f"q{i}", f"a{i}", False))
    moved = build_window(_rows(spec), cfg)
    assert moved.turns == 5 and moved.anchor != first.anchor


def test_the_replay_marks_its_stable_part_for_the_cache():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        for row in _rows([(f"q{i}", f"a{i}", False) for i in range(6)]):
            await provider.stm().append(row)
        state = PipelineState(session_id="s")
        state.model = "claude-sonnet-4-6"
        state.messages = [{"role": "user", "content": "now"}]
        await TurnWindowReplay().replay(state, provider)
        return state

    state = asyncio.run(go())
    stable = state.metadata[STABLE_PREFIX_KEY]
    AggressiveCacheStrategy().apply_cache_markers(state)
    marked = [
        i
        for i, m in enumerate(state.messages)
        if isinstance(m["content"], list) and any("cache_control" in b for b in m["content"])
    ]
    assert stable - 1 in marked
    assert len(marked) <= 2  # stable point + moving point


def test_stale_tool_output_is_trimmed_for_cost_but_the_replay_is_not():
    big = "log line\n" * 2000  # ~18k chars
    state = PipelineState(session_id="s")
    replayed = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": big}]},
        {"role": "assistant", "content": "earlier answer"},
    ]
    state.messages = list(replayed) + [{"role": "user", "content": "now"}]
    for i in range(10):
        state.messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]}
        )
        state.messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": big}]})
    state.metadata["memory.short_term_window"] = {"messages": len(replayed)}

    stage = ContextStage(prune_over_tokens=10_000)
    stage._prune_for_cost(state)

    assert state.messages[2]["content"][0]["content"] == big  # the replay: untouched
    trimmed = [m for m in state.messages[5:] if isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result" and len(m["content"][0]["content"]) < len(big)]
    assert trimmed  # older results of this turn: trimmed
    assert state.messages[-1]["content"][0]["content"] == big  # the latest: untouched
    assert any(e["type"] == "context.pruned" for e in state.events)


def test_a_turns_usage_counts_each_token_once():
    state = PipelineState(session_id="s")
    state.turn_token_usage = [
        # Anthropic: input excludes the cached part.
        TokenUsage(input_tokens=100, cache_read_input_tokens=9_000, cache_creation_input_tokens=900),
        # OpenAI chat: prompt_tokens already includes it.
        TokenUsage(input_tokens=12_000, cache_read_input_tokens=10_000, input_includes_cache_read=True),
    ]
    summary = turn_usage_summary(state)
    assert summary["calls"] == 2
    assert summary["first_prompt_tokens"] == 10_000
    assert summary["max_prompt_tokens"] == 12_000
    assert summary["input_tokens"] == 22_000
    assert summary["cache_read_share"] == round(19_000 / 22_000, 3)
    assert turn_input_tokens(state) == 22_000
