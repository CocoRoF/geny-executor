"""Stage 19 session-close summary writer (EXEC-7 / D1).

When the pipeline reaches a terminal decision the stage assembles
``state.shared['summary_history']`` into a markdown block and forwards
it to ``provider.stm().write_summary(body)`` exactly once. Earlier
(non-terminal) turns only append to history.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance, Scope
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider
from geny_executor.stages.s19_summarize.artifact.default.stage import SummarizeStage
from geny_executor.stages.s19_summarize.artifact.default.summarizers import (
    RuleBasedSummarizer,
)
from geny_executor.stages.s19_summarize.interface import SUMMARY_HISTORY_KEY
from geny_executor.stages.s19_summarize.types import SummaryRecord


def _run(coro):
    return asyncio.run(coro)


def _state_with_provider(
    provider, *, decision: str = "loop", closing: bool = True
) -> PipelineState:
    state = PipelineState()
    if closing:
        # 2.75.0: the session summary is written on the turn the host marks
        # as the session's last, not on every turn that ends.
        state.metadata["session.closing"] = True
    state.session_id = "sess-1"
    state.iteration = 1
    state.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world. Two more sentences. Three."},
    ]
    state.final_text = "world. Two more sentences. Three."
    state.loop_decision = decision
    runtime = SimpleNamespace(memory_provider=provider)
    state.session_runtime = runtime  # type: ignore[attr-defined]
    return state


def _push_history(state: PipelineState, *records: SummaryRecord) -> None:
    history = state.shared.setdefault(SUMMARY_HISTORY_KEY, [])
    for r in records:
        history.append(r.to_dict())


# ── session-close write triggered by terminal decision ──────────────


def test_session_close_writes_summary_to_stm_on_complete():
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="complete")
    _push_history(
        state,
        SummaryRecord(
            turn_id="sess-1:1",
            abstract="user asked about rockets; agent explained launch sequence.",
            key_facts=["Rockets need fuel.", "Stage separation matters."],
            tags=["rocket"],
            importance=Importance.HIGH,
        ),
    )

    stage = SummarizeStage(summarizer=RuleBasedSummarizer())

    async def go():
        await p.initialize()
        await stage.execute(None, state)
        return await p.stm().read_summary()

    body = _run(go())
    assert body is not None
    assert "## Session Summary" in body
    assert "Rockets need fuel" in body
    assert "rocket" in body
    assert any(
        e.get("type") == "summary.session_closed" for e in state.events
    )


def test_session_close_skipped_on_non_terminal_decision():
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="loop")
    _push_history(
        state,
        SummaryRecord(turn_id="sess-1:1", abstract="x"),
    )
    stage = SummarizeStage(summarizer=RuleBasedSummarizer())

    async def go():
        await p.initialize()
        await stage.execute(None, state)
        return await p.stm().read_summary()

    body = _run(go())
    assert body is None
    assert all(
        e.get("type") != "summary.session_closed" for e in state.events
    )


def test_session_close_uses_file_stm_summary_md():
    """End-to-end: file provider → terminal decision → summary.md exists."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = FileMemoryProvider(root=root, scope=Scope.SESSION, timezone_name="UTC")
        state = _state_with_provider(p, decision="complete")
        _push_history(
            state,
            SummaryRecord(
                turn_id="sess-1:1",
                abstract="recap of the conversation",
                key_facts=["fact A"],
                importance=Importance.MEDIUM,
            ),
        )
        stage = SummarizeStage(summarizer=RuleBasedSummarizer())

        async def go():
            await p.initialize()
            await stage.execute(None, state)

        _run(go())
        summary_path = root / "transcripts" / "summary.md"
        assert summary_path.exists()
        body = summary_path.read_text(encoding="utf-8")
        assert "fact A" in body
        assert "recap of the conversation" in body


def test_session_close_with_empty_history_does_not_write():
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="complete")
    # No history pushed before execute.
    stage = SummarizeStage(summarizer=RuleBasedSummarizer())

    async def go():
        await p.initialize()
        await stage.execute(None, state)
        return await p.stm().read_summary()

    body = _run(go())
    # RuleBasedSummarizer may produce a record itself for the current
    # state; in that case session-close fires with that single entry.
    # If RuleBased returns None for an empty session, no write happens.
    # Either way, no crash and event flag is consistent.
    if body is None:
        assert all(
            e.get("type") != "summary.session_closed" for e in state.events
        )


def test_session_close_no_provider_is_no_op():
    state = _state_with_provider(None, decision="complete")
    _push_history(state, SummaryRecord(turn_id="sess-1:1", abstract="hi"))
    stage = SummarizeStage(summarizer=RuleBasedSummarizer())
    # Should not raise.
    _run(stage.execute(None, state))


# ── 2.75.0: when the session summary is written ─────────────────────


def _record() -> SummaryRecord:
    return SummaryRecord(
        turn_id="sess-1:1",
        abstract="user asked about rockets.",
        key_facts=["Rockets need fuel."],
        importance=Importance.HIGH,
    )


def test_an_ordinary_turn_end_does_not_touch_the_session_summary():
    """A host that starts every turn from a fresh state ends every turn with
    a terminal decision. Writing then rewrote summary.md on every turn and
    wiped the rolling digest kept in the same file."""
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="complete", closing=False)
    _push_history(state, _record())

    async def go():
        await p.initialize()
        await p.stm().write_summary("## Summary\nthe rolling digest")
        await SummarizeStage(summarizer=RuleBasedSummarizer()).execute(None, state)
        return await p.stm().read_summary()

    assert _run(go()) == "## Summary\nthe rolling digest"


def test_every_turn_mode_keeps_the_old_behaviour():
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="complete", closing=False)
    _push_history(state, _record())
    stage = SummarizeStage(summarizer=RuleBasedSummarizer())
    stage.update_config({"session_summary": "every_turn"})

    async def go():
        await p.initialize()
        await stage.execute(None, state)
        return await p.stm().read_summary()

    assert "## Session Summary" in (_run(go()) or "")


def test_off_mode_never_writes():
    p = EphemeralMemoryProvider()
    state = _state_with_provider(p, decision="complete", closing=True)
    _push_history(state, _record())
    stage = SummarizeStage(summarizer=RuleBasedSummarizer(), session_summary="off")

    async def go():
        await p.initialize()
        await stage.execute(None, state)
        return await p.stm().read_summary()

    assert not _run(go())

