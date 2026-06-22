"""MemoryRollup — semantic rolling digest (2.16.0)."""
import pytest
from geny_executor.memory.rollup import (
    MemoryRollup, build_segment_instruction, PRESERVE_CLAUSE,
)


class _Turn:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _STM:
    def __init__(self, turns, summary=None):
        self._turns = turns
        self.summary = summary
        self.written = None
    async def recent(self, n=20):
        return self._turns[-n:]
    async def read_summary(self):
        return self.summary
    async def write_summary(self, body):
        self.written = body


class _Provider:
    def __init__(self, stm):
        self._stm = stm
    def stm(self):
        return self._stm


def _summarizer(captured):
    async def s(instruction):
        captured["instruction"] = instruction
        return "## Summary\nFolded digest.\n## Open Threads\n- finish X"
    return s


@pytest.mark.asyncio
async def test_segment_folds_prior_and_turns_then_writes_summary():
    stm = _STM([_Turn("user", "내 이름은 하렴"), _Turn("assistant", "안녕하세요!")],
               summary="## Summary\nprior digest")
    captured = {}
    r = MemoryRollup(_Provider(stm), summarize=_summarizer(captured))
    out = await r.summarize_segment()
    assert out and "Folded digest" in out
    assert stm.written == out                      # persisted to L1 slot
    assert "prior digest" in captured["instruction"]   # prior folded in
    assert "하렴" in captured["instruction"]            # new raw turns included
    assert "ALWAYS PRESERVE" in captured["instruction"]  # preservation clause present


@pytest.mark.asyncio
async def test_empty_stm_is_noop():
    stm = _STM([])
    r = MemoryRollup(_Provider(stm), summarize=_summarizer({}))
    assert await r.summarize_segment() is None
    assert stm.written is None


@pytest.mark.asyncio
async def test_summarizer_failure_never_raises_and_skips_write():
    stm = _STM([_Turn("user", "hi")], summary="")
    async def boom(_):
        raise RuntimeError("llm down")
    r = MemoryRollup(_Provider(stm), summarize=boom)
    report = await r.run()
    assert report.segment_written is False
    assert stm.written is None


@pytest.mark.asyncio
async def test_run_reports_chars():
    stm = _STM([_Turn("user", "hi")], summary="")
    r = MemoryRollup(_Provider(stm), summarize=_summarizer({}))
    report = await r.run()
    assert report.segment_written is True
    assert report.chars_out > 0


def test_instruction_has_structure_and_preserve():
    instr = build_segment_instruction(prior_digest="", raw_turns="[user] hi", max_chars=4000)
    assert PRESERVE_CLAUSE.split("\n")[0] in instr
    assert "## Facts & Decisions" in instr
    assert "4000" in instr


class _Notes:
    def __init__(self, pinned=""):
        self._pinned = pinned
        self.written = None
    async def load_pinned(self, *, category="critical", max_chars=3000):
        return self._pinned
    async def write(self, draft):
        self.written = draft
        return draft


class _ProviderWithNotes:
    def __init__(self, stm, notes):
        self._stm = stm
        self._notes = notes
    def stm(self):
        return self._stm
    def notes(self):
        return self._notes


@pytest.mark.asyncio
async def test_evergreen_merges_current_and_recent_then_writes_pinned_critical():
    from geny_executor.memory.rollup import EVERGREEN_FILENAME, EVERGREEN_CATEGORY
    stm = _STM([], summary="## Summary\n사장님 prefers rhythm games")
    notes = _Notes(pinned="## Identity\nGeny, a VTuber")
    captured = {}
    async def s(instr):
        captured["instr"] = instr
        return "## Identity\nGeny\n## User\n사장님 — likes rhythm games"
    r = MemoryRollup(_ProviderWithNotes(stm, notes), summarize=s)
    out = await r.rollup_evergreen()
    assert out and "사장님" in out
    # written as the rewritable pinned critical evergreen note
    assert notes.written is not None
    assert notes.written.filename == EVERGREEN_FILENAME
    assert notes.written.category == EVERGREEN_CATEGORY
    # merge saw both current evergreen + the latest rolling digest
    assert "Geny, a VTuber" in captured["instr"]
    assert "rhythm games" in captured["instr"]
    assert "ALWAYS PRESERVE" in captured["instr"]


@pytest.mark.asyncio
async def test_evergreen_noop_when_nothing():
    stm = _STM([], summary="")
    notes = _Notes(pinned="")
    r = MemoryRollup(_ProviderWithNotes(stm, notes), summarize=_summarizer({}))
    assert await r.rollup_evergreen() is None
    assert notes.written is None


@pytest.mark.asyncio
async def test_run_with_evergreen_flag():
    stm = _STM([_Turn("user", "hi")], summary="prior")
    notes = _Notes(pinned="x")
    r = MemoryRollup(_ProviderWithNotes(stm, notes), summarize=_summarizer({}))
    report = await r.run(evergreen=True)
    assert report.segment_written is True
    assert report.evergreen_written is True


@pytest.mark.asyncio
async def test_daily_writes_dated_digest_note():
    stm = _STM([], summary="## Summary\ntoday's digest")
    notes = _Notes()
    r = MemoryRollup(_ProviderWithNotes(stm, notes), summarize=_summarizer({}))
    out = await r.rollup_daily(day="2026-06-22")
    assert out and "today's digest" in out
    assert notes.written.filename == "__digest_2026-06-22__.md"
    assert notes.written.category == "daily"


@pytest.mark.asyncio
async def test_daily_noop_without_digest():
    stm = _STM([], summary="")
    notes = _Notes()
    r = MemoryRollup(_ProviderWithNotes(stm, notes), summarize=_summarizer({}))
    assert await r.rollup_daily(day="2026-06-22") is None
    assert notes.written is None
