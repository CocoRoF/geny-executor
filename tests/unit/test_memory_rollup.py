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
