"""Fact Ledger (2.46.0) — durable facts as schema-bound, diff-applied records.

The contract under test: the LLM judges (via structured extraction), the
schema constrains, the code stores. A bad pass can never erase the ledger,
corrections supersede instead of delete, and everything renders
deterministically into the pinned note the retriever always injects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.memory.facts import (
    FACT_EXTRACTION_SCHEMA,
    FACTS_FILENAME,
    Fact,
    FactExtraction,
    FactLedger,
    LedgerState,
    build_fact_extraction_instruction,
    render_ledger_markdown,
)
from geny_executor.memory.rollup import (
    EVERGREEN_SCHEMA,
    MemoryRollup,
    render_evergreen,
    render_segment_digest,
)


# ── ledger diff semantics ─────────────────────────────────────────────


def _state(*facts: Fact) -> LedgerState:
    return LedgerState(facts=list(facts))


def test_upsert_adds_new_fact():
    state = _state()
    n = FactLedger.apply_diff(
        state,
        upserts=[{
            "id": "user.address_as",
            "kind": "identity",
            "statement": "사용자를 '하렴 사장님'이라고 부른다",
            "importance": "critical",
            "evidence": "하렴 사장님이라고 부르라고",
        }],
        supersedes=[],
        now_iso="2026-07-06T00:00:00+00:00",
    )
    assert n == 1
    assert state.facts[0].id == "user.address_as"
    assert state.facts[0].status == "active"
    assert state.facts[0].created == "2026-07-06T00:00:00+00:00"


def test_correction_updates_in_place():
    state = _state(
        Fact(id="user.address_as", kind="identity",
             statement="사용자를 '하렴'이라고 부른다", importance="critical",
             created="t0", updated="t0"),
    )
    n = FactLedger.apply_diff(
        state,
        upserts=[{
            "id": "user.address_as",
            "kind": "identity",
            "statement": "사용자를 '하렴 사장님'이라고 부른다",
            "importance": "critical",
        }],
        supersedes=[],
        now_iso="t1",
    )
    assert n == 1
    assert len(state.facts) == 1  # same id → in-place, no duplicate
    assert "사장님" in state.facts[0].statement
    assert state.facts[0].created == "t0"  # provenance preserved
    assert state.facts[0].updated == "t1"


def test_supersede_keeps_the_record():
    state = _state(
        Fact(id="commitment.demo", kind="commitment",
             statement="금요일에 데모", importance="high"),
    )
    n = FactLedger.apply_diff(state, upserts=[], supersedes=["commitment.demo"],
                              now_iso="t1")
    assert n == 1
    assert state.facts[0].status == "superseded"  # kept, never deleted


def test_unknown_supersede_ignored_and_noop_upsert_stable():
    state = _state(
        Fact(id="user.name", kind="identity", statement="이름은 하렴",
             importance="critical", updated="t0"),
    )
    n = FactLedger.apply_diff(
        state,
        upserts=[{"id": "user.name", "kind": "identity",
                  "statement": "이름은 하렴", "importance": "critical"}],
        supersedes=["ghost.id"],
        now_iso="t1",
    )
    assert n == 0
    assert state.facts[0].updated == "t0"  # identical upsert → timestamps stable


def test_malformed_upsert_rows_skipped():
    state = _state()
    n = FactLedger.apply_diff(
        state,
        upserts=[{"kind": "identity"}, {"id": "", "statement": "x"}, "junk"],  # type: ignore[list-item]
        supersedes=[],
        now_iso="t",
    )
    assert n == 0 and state.facts == []


# ── rendering ─────────────────────────────────────────────────────────


def test_render_groups_by_kind_and_hides_superseded():
    md = render_ledger_markdown([
        Fact(id="user.address_as", kind="identity",
             statement="'하렴 사장님'이라고 부른다", importance="critical"),
        Fact(id="pref.tone", kind="preference", statement="반말 금지",
             importance="high"),
        Fact(id="old", kind="identity", statement="'하렴'이라고 부른다",
             importance="critical", status="superseded"),
    ])
    assert "## Identity" in md and "## Preferences" in md
    assert "'하렴 사장님'이라고 부른다" in md
    assert "'하렴'이라고 부른다" not in md.split("superseded")[0]
    assert "1 superseded" in md


def test_render_empty_ledger():
    assert "no durable facts" in render_ledger_markdown([])


# ── extraction pass (fake provider + fake structured LLM) ────────────


@dataclass
class _Turn:
    role: str
    content: str
    timestamp: datetime


class _FakeSTM:
    def __init__(self, turns):
        self._turns = turns

    async def recent(self, n=100):
        return self._turns[-n:]


class _FakeNotes:
    def __init__(self):
        self.store: Dict[str, Any] = {}

    async def read(self, filename):
        return self.store.get(filename)

    async def write(self, draft):
        @dataclass
        class _Note:
            frontmatter: Dict[str, Any]
            body: str
            title: str

        self.store[draft.filename] = _Note(
            frontmatter=dict(draft.frontmatter), body=draft.body,
            title=draft.title,
        )
        return object()


class _FakeProvider:
    def __init__(self, turns):
        self._stm = _FakeSTM(turns)
        self._notes = _FakeNotes()

    def stm(self):
        return self._stm

    def notes(self):
        return self._notes


def _turns_with_name(base: datetime) -> List[_Turn]:
    return [
        _Turn("user", "아니 하렴이 아니라 하렴 사장님이라고 부르라고", base),
        _Turn("assistant", "아, 하렴 사장님. 이제 제대로 부를게.",
              base + timedelta(seconds=5)),
    ]


@pytest.mark.asyncio
async def test_extraction_applies_diff_and_advances_cursor():
    base = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    provider = _FakeProvider(_turns_with_name(base))

    captured: Dict[str, Any] = {}

    async def fake_structured(instruction: str, schema: Dict[str, Any]):
        captured["instruction"] = instruction
        captured["schema"] = schema
        return {
            "upserts": [{
                "id": "user.address_as",
                "kind": "identity",
                "statement": "사용자를 '하렴 사장님'이라고 부른다",
                "importance": "critical",
                "evidence": "하렴 사장님이라고 부르라고",
            }],
            "supersedes": [],
        }

    report = await FactExtraction(
        provider, complete_structured=fake_structured,
    ).run()

    assert report.ran and report.changes == 1 and report.active_facts == 1
    assert captured["schema"] is FACT_EXTRACTION_SCHEMA
    assert "하렴 사장님이라고 부르라고" in captured["instruction"]

    note = provider._notes.store[FACTS_FILENAME]
    assert note.frontmatter["facts"][0]["id"] == "user.address_as"
    assert note.frontmatter["extraction_cursor"] >= base.isoformat()
    assert "'하렴 사장님'이라고 부른다" in note.body

    # Second pass with no new turns → skipped, ledger untouched.
    report2 = await FactExtraction(
        provider, complete_structured=fake_structured,
    ).run()
    assert not report2.ran and report2.skipped_reason == "no_new_user_turns"


@pytest.mark.asyncio
async def test_extraction_requires_user_turn():
    base = datetime(2026, 7, 6, tzinfo=timezone.utc)
    provider = _FakeProvider(
        [_Turn("assistant", "[THINKING_TRIGGER] 화면 코멘트", base)]
    )

    async def never_called(instruction, schema):  # pragma: no cover - guard
        raise AssertionError("agent monologue must not trigger extraction")

    report = await FactExtraction(
        provider, complete_structured=never_called,
    ).run()
    assert report.skipped_reason == "no_new_user_turns"


@pytest.mark.asyncio
async def test_structured_failure_leaves_ledger_untouched():
    base = datetime(2026, 7, 6, tzinfo=timezone.utc)
    provider = _FakeProvider(_turns_with_name(base))

    async def broken(instruction, schema):
        return None  # provider couldn't produce schema-bound output

    report = await FactExtraction(provider, complete_structured=broken).run()
    assert not report.ran
    assert report.skipped_reason == "no_structured_output"
    assert FACTS_FILENAME not in provider._notes.store  # nothing written


# ── rollup v2: structured contract protects existing state ───────────


class _RollupProvider(_FakeProvider):
    def __init__(self, turns, summary=""):
        super().__init__(turns)
        self._turns = turns
        self._summary = summary
        self.written: Optional[str] = None

    def stm(self):
        outer = self

        class _STM(_FakeSTM):
            async def read_summary(self):
                return outer._summary

            async def write_summary(self, body):
                outer.written = body

        return _STM(self._turns)


@pytest.mark.asyncio
async def test_structured_segment_renders_and_persists():
    base = datetime(2026, 7, 6, tzinfo=timezone.utc)
    provider = _RollupProvider(_turns_with_name(base), summary="prior")

    async def structured(instruction, schema):
        return {
            "summary": "호칭 정정이 있었던 세션.",
            "facts_decisions": ["호칭은 '하렴 사장님'"],
            "entities": [],
            "preferences_commitments": [],
            "open_threads": [],
            "relationship_mood": [],
        }

    async def legacy(_):  # pragma: no cover - structured path must win
        raise AssertionError("legacy summarize must not run in structured mode")

    rollup = MemoryRollup(
        provider, summarize=legacy, complete_structured=structured,
    )
    digest = await rollup.summarize_segment()
    assert digest and "## Summary" in digest and "하렴 사장님" in digest
    assert provider.written == digest


@pytest.mark.asyncio
async def test_structured_violation_keeps_previous_digest():
    base = datetime(2026, 7, 6, tzinfo=timezone.utc)
    provider = _RollupProvider(_turns_with_name(base), summary="GOOD OLD DIGEST")

    async def assistant_reply(instruction, schema):
        return None  # e.g. model answered conversationally → parse failed

    async def legacy(_):  # pragma: no cover
        raise AssertionError("must not fall back to freeform")

    rollup = MemoryRollup(
        provider, summarize=legacy, complete_structured=assistant_reply,
    )
    assert await rollup.summarize_segment() is None
    assert provider.written is None  # previous digest untouched


def test_renderers_are_deterministic_and_skip_empty_sections():
    md = render_evergreen({
        "identity": ["에이전트 이름은 엘렌"],
        "user": ["사용자는 '하렴 사장님'"],
        "durable_facts": [],
        "preferences_commitments": [],
        "long_running_threads": [],
    })
    assert md.count("##") == 2
    assert render_segment_digest({"summary": "", "facts_decisions": [],
                                  "entities": [], "preferences_commitments": [],
                                  "open_threads": [],
                                  "relationship_mood": []}) == ""
    assert EVERGREEN_SCHEMA["additionalProperties"] is False


def test_instruction_mentions_ledger_and_rules():
    text = build_fact_extraction_instruction(
        active_facts="- user.name: 이름은 하렴", new_turns="[user] 안녕",
    )
    assert "user.name" in text
    assert "supersedes" in text
    assert "ONLY JSON" in text


# ── APIResponse.structured surface ───────────────────────────────────


def test_api_response_structured_reads_cli_envelope():
    from geny_executor.llm_client.types import APIResponse

    resp = APIResponse(raw={"structured_output": {"name": "BOSS"}, "result": "chat"})
    assert resp.structured == {"name": "BOSS"}
    assert APIResponse(raw=None).structured is None
    assert APIResponse(raw={"result": "x"}).structured is None
