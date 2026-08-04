"""Identity card + pinned-layer hardening (2.64.4).

Field failure (2026-08-04, prod): a 5.6k-char always-rewritten evergreen note
sat first in the pinned pool (recency sort), was admitted whole past the
budget, and the retriever then DROPPED the oversized pinned layer entirely —
the persona asked its owner's name and violated a pinned prohibition. These
tests pin the fixes: per-note cap, ledger-first ordering, truncate-not-drop,
and a never-dropped identity card with a notes fallback.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from geny_executor.memory.facts import Fact, FactLedger, LedgerState, render_identity_card
from geny_executor.memory.provider import MemoryHooks, NoteDraft, Scope
from geny_executor.memory.providers.file.provider import FileMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever as GenyMemoryRetriever


def _fact(kind: str, statement: str, fid: str = None, importance: str = "critical") -> Fact:
    return Fact(id=fid or statement[:8], kind=kind, statement=statement, importance=importance)


class TestRenderIdentityCard:
    def test_identity_and_prohibition_selected(self):
        facts = [
            _fact("identity", "사용자 이름은 장하렴, 호칭은 '사장님' 고정", "f1"),
            _fact("preference", "문어 이야기는 절대 언급하지 않는다 (금지)", "f2"),
            _fact("knowledge", "드보트 140카오스 비교 기록", "f3"),
        ]
        card = render_identity_card(facts, max_chars=600)
        assert "장하렴" in card and "문어" in card
        assert "드보트" not in card, "non-identity knowledge must stay out of the card"
        assert card.startswith("[핀 고정 사실")

    def test_empty_when_nothing_qualifies(self):
        assert render_identity_card([_fact("knowledge", "x", "k1")], max_chars=600) == ""
        assert render_identity_card([], max_chars=600) == ""

    def test_bounded_by_max_chars(self):
        facts = [_fact("identity", "사실 " * 200, f"f{i}") for i in range(10)]
        card = render_identity_card(facts, max_chars=300)
        assert len(card) <= 300


@pytest.mark.asyncio
class TestPinnedHardening:
    async def _seed(self, td: str) -> FileMemoryProvider:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        # The field shape: a huge always-newest evergreen + small identity notes.
        await p.notes().write(NoteDraft(
            title="사용자 이름과 호칭", body="사용자 이름은 장하렴. 호칭은 사장님 고정.",
            category="critical", filename="name.md", tags=["identity", "호칭"],
        ))
        await p.notes().write(NoteDraft(
            title="금지 주제", body="문어 이야기 금지.",
            category="critical", filename="taboo.md", tags=["금지주제"],
        ))
        await p.notes().write(NoteDraft(
            title="Evergreen Memory", body="줄줄이 긴 서사 " * 400,  # ~4.8k chars, NEWEST
            category="critical", filename="__evergreen__.md", tags=["evergreen"],
        ))
        return p

    async def test_per_note_cap_keeps_small_facts_in(self):
        with tempfile.TemporaryDirectory() as td:
            p = await self._seed(td)
            pinned = await p.notes().load_pinned(category="critical", max_chars=2400)
            assert len(pinned) <= 2400, "load_pinned must never return over budget"
            assert "장하렴" in pinned, "identity note starved out by the oversized evergreen"
            assert "문어" in pinned, "prohibition note starved out"

    async def test_ledger_note_sorts_first(self):
        with tempfile.TemporaryDirectory() as td:
            p = await self._seed(td)
            state = LedgerState(facts=[_fact("identity", "레저사실: 장하렴", "L1")])
            assert await FactLedger(p).save(state)
            pinned = await p.notes().load_pinned(category="critical", max_chars=2400)
            assert pinned.find("레저사실") < pinned.find("줄줄이"), "__facts__.md must lead"

    async def test_retriever_truncates_pinned_instead_of_dropping(self):
        with tempfile.TemporaryDirectory() as td:
            p = await self._seed(td)
            hooks = MemoryHooks(max_inject_chars=1200, recent_turns=0, identity_card_chars=0)
            r = GenyMemoryRetriever(p, hooks=hooks)
            chunks, total = [], 1100  # nearly exhausted budget
            total = await r._load_pinned_facts(chunks, total, 1200, hooks)
            layers = [(c.metadata or {}).get("layer") for c in chunks]
            assert "pinned" in layers, "pinned layer must be truncated, never dropped"
            body = chunks[-1].content
            assert len(body) <= 100 + 20  # room (100) + marker slack


@pytest.mark.asyncio
class TestIdentityCardLayer:
    async def test_card_from_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
            await p.initialize()
            await FactLedger(p).save(LedgerState(facts=[
                _fact("identity", "사용자 이름은 장하렴, 호칭 '사장님'", "f1"),
            ]))
            hooks = MemoryHooks(identity_card_chars=600)
            r = GenyMemoryRetriever(p, hooks=hooks)
            card = await r._build_identity_card(hooks)
            assert "장하렴" in card

    async def test_card_falls_back_to_tagged_notes_when_ledger_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
            await p.initialize()
            await p.notes().write(NoteDraft(
                title="사용자 이름과 호칭", body="이름 장하렴, 호칭 사장님 절대 고정.",
                category="critical", filename="name.md", tags=["identity"],
            ))
            hooks = MemoryHooks(identity_card_chars=600)
            r = GenyMemoryRetriever(p, hooks=hooks)
            card = await r._build_identity_card(hooks)
            assert "장하렴" in card, "empty ledger must fall back to identity-tagged notes"

    async def test_card_survives_zero_budget_pressure(self):
        with tempfile.TemporaryDirectory() as td:
            p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
            await p.initialize()
            await FactLedger(p).save(LedgerState(facts=[
                _fact("identity", "사용자 이름은 장하렴", "f1"),
            ]))
            hooks = MemoryHooks(max_inject_chars=100, identity_card_chars=600)
            r = GenyMemoryRetriever(p, hooks=hooks)
            chunks = []
            await r._load_identity_card(chunks, total=10_000, hooks=hooks)
            assert chunks and chunks[0].key == "identity_card", (
                "identity card must inject even when the shared budget is exhausted"
            )


@pytest.mark.asyncio
async def test_search_layers_exclude_configured_categories():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        await p.notes().write(NoteDraft(
            title="관측 소음", body="드보트 화면 관측 로그", category="observations", filename="obs.md",
        ))
        await p.notes().write(NoteDraft(
            title="진짜 기록", body="드보트 140카오스 비교", category="topics", filename="real.md",
        ))
        hooks = MemoryHooks(search_exclude_categories=("observations",))
        r = GenyMemoryRetriever(p, hooks=hooks)
        pf = await r._prefetch_layers("드보트", hooks)
        kw = pf.get("kw_notes") or []
        cats = {(c.metadata or {}).get("category") for c in kw}
        assert "observations" not in cats, "excluded category leaked into kw layer"
        assert "topics" in cats, "legit results must survive the filter"
