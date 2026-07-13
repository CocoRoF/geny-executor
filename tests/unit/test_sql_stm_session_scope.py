"""2.53.0 — SQL STM session scoping: one database, many sessions.

A store constructed with ``session_id`` stamps + filters its rows and keeps
its summary in ``stm_summaries`` (keyed); a store without one keeps the
legacy whole-table view + singleton ``stm_summary`` row.
"""
from datetime import datetime, timezone

import pytest

from geny_executor.memory.factory import MemoryProviderFactory
from geny_executor.memory.provider import Turn


def _turn(role: str, content: str) -> Turn:
    return Turn(role=role, content=content, timestamp=datetime.now(timezone.utc))


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "mem.db")


async def _open(db_path: str, session_id: str = ""):
    cfg = {"provider": "sql", "dsn": db_path, "session_id": session_id}
    provider = MemoryProviderFactory().build(cfg)
    await provider.initialize()
    return provider


@pytest.mark.asyncio
async def test_appends_are_isolated_per_session(db_path):
    a = await _open(db_path, "sess-a")
    b = await _open(db_path, "sess-b")
    try:
        await a.stm().append(_turn("user", "alpha-only"))
        await b.stm().append(_turn("user", "beta-only"))
        await b.stm().append(_turn("assistant", "beta-reply"))

        a_recent = await a.stm().recent(n=10)
        b_recent = await b.stm().recent(n=10)
        assert [t.content for t in a_recent] == ["alpha-only"]
        assert [t.content for t in b_recent] == ["beta-only", "beta-reply"]

        # search scoped too
        assert len(await a.stm().search("beta", limit=5)) == 0
        assert len(await b.stm().search("beta", limit=5)) == 2
    finally:
        await a.close()
        await b.close()


@pytest.mark.asyncio
async def test_legacy_unscoped_store_sees_everything(db_path):
    a = await _open(db_path, "sess-a")
    legacy = await _open(db_path, "")
    try:
        await a.stm().append(_turn("user", "scoped row"))
        # legacy(미스코프) 스토어는 전체 뷰 유지 — 기존 배포 무변경 계약
        assert len(await legacy.stm().recent(n=10)) == 1
    finally:
        await a.close()
        await legacy.close()


@pytest.mark.asyncio
async def test_summaries_keyed_per_session(db_path):
    a = await _open(db_path, "sess-a")
    b = await _open(db_path, "sess-b")
    legacy = await _open(db_path, "")
    try:
        await a.stm().write_summary("digest A")
        await b.stm().write_summary("digest B")
        await legacy.stm().write_summary("digest legacy")

        assert await a.stm().read_summary() == "digest A"
        assert await b.stm().read_summary() == "digest B"
        assert await legacy.stm().read_summary() == "digest legacy"

        # 갱신 upsert
        await a.stm().write_summary("digest A2")
        assert await a.stm().read_summary() == "digest A2"
        assert await b.stm().read_summary() == "digest B"
    finally:
        await a.close()
        await b.close()
        await legacy.close()


@pytest.mark.asyncio
async def test_truncate_scoped(db_path):
    a = await _open(db_path, "sess-a")
    b = await _open(db_path, "sess-b")
    try:
        for i in range(5):
            await a.stm().append(_turn("user", f"a{i}"))
        await b.stm().append(_turn("user", "b0"))

        removed = await a.stm().truncate(keep_last=2)
        assert removed == 3
        assert [t.content for t in await a.stm().recent(n=10)] == ["a3", "a4"]
        # 다른 세션은 무손실
        assert [t.content for t in await b.stm().recent(n=10)] == ["b0"]
    finally:
        await a.close()
        await b.close()
