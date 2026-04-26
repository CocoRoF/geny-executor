"""Cron tools tests (PR-A.4.2).

Skips when croniter isn't installed — test_invalid_cron_expr asserts
rejection via croniter validation. CronJobStore tests live in
test_cron_store.py and don't need croniter (they pass without).
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "croniter",
    reason="cron extra not installed (pip install -e .[cron])",
)

from geny_executor.cron import (  # noqa: E402
    CronJob,
    CronJobStatus,
    InMemoryCronJobStore,
)
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
)


def test_all_three_registered():
    for name in ("CronCreate", "CronDelete", "CronList"):
        assert name in BUILT_IN_TOOL_CLASSES


def _ctx(store):
    return ToolContext(extras={"cron_store": store})


# ── CronCreate ───────────────────────────────────────────────────────


class TestCreate:
    @pytest.mark.asyncio
    async def test_creates_job(self):
        store = InMemoryCronJobStore()
        result = await CronCreateTool().execute(
            {"name": "nightly", "cron_expr": "0 3 * * *", "target_kind": "local_bash",
             "payload": {"command": "echo"}},
            _ctx(store),
        )
        assert result.is_error is False
        assert (await store.get("nightly")).cron_expr == "0 3 * * *"

    @pytest.mark.asyncio
    async def test_invalid_cron_expr(self):
        store = InMemoryCronJobStore()
        result = await CronCreateTool().execute(
            {"name": "bad", "cron_expr": "not a cron", "target_kind": "x"}, _ctx(store),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "INVALID_CRON_EXPR"

    @pytest.mark.asyncio
    async def test_duplicate_name(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        result = await CronCreateTool().execute(
            {"name": "x", "cron_expr": "* * * * *", "target_kind": "k"}, _ctx(store),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NAME_EXISTS"

    @pytest.mark.asyncio
    async def test_no_store(self):
        result = await CronCreateTool().execute(
            {"name": "x", "cron_expr": "* * * * *", "target_kind": "k"},
            ToolContext(extras={}),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_STORE"

    @pytest.mark.asyncio
    async def test_calls_runner_refresh(self):
        store = InMemoryCronJobStore()
        refreshed: list = []

        class _Runner:
            async def refresh(self):
                refreshed.append(True)

        ctx = ToolContext(extras={"cron_store": store, "cron_runner": _Runner()})
        await CronCreateTool().execute(
            {"name": "r", "cron_expr": "* * * * *", "target_kind": "k"}, ctx,
        )
        assert refreshed == [True]


# ── CronDelete ───────────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        result = await CronDeleteTool().execute({"name": "x"}, _ctx(store))
        assert result.is_error is False
        assert await store.get("x") is None

    @pytest.mark.asyncio
    async def test_not_found(self):
        result = await CronDeleteTool().execute(
            {"name": "ghost"}, _ctx(InMemoryCronJobStore()),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NOT_FOUND"


# ── CronList ─────────────────────────────────────────────────────────


class TestList:
    @pytest.mark.asyncio
    async def test_lists_all(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="a", cron_expr="* * * * *", target_kind="k"))
        await store.put(CronJob(name="b", cron_expr="* * * * *", target_kind="k"))
        result = await CronListTool().execute({}, _ctx(store))
        names = sorted(j["name"] for j in result.content["jobs"])
        assert names == ["a", "b"]

    @pytest.mark.asyncio
    async def test_only_enabled(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="a", cron_expr="* * * * *", target_kind="k"))
        await store.put(CronJob(name="b", cron_expr="* * * * *", target_kind="k",
                                status=CronJobStatus.DISABLED))
        result = await CronListTool().execute({"only_enabled": True}, _ctx(store))
        names = [j["name"] for j in result.content["jobs"]]
        assert names == ["a"]

    @pytest.mark.asyncio
    async def test_no_store(self):
        result = await CronListTool().execute({}, ToolContext(extras={}))
        assert result.is_error is True
