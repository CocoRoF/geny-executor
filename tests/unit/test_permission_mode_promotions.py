"""ACCEPT_EDITS / DONT_ASK promotion tests (PR-B.5.1)."""

from __future__ import annotations

from typing import Any, Callable, Dict

import pytest

from geny_executor.permission import (
    PermissionBehavior,
    PermissionMode,
    PermissionRule,
    PermissionSource,
    evaluate_permission,
)


class _FakeTool:
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def prepare_permission_matcher(self, inp):
        return lambda pattern: True


def _ask_rule(tool="*", source=PermissionSource.PROJECT):
    return PermissionRule(
        tool_name=tool,
        behavior=PermissionBehavior.ASK,
        source=source,
        reason="ask by default",
    )


def _deny_rule(tool="*", source=PermissionSource.PROJECT):
    return PermissionRule(
        tool_name=tool,
        behavior=PermissionBehavior.DENY,
        source=source,
        reason="forbidden",
    )


# ── ACCEPT_EDITS ─────────────────────────────────────────────────────


class TestAcceptEdits:
    @pytest.mark.asyncio
    async def test_promotes_ask_on_write(self):
        d = await evaluate_permission(
            tool=_FakeTool("Write"),
            tool_input={"file_path": "/x.txt"},
            rules=[_ask_rule("Write")],
            mode=PermissionMode.ACCEPT_EDITS,
        )
        assert d.behavior == PermissionBehavior.ALLOW
        assert "promoted" in (d.reason or "")

    @pytest.mark.asyncio
    async def test_promotes_ask_on_edit(self):
        d = await evaluate_permission(
            tool=_FakeTool("Edit"),
            tool_input={},
            rules=[_ask_rule()],
            mode=PermissionMode.ACCEPT_EDITS,
        )
        assert d.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_does_not_promote_non_edit_tool(self):
        d = await evaluate_permission(
            tool=_FakeTool("Bash"),
            tool_input={},
            rules=[_ask_rule()],
            mode=PermissionMode.ACCEPT_EDITS,
        )
        assert d.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_deny_unchanged(self):
        d = await evaluate_permission(
            tool=_FakeTool("Write"),
            tool_input={},
            rules=[_deny_rule("Write")],
            mode=PermissionMode.ACCEPT_EDITS,
        )
        assert d.behavior == PermissionBehavior.DENY


# ── DONT_ASK ─────────────────────────────────────────────────────────


class TestDontAsk:
    @pytest.mark.asyncio
    async def test_promotes_ask_on_any_tool(self):
        d = await evaluate_permission(
            tool=_FakeTool("Bash"),
            tool_input={},
            rules=[_ask_rule()],
            mode=PermissionMode.DONT_ASK,
        )
        assert d.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_deny_unchanged(self):
        d = await evaluate_permission(
            tool=_FakeTool("Bash"),
            tool_input={},
            rules=[_deny_rule()],
            mode=PermissionMode.DONT_ASK,
        )
        assert d.behavior == PermissionBehavior.DENY


# ── No mode → no promotion ───────────────────────────────────────────


class TestDefaultUnchanged:
    @pytest.mark.asyncio
    async def test_default_mode_keeps_ask(self):
        d = await evaluate_permission(
            tool=_FakeTool("Write"),
            tool_input={},
            rules=[_ask_rule()],
            mode=PermissionMode.DEFAULT,
        )
        assert d.behavior == PermissionBehavior.ASK


# ── Enum membership ──────────────────────────────────────────────────


def test_new_modes_in_enum():
    assert PermissionMode.ACCEPT_EDITS.value == "acceptEdits"
    assert PermissionMode.DONT_ASK.value == "dontAsk"
