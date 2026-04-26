"""HookRunner in-process handler tests (PR-B.1.1)."""

from __future__ import annotations

import os
from typing import List

import pytest

from geny_executor.hooks.config import HookConfig, HookConfigEntry
from geny_executor.hooks.events import HookEvent, HookEventPayload, HookOutcome
from geny_executor.hooks.runner import HookRunner


def _runner(*, enabled: bool = True) -> HookRunner:
    cfg = HookConfig(enabled=enabled, entries={}, audit_log_path=None)
    env = {"GENY_ALLOW_HOOKS": "1"} if enabled else {}
    return HookRunner(cfg, env=env)


def _payload() -> HookEventPayload:
    return HookEventPayload(
        event=HookEvent.PRE_TOOL_USE,
        session_id="s1",
        timestamp="2026-04-26T00:00:00Z",
        tool_name="test",
    )


# ── Registration / deregister ────────────────────────────────────────


class TestRegistration:
    def test_register_returns_deregister_fn(self):
        runner = _runner()
        async def h(p):
            return None
        deregister = runner.register_in_process(HookEvent.PRE_TOOL_USE, h)
        assert callable(deregister)
        deregister()
        assert runner.list_in_process_handlers() == {HookEvent.PRE_TOOL_USE: 0}

    def test_multiple_handlers_per_event(self):
        runner = _runner()
        runner.register_in_process(HookEvent.PRE_TOOL_USE, lambda p: None)
        runner.register_in_process(HookEvent.PRE_TOOL_USE, lambda p: None)
        runner.register_in_process(HookEvent.POST_TOOL_USE, lambda p: None)
        snap = runner.list_in_process_handlers()
        assert snap[HookEvent.PRE_TOOL_USE] == 2
        assert snap[HookEvent.POST_TOOL_USE] == 1


# ── Execution ────────────────────────────────────────────────────────


class TestFire:
    @pytest.mark.asyncio
    async def test_handler_runs_when_enabled(self):
        runner = _runner()
        called: List = []

        async def handler(payload):
            called.append(payload)
            return None

        runner.register_in_process(HookEvent.PRE_TOOL_USE, handler)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is False
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_sync_handler_supported(self):
        runner = _runner()
        called: List = []

        def handler(payload):
            called.append(payload)
            return None

        runner.register_in_process(HookEvent.PRE_TOOL_USE, handler)
        await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_blocked_short_circuits_subsequent_handlers(self):
        runner = _runner()
        calls: List[str] = []

        async def first(payload):
            calls.append("first")
            return HookOutcome.block("denied")

        async def second(payload):
            calls.append("second")
            return None

        runner.register_in_process(HookEvent.PRE_TOOL_USE, first)
        runner.register_in_process(HookEvent.PRE_TOOL_USE, second)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is True
        assert calls == ["first"]

    @pytest.mark.asyncio
    async def test_handler_exception_isolated(self):
        runner = _runner()
        calls: List[str] = []

        async def boom(payload):
            calls.append("boom")
            raise RuntimeError("test")

        async def survivor(payload):
            calls.append("survivor")
            return None

        runner.register_in_process(HookEvent.PRE_TOOL_USE, boom)
        runner.register_in_process(HookEvent.PRE_TOOL_USE, survivor)
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is False  # exceptions don't block
        assert calls == ["boom", "survivor"]

    @pytest.mark.asyncio
    async def test_disabled_runner_skips_handlers(self):
        runner = _runner(enabled=False)
        called: List = []
        runner.register_in_process(
            HookEvent.PRE_TOOL_USE, lambda p: called.append(1) or None,
        )
        await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert called == []  # disabled → handlers not called

    @pytest.mark.asyncio
    async def test_no_handlers_returns_passthrough(self):
        runner = _runner()
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is False

    @pytest.mark.asyncio
    async def test_in_process_runs_before_subprocess(self):
        # When in-process blocks, subprocess (which would have had no
        # entries anyway) is not consulted. Confirm short-circuit by
        # registering a blocking handler + asserting outcome.
        runner = _runner()
        runner.register_in_process(
            HookEvent.PRE_TOOL_USE,
            lambda p: HookOutcome.block("denied"),
        )
        outcome = await runner.fire(HookEvent.PRE_TOOL_USE, _payload())
        assert outcome.blocked is True
        assert outcome.stop_reason == "denied"
