"""Phase 2 Week 4 Checkpoint 3 — tool lifecycle hook tests.

The ``RegistryRouter`` is the single dispatch path through which every
Stage 10 executor runs a tool. These tests confirm that ``on_enter`` /
``on_exit`` / ``on_error`` are fired around ``execute`` with the right
arguments, in the right order, and — crucially — that a misbehaving hook
never turns a successful tool call into a failure.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.errors import ToolErrorCode, ToolFailure
from geny_executor.tools.registry import ToolRegistry


class _InstrumentedTool(Tool):
    """Tool that logs every lifecycle event into a shared trace list."""

    def __init__(
        self,
        name: str,
        trace: List[str],
        *,
        mode: str = "ok",
        on_enter_error: bool = False,
        on_exit_error: bool = False,
        on_error_error: bool = False,
    ):
        self._name = name
        self._trace = trace
        self._mode = mode  # "ok" | "fail" | "crash" | "soft_error"
        self._enter_error = on_enter_error
        self._exit_error = on_exit_error
        self._on_error_error = on_error_error

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "instrumented"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input, context):
        self._trace.append(f"execute({self._mode})")
        if self._mode == "fail":
            raise ToolFailure(
                "controlled failure",
                code=ToolErrorCode.TRANSPORT,
                details={"why": "test"},
            )
        if self._mode == "crash":
            raise RuntimeError("boom")
        if self._mode == "soft_error":
            return ToolResult(content="oops", is_error=True)
        return ToolResult(content="ok")

    async def on_enter(self, input, context):
        self._trace.append("on_enter")
        if self._enter_error:
            raise RuntimeError("enter hook broken")

    async def on_exit(self, result, context):
        err_flag = getattr(result, "is_error", False)
        self._trace.append(f"on_exit(is_error={err_flag})")
        if self._exit_error:
            raise RuntimeError("exit hook broken")

    async def on_error(self, error, context):
        self._trace.append(f"on_error({type(error).__name__})")
        if self._on_error_error:
            raise RuntimeError("error hook broken")


def _router_with(tool: Tool) -> RegistryRouter:
    reg = ToolRegistry()
    reg.register(tool)
    return RegistryRouter(reg)


def _ctx() -> ToolContext:
    return ToolContext(session_id="t", working_dir="")


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_enter_execute_exit_order(self):
        trace: List[str] = []
        tool = _InstrumentedTool("ok", trace, mode="ok")
        router = _router_with(tool)

        result = await router.route("ok", {}, _ctx())

        assert not result.is_error
        assert trace == ["on_enter", "execute(ok)", "on_exit(is_error=False)"]

    @pytest.mark.asyncio
    async def test_on_exit_sees_soft_error_result(self):
        """A tool returning is_error=True is still a 'normal' return —
        on_exit fires and observes the flag; on_error does NOT fire."""
        trace: List[str] = []
        tool = _InstrumentedTool("soft", trace, mode="soft_error")
        router = _router_with(tool)

        result = await router.route("soft", {}, _ctx())

        assert result.is_error is True
        assert trace == [
            "on_enter",
            "execute(soft_error)",
            "on_exit(is_error=True)",
        ]
        assert "on_error(RuntimeError)" not in trace


# ─────────────────────────────────────────────────────────────────
# Failure paths
# ─────────────────────────────────────────────────────────────────


class TestFailurePath:
    @pytest.mark.asyncio
    async def test_tool_failure_triggers_on_error(self):
        trace: List[str] = []
        tool = _InstrumentedTool("fail", trace, mode="fail")
        router = _router_with(tool)

        result = await router.route("fail", {}, _ctx())

        assert result.is_error
        assert trace == [
            "on_enter",
            "execute(fail)",
            "on_error(ToolFailure)",
        ]
        # Structured error code preserved through the router
        assert isinstance(result.content, dict)
        assert result.content.get("error", {}).get("code") == "transport_error"

    @pytest.mark.asyncio
    async def test_unexpected_exception_triggers_on_error(self):
        trace: List[str] = []
        tool = _InstrumentedTool("crash", trace, mode="crash")
        router = _router_with(tool)

        result = await router.route("crash", {}, _ctx())

        assert result.is_error
        assert trace == [
            "on_enter",
            "execute(crash)",
            "on_error(RuntimeError)",
        ]
        assert result.content["error"]["code"] == "tool_crashed"


# ─────────────────────────────────────────────────────────────────
# Hook robustness — a bad hook never breaks the tool call
# ─────────────────────────────────────────────────────────────────


class TestHookRobustness:
    @pytest.mark.asyncio
    async def test_on_enter_crash_does_not_block_execute(self, caplog):
        trace: List[str] = []
        tool = _InstrumentedTool("boom_enter", trace, mode="ok", on_enter_error=True)
        router = _router_with(tool)
        caplog.set_level("WARNING")

        result = await router.route("boom_enter", {}, _ctx())

        assert not result.is_error
        # execute still runs + on_exit still fires
        assert "execute(ok)" in trace
        assert any("on_exit" in e for e in trace)
        assert any("on_enter" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_on_exit_crash_does_not_corrupt_result(self, caplog):
        trace: List[str] = []
        tool = _InstrumentedTool("boom_exit", trace, mode="ok", on_exit_error=True)
        router = _router_with(tool)
        caplog.set_level("WARNING")

        result = await router.route("boom_exit", {}, _ctx())

        # Tool returned a clean result — exit hook fail must NOT flip it
        assert result.is_error is False
        assert result.content == "ok"
        assert any("on_exit" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_on_error_crash_does_not_mask_tool_error(self, caplog):
        trace: List[str] = []
        tool = _InstrumentedTool(
            "boom_err", trace, mode="crash", on_error_error=True
        )
        router = _router_with(tool)
        caplog.set_level("WARNING")

        result = await router.route("boom_err", {}, _ctx())

        # The *tool's* failure is what the caller sees — hook crash
        # just gets logged.
        assert result.is_error
        assert result.content["error"]["code"] == "tool_crashed"
        assert any("on_error" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────
# Short-circuit — invalid input bypasses all hooks
# ─────────────────────────────────────────────────────────────────


class TestShortCircuits:
    @pytest.mark.asyncio
    async def test_unknown_tool_fires_no_hooks(self):
        """Unknown tool returns early — no hooks to fire, and no crash."""
        router = RegistryRouter(ToolRegistry())
        result = await router.route("nope", {}, _ctx())
        assert result.is_error
        assert result.content["error"]["code"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_invalid_input_skips_hooks(self):
        """Schema validation fails before any hook fires."""
        trace: List[str] = []

        class _StrictTool(_InstrumentedTool):
            @property
            def input_schema(self):
                return {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                }

        tool = _StrictTool("strict", trace, mode="ok")
        router = _router_with(tool)

        result = await router.route("strict", {}, _ctx())

        assert result.is_error
        assert result.content["error"]["code"] == "invalid_input"
        # Hooks must NOT have been fired — input was rejected upstream
        assert trace == []
