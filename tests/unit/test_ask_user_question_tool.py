"""AskUserQuestionTool tests (PR-A.3.1)."""

from __future__ import annotations

import asyncio

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    AskUserQuestionTool,
    BUILT_IN_TOOL_CLASSES,
    QuestionCancelled,
)


# ── Stubs ────────────────────────────────────────────────────────────


def _make_handler(answer="ok", *, raise_exc=None, sleep=None):
    async def handler(*, question, options, default, timeout_seconds, prompt_id):
        if sleep is not None:
            await asyncio.sleep(sleep)
        if raise_exc is not None:
            raise raise_exc
        return answer
    return handler


# ── Registry ─────────────────────────────────────────────────────────


def test_registered():
    assert "AskUserQuestion" in BUILT_IN_TOOL_CLASSES
    assert BUILT_IN_TOOL_CLASSES["AskUserQuestion"] is AskUserQuestionTool


# ── Happy path ───────────────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_answer(self):
        ctx = ToolContext(extras={"question_handler": _make_handler("hello")})
        result = await AskUserQuestionTool().execute({"question": "what?"}, ctx)
        assert result.is_error is False
        assert result.content["answer"] == "hello"
        assert "prompt_id" in result.content

    @pytest.mark.asyncio
    async def test_passes_options_and_default(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return "x"

        ctx = ToolContext(extras={"question_handler": handler})
        await AskUserQuestionTool().execute(
            {
                "question": "pick",
                "options": ["a", "b", "c"],
                "default": "a",
                "timeout_seconds": 30,
            },
            ctx,
        )
        assert captured["options"] == ["a", "b", "c"]
        assert captured["default"] == "a"
        assert captured["timeout_seconds"] == 30
        assert captured["prompt_id"]

    @pytest.mark.asyncio
    async def test_non_string_answer_coerced(self):
        ctx = ToolContext(extras={"question_handler": _make_handler(42)})
        result = await AskUserQuestionTool().execute({"question": "?"}, ctx)
        assert result.content["answer"] == "42"


# ── Errors ───────────────────────────────────────────────────────────


class TestErrors:
    @pytest.mark.asyncio
    async def test_no_handler(self):
        result = await AskUserQuestionTool().execute(
            {"question": "?"}, ToolContext(extras={}),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_HANDLER"

    @pytest.mark.asyncio
    async def test_missing_question(self):
        ctx = ToolContext(extras={"question_handler": _make_handler()})
        result = await AskUserQuestionTool().execute({"question": ""}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "BAD_INPUT"

    @pytest.mark.asyncio
    async def test_timeout(self):
        ctx = ToolContext(extras={"question_handler": _make_handler(sleep=10)})
        result = await AskUserQuestionTool().execute(
            {"question": "?", "timeout_seconds": 5}, ctx,
        )
        # Tool wraps handler in wait_for(timeout); the wrapper
        # itself uses the same number, so we use a tight timeout to
        # observe the fail path quickly. Asyncio will raise inside
        # the wait_for and the tool catches it.
        # Use a short timeout (5s minimum allowed by schema) to avoid
        # making the test slow; 5s is the schema floor so we can
        # allow handler to sleep well past it.
        assert result.is_error is True
        assert result.content["error"]["code"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_cancelled(self):
        ctx = ToolContext(extras={"question_handler": _make_handler(raise_exc=QuestionCancelled("dismissed"))})
        result = await AskUserQuestionTool().execute({"question": "?"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "CANCELLED"
        assert "dismissed" in result.content["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_failure(self):
        ctx = ToolContext(extras={"question_handler": _make_handler(raise_exc=RuntimeError("boom"))})
        result = await AskUserQuestionTool().execute({"question": "?"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "HANDLER_FAILED"


# ── Capabilities ─────────────────────────────────────────────────────


def test_capabilities_block_concurrent():
    caps = AskUserQuestionTool().capabilities({})
    assert caps.concurrency_safe is False
    assert caps.read_only is True
