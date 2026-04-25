"""AgentTool tests (PR-A.1.4)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import AgentTool, BUILT_IN_TOOL_CLASSES


class _FakeOrch:
    def __init__(self, response: Any = "ok", *, raise_exc: Optional[Exception] = None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_call = None

    async def run_subagent(self, subagent_type: str, prompt: str, *, model=None):
        self.last_call = (subagent_type, prompt, model)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _ctx_with(**extras) -> ToolContext:
    return ToolContext(extras=extras)


# ── Registry membership ──────────────────────────────────────────────


def test_registered_in_builtin_classes():
    assert "Agent" in BUILT_IN_TOOL_CLASSES
    assert BUILT_IN_TOOL_CLASSES["Agent"] is AgentTool


def test_input_schema_requires_subagent_type_and_prompt():
    schema = AgentTool().input_schema
    assert set(schema["required"]) == {"subagent_type", "prompt"}
    props = schema["properties"]
    assert "model" in props


# ── Happy path ───────────────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_dispatches_to_orchestrator(self):
        orch = _FakeOrch(response="hello world")
        ctx = _ctx_with(agent_orchestrator=orch)
        result = await AgentTool().execute(
            {"subagent_type": "researcher", "prompt": "go"}, ctx,
        )
        assert result.is_error is False
        assert result.content["subagent_type"] == "researcher"
        assert result.content["result"] == "hello world"
        assert orch.last_call == ("researcher", "go", None)

    @pytest.mark.asyncio
    async def test_passes_model_override(self):
        orch = _FakeOrch()
        ctx = _ctx_with(agent_orchestrator=orch)
        await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y", "model": "claude-haiku"}, ctx,
        )
        assert orch.last_call[2] == "claude-haiku"

    @pytest.mark.asyncio
    async def test_serializes_dict_response(self):
        orch = _FakeOrch(response={"score": 0.9})
        ctx = _ctx_with(agent_orchestrator=orch)
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.content["result"] == {"score": 0.9}


# ── Error paths ──────────────────────────────────────────────────────


class TestErrors:
    @pytest.mark.asyncio
    async def test_no_orchestrator_returns_structured_error(self):
        ctx = _ctx_with()  # no extras
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_ORCHESTRATOR"

    @pytest.mark.asyncio
    async def test_missing_subagent_type(self):
        ctx = _ctx_with(agent_orchestrator=_FakeOrch())
        result = await AgentTool().execute({"prompt": "y"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "BAD_INPUT"

    @pytest.mark.asyncio
    async def test_missing_prompt(self):
        ctx = _ctx_with(agent_orchestrator=_FakeOrch())
        result = await AgentTool().execute({"subagent_type": "x"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "BAD_INPUT"

    @pytest.mark.asyncio
    async def test_unknown_subagent_type(self):
        orch = _FakeOrch(raise_exc=KeyError("ghost"))
        ctx = _ctx_with(agent_orchestrator=orch)
        result = await AgentTool().execute(
            {"subagent_type": "ghost", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "UNKNOWN_TYPE"

    @pytest.mark.asyncio
    async def test_orchestrator_failure(self):
        orch = _FakeOrch(raise_exc=RuntimeError("boom"))
        ctx = _ctx_with(agent_orchestrator=orch)
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "SUBAGENT_FAILED"
        assert "boom" in result.content["error"]["message"]

    @pytest.mark.asyncio
    async def test_orchestrator_without_run_subagent_or_spawn(self):
        class _Empty:
            pass

        ctx = _ctx_with(agent_orchestrator=_Empty())
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "ORCHESTRATOR_API"


# ── Recursion guard ──────────────────────────────────────────────────


class TestRecursion:
    @pytest.mark.asyncio
    async def test_max_depth_reached_refuses(self):
        ctx = _ctx_with(
            agent_orchestrator=_FakeOrch(),
            agent_depth=3,
            agent_max_depth=3,
        )
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "MAX_DEPTH"

    @pytest.mark.asyncio
    async def test_below_max_depth_proceeds(self):
        orch = _FakeOrch(response="ok")
        ctx = _ctx_with(
            agent_orchestrator=orch,
            agent_depth=2,
            agent_max_depth=3,
        )
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_default_max_depth_when_unset(self):
        orch = _FakeOrch(response="ok")
        ctx = _ctx_with(agent_orchestrator=orch)  # no depth set
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_custom_max_depth_respected(self):
        ctx = _ctx_with(
            agent_orchestrator=_FakeOrch(),
            agent_depth=1,
            agent_max_depth=1,
        )
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "MAX_DEPTH"


# ── ``spawn`` fallback ───────────────────────────────────────────────


class TestSpawnFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_spawn_when_run_subagent_absent(self):
        class _SpawnOrch:
            async def spawn(self, subagent_type, prompt, *, model=None):
                return f"via_spawn:{subagent_type}"

        ctx = _ctx_with(agent_orchestrator=_SpawnOrch())
        result = await AgentTool().execute(
            {"subagent_type": "x", "prompt": "y"}, ctx,
        )
        assert result.is_error is False
        assert result.content["result"] == "via_spawn:x"
