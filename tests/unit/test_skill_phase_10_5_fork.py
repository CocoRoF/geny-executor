"""Phase 10.5 — fork execution mode."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from geny_executor.skills.fork import (
    ForkResult,
    SkillForkRunner,
    make_default_fork_runner,
)
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.skill_tool import (
    SkillTool,
    SkillToolProvider,
    build_skill_tool,
)
from geny_executor.skills.types import Skill, SkillMetadata
from geny_executor.tools.base import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(
        session_id="s",
        working_dir="/tmp",
        permission_rules=[],
    )


def _fork_skill(
    *,
    skill_id: str = "fork-skill",
    body: str = "Do the thing.",
    model_override: str = None,
    allowed_tools: tuple = (),
) -> Skill:
    return Skill(
        id=skill_id,
        metadata=SkillMetadata(
            name=skill_id,
            description="forked",
            execution_mode="fork",
            model_override=model_override,
            allowed_tools=allowed_tools,
        ),
        body=body,
    )


# ── No runner wired → clean error ────────────────────────────────────


@pytest.mark.asyncio
async def test_fork_without_runner_returns_clear_error() -> None:
    tool = SkillTool(_fork_skill())
    res = await tool.execute({"args": {}}, _ctx())
    assert res.is_error is True
    assert "SkillForkRunner" in res.content
    assert "execution_mode='inline'" in res.content


# ── Runner is invoked + result forwarded ─────────────────────────────


@pytest.mark.asyncio
async def test_fork_runner_is_invoked_with_correct_inputs() -> None:
    captured: Dict[str, Any] = {}

    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        captured["skill_id"] = skill.id
        captured["rendered_body"] = rendered_body
        captured["invoke_args"] = invoke_args
        captured["parent_session_id"] = parent_context.session_id
        return ForkResult(content="fork said hi", metadata={"answer": 42})

    tool = SkillTool(_fork_skill(body="Do X"), fork_runner=runner)
    res = await tool.execute({"args": {"foo": "bar"}}, _ctx())

    assert res.is_error is False
    assert res.content == "fork said hi"
    assert captured["skill_id"] == "fork-skill"
    assert "Do X" in captured["rendered_body"]
    assert captured["invoke_args"] == {"foo": "bar"}
    assert captured["parent_session_id"] == "s"


@pytest.mark.asyncio
async def test_fork_runner_metadata_merged_with_defaults() -> None:
    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        return ForkResult(
            content="ok",
            metadata={"custom_field": "value"},
        )

    skill = _fork_skill(model_override="claude-opus-4-7")
    tool = SkillTool(skill, fork_runner=runner)
    res = await tool.execute({"args": {}}, _ctx())

    # Custom metadata kept
    assert res.metadata["custom_field"] == "value"
    # Default fields filled in by SkillTool wrapper
    assert res.metadata["skill_id"] == "fork-skill"
    assert res.metadata["execution_mode"] == "fork"
    assert res.metadata["model_override"] == "claude-opus-4-7"
    assert res.metadata["args"] == {}


@pytest.mark.asyncio
async def test_fork_runner_can_set_is_error() -> None:
    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        return ForkResult(content="something failed", is_error=True)

    tool = SkillTool(_fork_skill(), fork_runner=runner)
    res = await tool.execute({"args": {}}, _ctx())
    assert res.is_error is True
    assert res.content == "something failed"


# ── Body still goes through ${name} substitution before fork ─────────


@pytest.mark.asyncio
async def test_fork_runner_receives_substituted_body() -> None:
    """Args are substituted into the body *before* the runner sees it,
    so a fork runner that uses the body as system prompt gets the
    operator's intent woven in."""
    captured: Dict[str, Any] = {}

    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        captured["body"] = rendered_body
        return ForkResult(content="ok")

    skill = Skill(
        id="echo-fork",
        metadata=SkillMetadata(
            name="e",
            description="e",
            execution_mode="fork",
            arguments=("name",),
        ),
        body="Hello ${name}, please proceed.",
    )
    tool = SkillTool(skill, fork_runner=runner)
    await tool.execute({"args": {"name": "Operator"}}, _ctx())
    assert "Hello Operator, please proceed." == captured["body"]


# ── Runner exception → clean ToolResult error ────────────────────────


@pytest.mark.asyncio
async def test_fork_runner_exception_becomes_tool_error() -> None:
    """A runner that raises should not crash the parent — convert to
    a structured error result."""

    async def boom(*, skill, rendered_body, invoke_args, parent_context):
        raise RuntimeError("network exploded")

    tool = SkillTool(_fork_skill(), fork_runner=boom)
    res = await tool.execute({"args": {}}, _ctx())
    assert res.is_error is True
    assert "network exploded" in res.content
    assert res.metadata["fork_runner_error"] == "network exploded"


# ── SkillToolProvider propagates the runner ──────────────────────────


@pytest.mark.asyncio
async def test_provider_propagates_fork_runner_to_tools() -> None:
    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        return ForkResult(content=f"ran {skill.id}")

    registry = SkillRegistry()
    registry.register(_fork_skill(skill_id="alpha"))
    registry.register(_fork_skill(skill_id="beta"))
    provider = SkillToolProvider(registry, fork_runner=runner)
    tools = provider.list_tools()
    assert len(tools) == 2

    res = await tools[0].execute({"args": {}}, _ctx())
    assert "ran" in res.content


def test_build_skill_tool_accepts_fork_runner() -> None:
    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        return ForkResult(content="ok")

    skill = _fork_skill()
    tool = build_skill_tool(skill, fork_runner=runner)
    assert tool._fork_runner is runner  # type: ignore[attr-defined]


# ── Inline mode unaffected ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_fork_runner_not_called_for_inline_skills() -> None:
    """Wiring a runner doesn't change behaviour for inline skills —
    the runner only fires when execution_mode == 'fork'."""
    fired: Dict[str, bool] = {"runner": False}

    async def runner(*, skill, rendered_body, invoke_args, parent_context):
        fired["runner"] = True
        return ForkResult(content="should not be reached")

    skill = Skill(
        id="inline-skill",
        metadata=SkillMetadata(
            name="i",
            description="i",
            execution_mode="inline",
        ),
        body="hi",
    )
    tool = SkillTool(skill, fork_runner=runner)
    res = await tool.execute({"args": {}}, _ctx())
    assert fired["runner"] is False
    assert "Skill invoked" in res.content  # inline mode header


# ── make_default_fork_runner ─────────────────────────────────────────


def test_default_fork_runner_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = make_default_fork_runner()
    assert runner is None


def test_default_fork_runner_built_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-12345")
    runner = make_default_fork_runner()
    assert runner is not None
    assert callable(runner)


def test_default_fork_runner_explicit_api_key_overrides_env(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = make_default_fork_runner(api_key="sk-explicit")
    assert runner is not None
