"""Phase 4 Week 8 — SkillTool + SkillToolProvider tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.skills import (
    Skill,
    SkillMetadata,
    SkillRegistry,
    SkillTool,
    SkillToolProvider,
    build_skill_tool,
)
from geny_executor.tools.base import ToolContext


def _skill(
    sid: str = "refactor",
    *,
    name: str = "Refactor TS",
    description: str = "Plan and execute TypeScript refactors",
    body: str = "# steps\nDo the thing\n",
    allowed_tools: tuple = (),
    model_override=None,
    execution_mode: str = "inline",
    version=None,
) -> Skill:
    return Skill(
        id=sid,
        metadata=SkillMetadata(
            name=name,
            description=description,
            version=version,
            allowed_tools=allowed_tools,
            model_override=model_override,
            execution_mode=execution_mode,
        ),
        body=body,
        source=Path("/fake/path/SKILL.md"),
    )


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="/w")


# ─────────────────────────────────────────────────────────────────
# SkillTool
# ─────────────────────────────────────────────────────────────────


class TestSkillToolBasics:
    def test_name_is_skill_id(self):
        tool = SkillTool(_skill("my-skill"))
        assert tool.name == "my-skill"

    def test_description_includes_mode_marker(self):
        tool = SkillTool(_skill(description="Does stuff", execution_mode="inline"))
        assert "Does stuff" in tool.description
        assert "[skill, inline]" in tool.description

    def test_capabilities_are_safe(self):
        caps = SkillTool(_skill()).capabilities({})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.destructive is False

    def test_schema_has_args(self):
        schema = SkillTool(_skill()).input_schema
        assert "args" in schema["properties"]
        # args required? No — optional with default {}
        assert "args" not in schema.get("required", [])

    def test_exposes_underlying_skill(self):
        s = _skill()
        tool = SkillTool(s)
        assert tool.skill is s

    def test_build_skill_tool_factory(self):
        s = _skill()
        tool = build_skill_tool(s)
        assert isinstance(tool, SkillTool)
        assert tool.skill is s


# ─────────────────────────────────────────────────────────────────
# Execute — inline mode
# ─────────────────────────────────────────────────────────────────


class TestExecuteInline:
    @pytest.mark.asyncio
    async def test_returns_skill_body(self):
        tool = SkillTool(_skill(body="step 1\nstep 2\n"))
        result = await tool.execute({}, _ctx())
        assert not result.is_error
        assert "step 1" in result.content
        assert "step 2" in result.content

    @pytest.mark.asyncio
    async def test_header_lists_skill_name_and_id(self):
        tool = SkillTool(_skill("my-skill", name="My Skill"))
        result = await tool.execute({}, _ctx())
        assert "Skill invoked: My Skill (my-skill)" in result.content

    @pytest.mark.asyncio
    async def test_header_includes_version_when_present(self):
        tool = SkillTool(_skill(version="1.2.3"))
        result = await tool.execute({}, _ctx())
        assert "version: 1.2.3" in result.content

    @pytest.mark.asyncio
    async def test_header_includes_allowed_tools(self):
        tool = SkillTool(_skill(allowed_tools=("Read", "Grep")))
        result = await tool.execute({}, _ctx())
        assert "allowed tools: Read, Grep" in result.content

    @pytest.mark.asyncio
    async def test_header_includes_model_override(self):
        tool = SkillTool(_skill(model_override="claude-opus-4-7"))
        result = await tool.execute({}, _ctx())
        assert "model override: claude-opus-4-7" in result.content

    @pytest.mark.asyncio
    async def test_metadata_payload(self):
        tool = SkillTool(
            _skill(
                "s1",
                name="N",
                allowed_tools=("Read",),
                model_override="claude-sonnet-4-6",
            )
        )
        result = await tool.execute({"args": {"x": 1}}, _ctx())
        meta = result.metadata
        assert meta["skill_id"] == "s1"
        assert meta["skill_name"] == "N"
        assert meta["execution_mode"] == "inline"
        assert meta["allowed_tools"] == ["Read"]
        assert meta["model_override"] == "claude-sonnet-4-6"
        assert meta["args"] == {"x": 1}
        assert meta["skill_context"]["session_id"] == "s"


class TestTemplateRendering:
    @pytest.mark.asyncio
    async def test_interpolates_invoke_args(self):
        tool = SkillTool(_skill(body="Hello {name}!\nEvery {fruit} is good.\n"))
        result = await tool.execute({"args": {"name": "Alice", "fruit": "apple"}}, _ctx())
        assert "Hello Alice!" in result.content
        assert "Every apple is good." in result.content
        assert result.metadata["rendered_template"] is True

    @pytest.mark.asyncio
    async def test_unknown_placeholder_kept_intact(self):
        """Skill bodies may contain example syntax using braces that
        aren't meant for interpolation — these must pass through."""
        tool = SkillTool(_skill(body="Use {unknown_ref} here.\n"))
        result = await tool.execute({"args": {}}, _ctx())
        assert "Use {unknown_ref} here." in result.content

    @pytest.mark.asyncio
    async def test_malformed_format_spec_returns_body_unchanged(self):
        """A bad format spec must not crash the tool — we fall back to
        the raw body so the LLM can still read the guidance."""
        tool = SkillTool(_skill(body="Some {bad:weird_spec} text.\n"))
        result = await tool.execute({"args": {"bad": "x"}}, _ctx())
        assert not result.is_error
        assert "Some {bad:weird_spec} text." in result.content

    @pytest.mark.asyncio
    async def test_empty_args_leaves_body_untouched(self):
        tool = SkillTool(_skill(body="static body\n"))
        result = await tool.execute({}, _ctx())
        assert result.metadata["rendered_template"] is False


class TestForkMode:
    @pytest.mark.asyncio
    async def test_fork_mode_returns_clean_error(self):
        tool = SkillTool(_skill(execution_mode="fork"))
        result = await tool.execute({}, _ctx())
        assert result.is_error
        assert "fork" in result.content.lower()
        assert "not yet available" in result.content


class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_non_dict_args_error(self):
        tool = SkillTool(_skill())
        result = await tool.execute({"args": "not a dict"}, _ctx())
        assert result.is_error
        assert "args" in result.content


# ─────────────────────────────────────────────────────────────────
# SkillToolProvider
# ─────────────────────────────────────────────────────────────────


class TestSkillToolProvider:
    def test_exposes_every_registered_skill(self):
        registry = SkillRegistry()
        registry.register(_skill("a"))
        registry.register(_skill("b"))
        provider = SkillToolProvider(registry)
        tool_names = [t.name for t in provider.list_tools()]
        # list_tools goes through registry.list_all which sorts
        assert tool_names == ["a", "b"]
        assert all(isinstance(t, SkillTool) for t in provider.list_tools())

    def test_default_name(self):
        provider = SkillToolProvider(SkillRegistry())
        assert provider.name == "skills"

    def test_custom_name(self):
        provider = SkillToolProvider(SkillRegistry(), name="user-skills")
        assert provider.name == "user-skills"

    def test_description_reflects_count(self):
        registry = SkillRegistry()
        registry.register(_skill("a"))
        provider = SkillToolProvider(registry)
        assert "1 skill" in provider.description
        assert "skill (" not in provider.description or "1 skill" in provider.description

    def test_reflects_registry_mutation_before_list(self):
        """Since list_tools reads the registry each call, mutations
        before list_tools() is invoked propagate."""
        registry = SkillRegistry()
        provider = SkillToolProvider(registry)
        assert provider.list_tools() == []
        registry.register(_skill("new"))
        tool_names = [t.name for t in provider.list_tools()]
        assert tool_names == ["new"]


# ─────────────────────────────────────────────────────────────────
# Integration — Pipeline wiring
# ─────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_skills_register_through_manifest(self):
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        registry = SkillRegistry()
        registry.register(_skill("refactor"))
        registry.register(_skill("deploy"))

        manifest = EnvironmentManifest()
        pipeline = await Pipeline.from_manifest_async(
            manifest,
            tool_providers=[SkillToolProvider(registry)],
        )

        tool_names = {t.name for t in pipeline.tool_registry.list_all()}
        assert {"refactor", "deploy"} == tool_names
