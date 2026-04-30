"""Phase 10.1 — argument substitution + invocation flags + when_to_use.

Schema additions are loader-level; behavioural changes are SkillTool-
level. Tests cover both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.skills.loader import SkillLoadError, parse_skill_file
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.skill_tool import (
    SkillTool,
    SkillToolProvider,
    _DOLLAR_PLACEHOLDER,
    _render_body,
)
from geny_executor.skills.types import Skill, SkillMetadata
from geny_executor.tools.base import ToolContext


def _write_skill(tmp_path: Path, *, frontmatter: str, body: str = "body\n") -> Path:
    skill_dir = tmp_path / "skill-x"
    skill_dir.mkdir(exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\nname: Test\ndescription: A skill\n{frontmatter}---\n\n{body}",
        encoding="utf-8",
    )
    return md


# ── arguments / argument_hint ────────────────────────────────────────


def test_arguments_list_loads_as_tuple(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="arguments: [foo, bar]\n")
    skill = parse_skill_file(md)
    assert skill.metadata.arguments == ("foo", "bar")


def test_arguments_single_string_promotes_to_tuple(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="arguments: foo\n")
    skill = parse_skill_file(md)
    assert skill.metadata.arguments == ("foo",)


def test_arguments_blank_strings_dropped(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="arguments: ['', foo, '   ']\n")
    skill = parse_skill_file(md)
    assert skill.metadata.arguments == ("foo",)


def test_arguments_non_string_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="arguments: [foo, 123]\n")
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


def test_argument_hint_loads(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='argument_hint: "<file> [count]"\n')
    skill = parse_skill_file(md)
    assert skill.metadata.argument_hint == "<file> [count]"


def test_argument_hint_blank_becomes_none(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='argument_hint: "   "\n')
    skill = parse_skill_file(md)
    assert skill.metadata.argument_hint is None


# ── when_to_use ──────────────────────────────────────────────────────


def test_when_to_use_loads(tmp_path: Path) -> None:
    md = _write_skill(
        tmp_path,
        frontmatter='when_to_use: "When the user asks for a PR draft."\n',
    )
    skill = parse_skill_file(md)
    assert skill.metadata.when_to_use == "When the user asks for a PR draft."


def test_when_to_use_in_tool_description(tmp_path: Path) -> None:
    md = _write_skill(
        tmp_path,
        frontmatter='when_to_use: "Use for git diff summarisation."\n',
    )
    skill = parse_skill_file(md)
    tool = SkillTool(skill)
    assert "When to use: Use for git diff summarisation." in tool.description


# ── user_invocable / disable_model_invocation ────────────────────────


def test_user_invocable_default_true(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="")
    skill = parse_skill_file(md)
    assert skill.metadata.user_invocable is True
    assert skill.metadata.disable_model_invocation is False


def test_user_invocable_false_loads(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="user_invocable: false\n")
    skill = parse_skill_file(md)
    assert skill.metadata.user_invocable is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("yes", True),
        ("no", False),
        ("ON", True),
        ("OFF", False),
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
    ],
)
def test_invocation_flags_accept_yaml_booleanish(
    tmp_path: Path, raw: str, expected: bool
) -> None:
    md = _write_skill(tmp_path, frontmatter=f'user_invocable: "{raw}"\n')
    skill = parse_skill_file(md)
    assert skill.metadata.user_invocable is expected


def test_invocation_flag_rejects_garbage(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='user_invocable: "maybe"\n')
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


def test_disable_model_invocation_filters_tool_provider() -> None:
    """A skill marked ``disable_model_invocation: true`` must not
    appear in the SkillToolProvider's listing — the model can't reach
    it via tool_use, only the user via slash commands."""
    skill_a = Skill(
        id="visible",
        metadata=SkillMetadata(name="A", description="visible"),
        body="body",
    )
    skill_b = Skill(
        id="user-only",
        metadata=SkillMetadata(
            name="B",
            description="user-only",
            disable_model_invocation=True,
        ),
        body="body",
    )
    registry = SkillRegistry()
    registry.register_many([skill_a, skill_b])
    provider = SkillToolProvider(registry)
    tools = provider.list_tools()
    names = {t.name for t in tools}
    assert names == {"visible"}


# ── ${name} substitution ─────────────────────────────────────────────


def test_dollar_placeholder_substitutes_args() -> None:
    skill = Skill(
        id="echo",
        metadata=SkillMetadata(
            name="Echo",
            description="echoes",
            arguments=("greeting", "name"),
        ),
        body="${greeting}, ${name}!",
    )
    out = _render_body(skill, {"greeting": "Hi", "name": "Alex"})
    assert out == "Hi, Alex!"


def test_dollar_placeholder_missing_arg_becomes_empty_string() -> None:
    skill = Skill(
        id="echo",
        metadata=SkillMetadata(
            name="Echo",
            description="echoes",
            arguments=("greeting",),
        ),
        body="${greeting}, ${missing}!",
    )
    out = _render_body(skill, {"greeting": "Hi"})
    assert out == "Hi, !"


def test_dollar_placeholder_no_args_strips_placeholders() -> None:
    skill = Skill(
        id="echo",
        metadata=SkillMetadata(name="Echo", description="echoes"),
        body="Hello ${who}!",
    )
    out = _render_body(skill, {})
    assert out == "Hello !"


def test_dollar_placeholder_pattern_matches_python_identifiers() -> None:
    """Names match Python-identifier rules. ``${1bad}`` etc. don't."""
    matches = _DOLLAR_PLACEHOLDER.findall("${a} ${b_2} ${1bad} ${c-d}")
    assert matches == ["a", "b_2"]


def test_legacy_brace_placeholder_still_works() -> None:
    """Skills written before 10.1 used ``{name}`` brace style.
    Renderer keeps that path so existing skills don't regress."""
    skill = Skill(
        id="legacy",
        metadata=SkillMetadata(name="Legacy", description="legacy"),
        body="Hello {who}!",
    )
    out = _render_body(skill, {"who": "world"})
    assert out == "Hello world!"


def test_legacy_brace_unknown_name_passes_through() -> None:
    """Unknown brace names stay as ``{name}`` — backward-compat with
    skills that contain literal example braces in markdown."""
    skill = Skill(
        id="legacy",
        metadata=SkillMetadata(name="Legacy", description="legacy"),
        body="curly {example} stays",
    )
    out = _render_body(skill, {"other": "value"})
    assert out == "curly {example} stays"


def test_dollar_placeholder_coerces_non_string_values() -> None:
    skill = Skill(
        id="echo",
        metadata=SkillMetadata(
            name="Echo",
            description="echoes",
            arguments=("count",),
        ),
        body="count=${count}",
    )
    out = _render_body(skill, {"count": 42})
    assert out == "count=42"


def test_dollar_placeholder_none_becomes_empty() -> None:
    skill = Skill(
        id="echo",
        metadata=SkillMetadata(
            name="Echo", description="echoes", arguments=("opt",)
        ),
        body="opt=[${opt}]",
    )
    out = _render_body(skill, {"opt": None})
    assert out == "opt=[]"


# ── input_schema reflects declared arguments ─────────────────────────


def test_input_schema_documents_declared_arguments() -> None:
    skill = Skill(
        id="declared",
        metadata=SkillMetadata(
            name="D",
            description="d",
            arguments=("input_path", "max_lines"),
            argument_hint="<path> [n]",
        ),
        body="${input_path} ${max_lines}",
    )
    tool = SkillTool(skill)
    desc = tool.input_schema["properties"]["args"]["description"]
    assert "input_path" in desc and "max_lines" in desc
    assert "Hint: <path> [n]" in desc


def test_input_schema_default_when_no_arguments() -> None:
    skill = Skill(
        id="bare",
        metadata=SkillMetadata(name="B", description="b"),
        body="hi",
    )
    tool = SkillTool(skill)
    desc = tool.input_schema["properties"]["args"]["description"]
    # Default copy retained for skills without declared arguments.
    assert "consult the skill description" in desc.lower()


# ── execute() ToolResult metadata still serialises new fields ────────


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="/tmp", parent_tool_use_id="p")


@pytest.mark.asyncio
async def test_execute_renders_body_with_dollar_args() -> None:
    skill = Skill(
        id="hello",
        metadata=SkillMetadata(
            name="Hello",
            description="say hi",
            arguments=("name",),
        ),
        body="hi ${name}",
    )
    tool = SkillTool(skill)
    res = await tool.execute({"args": {"name": "world"}}, _ctx())
    assert res.is_error is False
    assert "hi world" in res.content


@pytest.mark.asyncio
async def test_execute_unknown_args_dont_crash() -> None:
    skill = Skill(
        id="hello",
        metadata=SkillMetadata(
            name="Hello", description="say hi", arguments=("name",)
        ),
        body="hi ${name}",
    )
    tool = SkillTool(skill)
    # Pass extra args the skill doesn't declare — should still render.
    res = await tool.execute({"args": {"name": "x", "junk": "y"}}, _ctx())
    assert res.is_error is False
    assert "hi x" in res.content
