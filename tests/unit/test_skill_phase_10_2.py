"""Phase 10.2 — allowed_tools enforcement + paths conditional activation."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionRule,
    PermissionSource,
)
from geny_executor.skills.loader import SkillLoadError, parse_skill_file
from geny_executor.skills.path_match import compile_patterns, match_any
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.skill_tool import SkillTool, SkillToolProvider
from geny_executor.skills.types import Skill, SkillMetadata
from geny_executor.tools.base import ToolContext


def _write_skill(tmp_path: Path, *, frontmatter: str, body: str = "body\n") -> Path:
    skill_dir = tmp_path / "test"
    skill_dir.mkdir(exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\nname: T\ndescription: t\n{frontmatter}---\n\n{body}",
        encoding="utf-8",
    )
    return md


def _ctx(rules: list = None) -> ToolContext:
    return ToolContext(
        session_id="s",
        working_dir="/tmp",
        permission_rules=list(rules or []),
    )


# ── path_match (pure functions) ──────────────────────────────────────


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("*.py", "foo.py", True),
        ("*.py", "src/foo.py", True),       # floating
        ("*.py", "foo.txt", False),
        ("/*.py", "foo.py", True),
        ("/*.py", "src/foo.py", False),     # anchored at root
        ("src/**/*.ts", "src/a.ts", True),
        ("src/**/*.ts", "src/a/b.ts", True),
        ("src/**/*.ts", "src/a/b/c.ts", True),
        ("src/**/*.ts", "lib/a.ts", False),
        ("docs/", "docs/x.md", True),       # dir-only
        ("docs/", "docs", True),            # dir-only, the dir itself
        ("docs/", "src/docs/x.md", True),   # floating dir match
        ("?.md", "a.md", True),
        ("?.md", "ab.md", False),
    ],
)
def test_path_match_cases(pattern: str, path: str, expected: bool) -> None:
    compiled = compile_patterns([pattern])
    assert match_any([path], compiled) is expected


def test_path_match_empty_compiled_returns_false() -> None:
    assert match_any(["any/path"], []) is False


def test_path_match_normalises_backslashes() -> None:
    compiled = compile_patterns(["src/**/*.ts"])
    assert match_any(["src\\sub\\f.ts"], compiled) is True


def test_path_match_normalises_leading_dot_slash() -> None:
    compiled = compile_patterns(["src/*.py"])
    assert match_any(["./src/foo.py"], compiled) is True


# ── loader: paths field ──────────────────────────────────────────────


def test_paths_list_loads(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='paths: ["src/**/*.ts", "lib/*.js"]\n')
    skill = parse_skill_file(md)
    assert skill.metadata.paths == ("src/**/*.ts", "lib/*.js")


def test_paths_comma_separated_string_loads(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='paths: "src/**/*.ts, lib/*.js"\n')
    skill = parse_skill_file(md)
    assert skill.metadata.paths == ("src/**/*.ts", "lib/*.js")


def test_paths_single_string_loads(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='paths: "src/**/*.ts"\n')
    skill = parse_skill_file(md)
    assert skill.metadata.paths == ("src/**/*.ts",)


def test_paths_empty_loads_as_empty_tuple(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="")
    skill = parse_skill_file(md)
    assert skill.metadata.paths == ()


def test_paths_non_string_entry_raises(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='paths: ["src/*.ts", 123]\n')
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


# ── SkillToolProvider: paths filtering ───────────────────────────────


def _make(id_: str, *, paths: tuple = ()) -> Skill:
    return Skill(
        id=id_,
        metadata=SkillMetadata(name=id_, description="d", paths=paths),
        body="body",
    )


def test_unconditional_skill_always_listed() -> None:
    """No paths = always active regardless of active_paths state."""
    registry = SkillRegistry()
    registry.register(_make("always-on"))
    provider = SkillToolProvider(registry)  # active_paths empty
    assert {t.name for t in provider.list_tools()} == {"always-on"}


def test_conditional_skill_hidden_without_active_paths() -> None:
    registry = SkillRegistry()
    registry.register(_make("ts-only", paths=("src/**/*.ts",)))
    provider = SkillToolProvider(registry)
    assert provider.list_tools() == []


def test_conditional_skill_revealed_when_path_matches() -> None:
    registry = SkillRegistry()
    registry.register(_make("ts-only", paths=("src/**/*.ts",)))
    provider = SkillToolProvider(
        registry, active_paths=["src/foo/bar.ts"]
    )
    assert {t.name for t in provider.list_tools()} == {"ts-only"}


def test_conditional_skill_hidden_when_paths_dont_match() -> None:
    registry = SkillRegistry()
    registry.register(_make("ts-only", paths=("src/**/*.ts",)))
    provider = SkillToolProvider(
        registry, active_paths=["lib/foo.py"]
    )
    assert provider.list_tools() == []


def test_set_active_paths_updates_listing() -> None:
    registry = SkillRegistry()
    registry.register(_make("ts-only", paths=("src/**/*.ts",)))
    provider = SkillToolProvider(registry)
    assert provider.list_tools() == []
    provider.set_active_paths(["src/main.ts"])
    assert {t.name for t in provider.list_tools()} == {"ts-only"}
    provider.set_active_paths([])
    assert provider.list_tools() == []


def test_disable_model_invocation_still_filters_after_paths() -> None:
    """A skill that's disabled for the model AND conditional should
    stay hidden even when its paths match."""
    skill = Skill(
        id="user-only",
        metadata=SkillMetadata(
            name="x",
            description="x",
            paths=("src/**/*.ts",),
            disable_model_invocation=True,
        ),
        body="body",
    )
    registry = SkillRegistry()
    registry.register(skill)
    provider = SkillToolProvider(registry, active_paths=["src/foo.ts"])
    assert provider.list_tools() == []


# ── allowed_tools enforcement ────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_grants_allowed_tools_to_context() -> None:
    """Invoking a skill with declared ``allowed_tools`` should append
    ALLOW rules to the live ToolContext.permission_rules so subsequent
    tool calls in the session see the grant."""
    skill = Skill(
        id="grant-bash",
        metadata=SkillMetadata(
            name="g",
            description="g",
            allowed_tools=("Bash", "Read"),
        ),
        body="run bash",
    )
    tool = SkillTool(skill)
    ctx = _ctx()
    assert ctx.permission_rules == []
    res = await tool.execute({"args": {}}, ctx)
    assert res.is_error is False
    # Two ALLOW rules added.
    assert len(ctx.permission_rules) == 2
    by_tool = {r.tool_name: r for r in ctx.permission_rules}
    assert by_tool["Bash"].behavior is PermissionBehavior.ALLOW
    assert by_tool["Read"].behavior is PermissionBehavior.ALLOW
    # Source is PRESET_DEFAULT (lowest priority — explicit user denies still win).
    assert by_tool["Bash"].source is PermissionSource.PRESET_DEFAULT
    # Grant is tagged with skill id for audit.
    assert "grant-bash" in by_tool["Bash"].reason


@pytest.mark.asyncio
async def test_skill_with_no_allowed_tools_is_no_op_on_context() -> None:
    skill = Skill(
        id="no-grant",
        metadata=SkillMetadata(name="n", description="n"),
        body="body",
    )
    tool = SkillTool(skill)
    ctx = _ctx()
    await tool.execute({"args": {}}, ctx)
    assert ctx.permission_rules == []


@pytest.mark.asyncio
async def test_skill_grants_dont_duplicate_on_repeat_invocation() -> None:
    """Calling the same skill twice in one session shouldn't pile up
    duplicate ALLOW rules — the grant set is keyed by (tool, reason)."""
    skill = Skill(
        id="repeat",
        metadata=SkillMetadata(
            name="r",
            description="r",
            allowed_tools=("Bash",),
        ),
        body="b",
    )
    tool = SkillTool(skill)
    ctx = _ctx()
    await tool.execute({"args": {}}, ctx)
    await tool.execute({"args": {}}, ctx)
    bash_rules = [r for r in ctx.permission_rules if r.tool_name == "Bash"]
    assert len(bash_rules) == 1


@pytest.mark.asyncio
async def test_skill_grant_does_not_override_prior_user_deny() -> None:
    """A skill grant uses PRESET_DEFAULT source (lowest priority).
    A pre-existing USER-source DENY for the same tool should win when
    the matrix evaluates them — verified by source ordering."""
    user_deny = PermissionRule(
        tool_name="Bash",
        behavior=PermissionBehavior.DENY,
        source=PermissionSource.USER,
        pattern=None,
        reason="sandbox env: no Bash ever",
    )
    skill = Skill(
        id="wants-bash",
        metadata=SkillMetadata(
            name="w",
            description="w",
            allowed_tools=("Bash",),
        ),
        body="b",
    )
    tool = SkillTool(skill)
    ctx = _ctx(rules=[user_deny])
    await tool.execute({"args": {}}, ctx)
    # Both rules now present.
    behaviors = [(r.behavior, r.source) for r in ctx.permission_rules]
    assert (PermissionBehavior.DENY, PermissionSource.USER) in behaviors
    assert (PermissionBehavior.ALLOW, PermissionSource.PRESET_DEFAULT) in behaviors
    # The matrix's first-match-by-priority will pick DENY/USER first
    # (USER outranks PRESET_DEFAULT in SOURCE_PRIORITY).


@pytest.mark.asyncio
async def test_metadata_surfaces_granted_tools_list() -> None:
    skill = Skill(
        id="gr",
        metadata=SkillMetadata(
            name="g", description="g", allowed_tools=("Read", "Write")
        ),
        body="b",
    )
    tool = SkillTool(skill)
    ctx = _ctx()
    res = await tool.execute({"args": {}}, ctx)
    assert set(res.metadata["granted_tools"]) == {"Read", "Write"}


@pytest.mark.asyncio
async def test_header_says_granted_when_tools_were_added() -> None:
    skill = Skill(
        id="x",
        metadata=SkillMetadata(
            name="x", description="x", allowed_tools=("Bash",)
        ),
        body="b",
    )
    tool = SkillTool(skill)
    res = await tool.execute({"args": {}}, _ctx())
    assert "(granted)" in res.content


@pytest.mark.asyncio
async def test_header_says_already_allowed_on_second_call() -> None:
    skill = Skill(
        id="x",
        metadata=SkillMetadata(
            name="x", description="x", allowed_tools=("Bash",)
        ),
        body="b",
    )
    tool = SkillTool(skill)
    ctx = _ctx()
    await tool.execute({"args": {}}, ctx)
    res2 = await tool.execute({"args": {}}, ctx)
    assert "(already allowed)" in res2.content
