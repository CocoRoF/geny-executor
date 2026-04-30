"""Phase 10.3 — shell-block execution + bundled-asset extraction."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from geny_executor.skills.loader import SkillLoadError, parse_skill_file
from geny_executor.skills.shell_blocks import (
    ShellBlock,
    execute_blocks,
    is_trusted_source,
    parse_blocks,
)
from geny_executor.skills.skill_tool import SkillTool, _render_body
from geny_executor.skills.types import Skill, SkillMetadata
from geny_executor.tools.base import ToolContext


HAS_BASH = shutil.which("bash") is not None


def _ctx(working_dir: str = "") -> ToolContext:
    return ToolContext(
        session_id="s",
        working_dir=working_dir,
        permission_rules=[],
    )


def _write_skill(tmp_path: Path, *, frontmatter: str = "", body: str = "body\n") -> Path:
    skill_dir = tmp_path / "test"
    skill_dir.mkdir(exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\nname: T\ndescription: t\n{frontmatter}---\n\n{body}",
        encoding="utf-8",
    )
    return md


# ── Block parsing ────────────────────────────────────────────────────


def test_parse_fenced_block_single() -> None:
    body = "before\n```!\necho hi\n```\nafter\n"
    blocks = parse_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].kind == "fenced"
    assert blocks[0].command == "echo hi"


def test_parse_fenced_block_multiline() -> None:
    body = "```!\nls -la\necho done\n```\n"
    blocks = parse_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].kind == "fenced"
    assert blocks[0].command == "ls -la\necho done"


def test_parse_inline_block() -> None:
    body = "Use !`git rev-parse HEAD` to find the current commit.\n"
    blocks = parse_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].kind == "inline"
    assert blocks[0].command == "git rev-parse HEAD"


def test_parse_mixed_blocks_in_order() -> None:
    body = (
        "step 1: !`echo one`\n"
        "```!\necho two\n```\n"
        "step 3: !`echo three`\n"
    )
    blocks = parse_blocks(body)
    assert [b.kind for b in blocks] == ["inline", "fenced", "inline"]
    assert [b.command for b in blocks] == ["echo one", "echo two", "echo three"]


def test_parse_no_blocks_returns_empty() -> None:
    assert parse_blocks("just markdown, no shell\n") == []


def test_parse_inline_inside_fenced_ignored() -> None:
    r"""An ``!`...`` that lives inside a fenced ``\`\`\`!`` block must
    not double-count — the fenced block already swallows it."""
    body = "```!\necho 'hello !`world`'\n```\n"
    blocks = parse_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].kind == "fenced"


def test_parse_ignores_inline_with_newline() -> None:
    """Inline pattern bans newlines so a stray ``!`echo \\n hi`...`` in
    prose can't swallow paragraphs."""
    body = "!`echo line1\nline2`\n"
    assert parse_blocks(body) == []


# ── Execution ────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_fenced_block_substitutes_stdout() -> None:
    body = "before\n```!\necho hello\n```\nafter\n"
    summary = await execute_blocks(body)
    assert "hello" in summary.rendered_body
    assert "```!" not in summary.rendered_body
    assert summary.any_failed is False


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_inline_block_substitutes_stdout() -> None:
    body = "the word is !`echo magic` today\n"
    summary = await execute_blocks(body)
    assert "the word is magic today" in summary.rendered_body


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_failed_block_includes_exit_code() -> None:
    body = "```!\nexit 42\n```\n"
    summary = await execute_blocks(body)
    assert "exit=42" in summary.rendered_body
    assert summary.any_failed is True


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_failed_block_includes_stderr_when_present() -> None:
    body = "```!\nls /nonexistent_path_xyz_123\n```\n"
    summary = await execute_blocks(body)
    assert summary.any_failed is True
    # bash + ls writes to stderr; either the message or the exit
    # marker should land in the rendered body.
    assert "exit=" in summary.rendered_body


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_timeout_marker() -> None:
    body = "```!\nsleep 5\n```\n"
    summary = await execute_blocks(body, timeout_s=0.5)
    assert "[shell timed out" in summary.rendered_body
    assert summary.any_failed is True


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_uses_cwd(tmp_path: Path) -> None:
    body = "```!\npwd\n```\n"
    summary = await execute_blocks(body, cwd=str(tmp_path))
    assert str(tmp_path) in summary.rendered_body


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_execute_uses_env_overlay() -> None:
    body = '```!\necho "$GENY_TEST_VAR"\n```\n'
    summary = await execute_blocks(body, env={"GENY_TEST_VAR": "from-overlay"})
    assert "from-overlay" in summary.rendered_body


@pytest.mark.asyncio
async def test_execute_skips_when_trust_false() -> None:
    body = "```!\necho danger\n```\n"
    summary = await execute_blocks(body, trust_shell=False)
    assert "[shell skipped" in summary.rendered_body
    assert "danger" not in summary.rendered_body
    assert summary.any_skipped is True


@pytest.mark.asyncio
async def test_execute_no_blocks_returns_unchanged() -> None:
    body = "no blocks at all\n"
    summary = await execute_blocks(body)
    assert summary.rendered_body == body
    assert summary.outcomes == []


# ── is_trusted_source ────────────────────────────────────────────────


def test_is_trusted_default_true() -> None:
    assert is_trusted_source(None, {}) is True
    assert is_trusted_source(Path("/some/path/SKILL.md"), {}) is True


def test_is_trusted_mcp_false() -> None:
    assert is_trusted_source(None, {"source_kind": "mcp"}) is False


def test_is_trusted_unknown_extras_value_still_true() -> None:
    """Unknown source_kind values are treated as trusted — only
    ``"mcp"`` flips the switch. Hosts wiring other untrusted bridges
    register them by adding the marker explicitly."""
    assert is_trusted_source(None, {"source_kind": "plugin"}) is True


# ── Loader: shell + shell_timeout_s ──────────────────────────────────


def test_loader_shell_default(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="")
    skill = parse_skill_file(md)
    assert skill.metadata.shell == "bash"
    assert skill.metadata.shell_timeout_s == 30.0


def test_loader_shell_override(tmp_path: Path) -> None:
    md = _write_skill(
        tmp_path, frontmatter="shell: zsh\nshell_timeout_s: 60\n"
    )
    skill = parse_skill_file(md)
    assert skill.metadata.shell == "zsh"
    assert skill.metadata.shell_timeout_s == 60.0


def test_loader_shell_timeout_must_be_positive(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="shell_timeout_s: 0\n")
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


def test_loader_shell_timeout_must_be_number(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter='shell_timeout_s: "fast"\n')
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


def test_loader_shell_must_be_string(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, frontmatter="shell:\n")
    with pytest.raises(SkillLoadError):
        parse_skill_file(md)


# ── ${SKILL_DIR} substitution ────────────────────────────────────────


def test_skill_dir_resolves_to_assets_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / "with-assets"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: a\ndescription: a\n---\n\nload from ${SKILL_DIR}/data.json",
        encoding="utf-8",
    )
    skill = parse_skill_file(md)
    out = _render_body(skill, {})
    assert str(skill_dir) in out
    assert "${SKILL_DIR}" not in out


def test_skill_dir_empty_for_in_code_skills() -> None:
    """Skills built in code (no source) get an empty SKILL_DIR — the
    placeholder is removed so the body doesn't ship literal ``${SKILL_DIR}``."""
    skill = Skill(
        id="bare",
        metadata=SkillMetadata(name="b", description="b"),
        body="path: ${SKILL_DIR}/x",
        source=None,
    )
    out = _render_body(skill, {})
    assert "${SKILL_DIR}" not in out
    assert out == "path: /x"


def test_args_can_override_skill_dir() -> None:
    """Arg names always win over built-in placeholders so an author
    that *really* needs to rebind SKILL_DIR can."""
    skill = Skill(
        id="rebind",
        metadata=SkillMetadata(
            name="r", description="r", arguments=("SKILL_DIR",)
        ),
        body="${SKILL_DIR}",
        source=Path("/real/SKILL.md"),
    )
    out = _render_body(skill, {"SKILL_DIR": "/override"})
    assert out == "/override"


# ── End-to-end: SkillTool.execute() with shell + assets ──────────────


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_skill_tool_runs_shell_and_includes_output(tmp_path: Path) -> None:
    skill_dir = tmp_path / "echoer"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: e\ndescription: e\nshell: bash\n---\n\n"
        "Greeting: !`echo from-shell`\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(md)
    tool = SkillTool(skill)
    res = await tool.execute({"args": {}}, _ctx())
    assert res.is_error is False
    assert "from-shell" in res.content
    assert res.metadata["shell_blocks_run"] == 1
    assert res.metadata["shell_blocks_skipped"] == 0
    assert res.metadata["shell_blocks_failed"] == 0


@pytest.mark.asyncio
async def test_skill_tool_strips_shell_for_mcp_skill() -> None:
    """A skill marked with ``source_kind=mcp`` in extras must not run
    its shell blocks even when bash is available — the body comes
    from a remote and shouldn't reach the host subprocess."""
    skill = Skill(
        id="mcp__remote__danger",
        metadata=SkillMetadata(
            name="x",
            description="x",
            extras={"source_kind": "mcp"},
        ),
        body="```!\necho should-not-run\n```\n",
        source=None,
    )
    tool = SkillTool(skill)
    res = await tool.execute({"args": {}}, _ctx())
    assert res.is_error is False
    assert "should-not-run" not in res.content
    assert "[shell skipped" in res.content
    assert res.metadata["shell_blocks_skipped"] == 1
    assert res.metadata["shell_blocks_run"] == 0


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_skill_tool_shell_uses_skill_dir_in_command(tmp_path: Path) -> None:
    """`${SKILL_DIR}` survives into the shell command — useful for
    skills that ship helper scripts as bundled assets."""
    skill_dir = tmp_path / "scripted"
    skill_dir.mkdir()
    helper = skill_dir / "hello.sh"
    helper.write_text("#!/bin/bash\necho 'from helper'\n", encoding="utf-8")
    helper.chmod(0o755)
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: s\ndescription: s\n---\n\n"
        "```!\nbash ${SKILL_DIR}/hello.sh\n```\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(md)
    tool = SkillTool(skill)
    res = await tool.execute({"args": {}}, _ctx())
    assert "from helper" in res.content


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_skill_tool_shell_block_count_in_header(tmp_path: Path) -> None:
    skill_dir = tmp_path / "two-blocks"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: t\ndescription: t\n---\n\n"
        "first: !`echo a`\nsecond: !`echo b`\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(md)
    tool = SkillTool(skill)
    res = await tool.execute({"args": {}}, _ctx())
    assert "shell blocks: 2 ran" in res.content


@pytest.mark.skipif(not HAS_BASH, reason="bash not available")
@pytest.mark.asyncio
async def test_skill_tool_arg_substitution_into_shell_command(tmp_path: Path) -> None:
    """Args are substituted *before* shell execution so the command
    can use them. End-to-end check on the renderer + executor combo."""
    skill_dir = tmp_path / "echoes-arg"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\nname: a\ndescription: a\narguments: [name]\n---\n\n"
        "Hello !`echo ${name}`\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(md)
    tool = SkillTool(skill)
    res = await tool.execute({"args": {"name": "operator"}}, _ctx())
    assert "Hello operator" in res.content
