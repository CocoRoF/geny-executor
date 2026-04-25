"""MdTemplateCommand + discovery tests (PR-A.2.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.slash_commands import (
    SlashCategory,
    SlashCommandRegistry,
    SlashContext,
)
from geny_executor.slash_commands.md_template import (
    MdTemplateCommand,
    load_md_command,
    load_md_commands_into,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ── Frontmatter parsing ──────────────────────────────────────────────


class TestLoadMdCommand:
    def test_parses_frontmatter_and_body(self, tmp_path: Path):
        md = tmp_path / "runtests.md"
        _write(md, "---\ndescription: Run tests\ncategory: control\n---\nRun the tests now.\n")
        cmd = load_md_command(md)
        assert cmd is not None
        assert cmd.name == "runtests"
        assert cmd.description == "Run tests"
        assert cmd.category == SlashCategory.CONTROL

    def test_default_category_is_domain(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\ndescription: x\n---\nbody\n")
        cmd = load_md_command(md)
        assert cmd.category == SlashCategory.DOMAIN

    def test_aliases_flow_list(self, tmp_path: Path):
        md = tmp_path / "test.md"
        _write(md, "---\naliases: [t, ts]\n---\nbody\n")
        cmd = load_md_command(md)
        assert cmd.aliases == ["t", "ts"]

    def test_aliases_single_token(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\naliases: shortcut\n---\nbody\n")
        cmd = load_md_command(md)
        assert cmd.aliases == ["shortcut"]

    def test_no_frontmatter_returns_none(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "just a body\n")
        assert load_md_command(md) is None

    def test_empty_body_returns_none(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\ndescription: x\n---\n\n")
        assert load_md_command(md) is None

    def test_invalid_name_returns_none(self, tmp_path: Path):
        md = tmp_path / "123bad.md"
        _write(md, "---\n---\nbody\n")
        assert load_md_command(md) is None

    def test_oversize_returns_none(self, tmp_path: Path):
        md = tmp_path / "huge.md"
        _write(md, "---\n---\n" + "x" * (70 * 1024))
        assert load_md_command(md) is None

    def test_unknown_frontmatter_keys_ignored(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\nfoo: bar\nbaz: qux\n---\nbody\n")
        cmd = load_md_command(md)
        assert cmd is not None


# ── Argument substitution ────────────────────────────────────────────


class TestArgSubstitution:
    @pytest.mark.asyncio
    async def test_arg_n_substitution(self, tmp_path: Path):
        md = tmp_path / "echo.md"
        _write(md, "---\n---\nFirst arg: $ARG_1, second: $ARG_2.\n")
        cmd = load_md_command(md)
        result = await cmd.execute(["alpha", "beta"], SlashContext())
        assert "First arg: alpha, second: beta" in result.follow_up_prompt

    @pytest.mark.asyncio
    async def test_args_glob(self, tmp_path: Path):
        md = tmp_path / "echo.md"
        _write(md, "---\n---\nAll: $ARGS\n")
        cmd = load_md_command(md)
        result = await cmd.execute(["a", "b", "c"], SlashContext())
        assert "All: a b c" in result.follow_up_prompt

    @pytest.mark.asyncio
    async def test_unbound_arg_left_intact(self, tmp_path: Path):
        md = tmp_path / "echo.md"
        _write(md, "---\n---\n$ARG_1 then $ARG_5\n")
        cmd = load_md_command(md)
        result = await cmd.execute(["only-one"], SlashContext())
        # ARG_1 substituted; ARG_5 left as literal so the LLM sees it.
        assert "only-one then $ARG_5" in result.follow_up_prompt

    @pytest.mark.asyncio
    async def test_no_args(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\n---\nstatic body\n")
        cmd = load_md_command(md)
        result = await cmd.execute([], SlashContext())
        assert "static body" in result.follow_up_prompt

    @pytest.mark.asyncio
    async def test_metadata_includes_source(self, tmp_path: Path):
        md = tmp_path / "x.md"
        _write(md, "---\n---\nbody\n")
        cmd = load_md_command(md)
        result = await cmd.execute([], SlashContext())
        assert result.metadata["source"].endswith("x.md")


# ── Discovery ────────────────────────────────────────────────────────


class TestLoadMany:
    def test_loads_all_md_files(self, tmp_path: Path):
        _write(tmp_path / "a.md", "---\n---\nA body\n")
        _write(tmp_path / "b.md", "---\n---\nB body\n")
        _write(tmp_path / "ignored.txt", "not md\n")
        reg = SlashCommandRegistry()
        loaded = load_md_commands_into(reg, tmp_path)
        assert loaded == 2
        assert reg.resolve("a") is not None
        assert reg.resolve("b") is not None

    def test_skips_invalid_files_with_warning(self, tmp_path: Path):
        _write(tmp_path / "ok.md", "---\n---\nbody\n")
        _write(tmp_path / "bad.md", "no frontmatter\n")
        _write(tmp_path / "empty.md", "---\n---\n\n")
        reg = SlashCommandRegistry()
        loaded = load_md_commands_into(reg, tmp_path)
        assert loaded == 1
        assert reg.resolve("ok") is not None

    def test_missing_dir_returns_zero(self, tmp_path: Path):
        reg = SlashCommandRegistry()
        loaded = load_md_commands_into(reg, tmp_path / "nope")
        assert loaded == 0


# ── Registry integration ─────────────────────────────────────────────


class TestRegistryDiscovery:
    def test_discover_paths_loads_md_commands(self, tmp_path: Path):
        _write(tmp_path / "deploy.md", "---\ndescription: Trigger deploy\n---\nDeploy now\n")
        reg = SlashCommandRegistry()
        loaded = reg.discover_paths(tmp_path)
        assert loaded == 1
        cmd = reg.resolve("deploy")
        assert cmd is not None
        assert cmd.description == "Trigger deploy"
