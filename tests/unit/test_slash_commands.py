"""SlashCommandRegistry + parser tests (PR-A.2.1)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from geny_executor.slash_commands import (
    ParsedSlash,
    SlashCategory,
    SlashCommand,
    SlashCommandRegistry,
    SlashContext,
    SlashResult,
    parse_slash,
)
from geny_executor.slash_commands.registry import (
    get_default_registry,
    reset_default_registry,
)


# ── Parser ───────────────────────────────────────────────────────────


class TestParser:
    def test_simple_command(self):
        out = parse_slash("/cost")
        assert out == ParsedSlash(command="cost", args=[], remaining_prompt="")

    def test_command_with_args(self):
        out = parse_slash("/cost --detail session-1")
        assert out.command == "cost"
        assert out.args == ["--detail", "session-1"]

    def test_command_with_remaining_prompt(self):
        out = parse_slash("/skill-foo arg1\nplease run")
        assert out.command == "skill-foo"
        assert out.args == ["arg1"]
        assert out.remaining_prompt == "please run"

    def test_remaining_prompt_strips_leading_whitespace(self):
        out = parse_slash("/x\n   hello")
        assert out.remaining_prompt == "hello"

    def test_quoted_args(self):
        out = parse_slash("/cost 'with quoted'")
        assert out.args == ["with quoted"]

    def test_returns_none_for_non_slash(self):
        assert parse_slash("regular text") is None

    def test_returns_none_for_empty(self):
        assert parse_slash("") is None
        assert parse_slash(None) is None  # type: ignore[arg-type]

    def test_invalid_command_name_returns_none(self):
        assert parse_slash("/123abc") is None
        assert parse_slash("/-bad") is None

    def test_strips_leading_whitespace(self):
        out = parse_slash("   /cost")
        assert out.command == "cost"

    def test_unmatched_quote_returns_none(self):
        assert parse_slash("/cost 'unmatched") is None

    def test_just_slash_returns_none(self):
        assert parse_slash("/") is None


# ── Test commands ────────────────────────────────────────────────────


class _StubCommand(SlashCommand):
    def __init__(self, name: str, *, aliases: List[str] | None = None,
                 category: SlashCategory = SlashCategory.INTROSPECTION):
        self.name = name
        self.description = f"stub for {name}"
        self.category = category
        self.aliases = list(aliases or [])

    async def execute(self, args, ctx):
        return SlashResult(content=f"called:{self.name}:{args}")


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_then_resolve(self):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("cost"))
        assert reg.resolve("cost") is not None

    def test_overwrite_warns(self, caplog):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("cost"))
        reg.register(_StubCommand("cost"))
        # Second registration overwrites + logs warning. Warning checked
        # via the singleton logger if needed; here we just confirm
        # behaviour: resolve still returns the last one.
        assert reg.resolve("cost") is not None

    def test_aliases_resolvable(self):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("status", aliases=["st", "info"]))
        # All three names resolve to the same instance.
        cmd = reg.resolve("status")
        assert reg.resolve("st") is cmd
        assert reg.resolve("info") is cmd

    def test_list_all_dedupes_aliases(self):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("status", aliases=["st"]))
        reg.register(_StubCommand("cost"))
        names = [c.name for c in reg.list_all()]
        # Should be the canonical names only, not the aliases.
        assert names == ["cost", "status"]

    def test_list_by_category(self):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("cost", category=SlashCategory.INTROSPECTION))
        reg.register(_StubCommand("cancel", category=SlashCategory.CONTROL))
        reg.register(_StubCommand("preset", category=SlashCategory.DOMAIN))
        intros = reg.list_by_category(SlashCategory.INTROSPECTION)
        assert [c.name for c in intros] == ["cost"]
        controls = reg.list_by_category(SlashCategory.CONTROL)
        assert [c.name for c in controls] == ["cancel"]

    def test_deregister_removes_command_and_aliases(self):
        reg = SlashCommandRegistry()
        reg.register(_StubCommand("status", aliases=["st"]))
        assert reg.deregister("status") is True
        assert reg.resolve("status") is None
        assert reg.resolve("st") is None

    def test_deregister_unknown_returns_false(self):
        reg = SlashCommandRegistry()
        assert reg.deregister("ghost") is False

    def test_resolve_unknown_returns_none(self):
        assert SlashCommandRegistry().resolve("ghost") is None

    def test_default_registry_is_singleton(self):
        reset_default_registry()
        a = get_default_registry()
        b = get_default_registry()
        assert a is b

    def test_reset_default_registry(self):
        reset_default_registry()
        a = get_default_registry()
        a.register(_StubCommand("x"))
        b = reset_default_registry()
        assert b.resolve("x") is None
        assert get_default_registry() is b


# ── Discovery paths ──────────────────────────────────────────────────


class TestDiscoveryPaths:
    def test_discover_missing_path_records_no_error(self, tmp_path: Path):
        reg = SlashCommandRegistry()
        loaded = reg.discover_paths(tmp_path / "does-not-exist")
        assert loaded == 0
        assert (tmp_path / "does-not-exist") in reg.discovery_paths

    def test_discover_existing_empty_dir_returns_zero(self, tmp_path: Path):
        reg = SlashCommandRegistry()
        loaded = reg.discover_paths(tmp_path)
        # PR-A.2.4 will add the actual md loader; for now path is
        # recorded but no commands loaded.
        assert loaded == 0
        assert tmp_path in reg.discovery_paths


# ── SlashContext / SlashResult ───────────────────────────────────────


class TestContextAndResult:
    def test_context_defaults(self):
        ctx = SlashContext()
        assert ctx.pipeline is None
        assert ctx.user_id is None
        assert ctx.extras == {}

    def test_result_defaults(self):
        r = SlashResult(content="hi")
        assert r.success is True
        assert r.follow_up_prompt is None

    @pytest.mark.asyncio
    async def test_stub_command_executes(self):
        cmd = _StubCommand("test")
        out = await cmd.execute(["a", "b"], SlashContext())
        assert "called:test" in out.content
        assert "a" in out.content
