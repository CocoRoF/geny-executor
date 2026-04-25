"""Built-in introspection slash commands tests (PR-A.2.2)."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from geny_executor.slash_commands import (
    SlashCategory,
    SlashCommand,
    SlashCommandRegistry,
    SlashContext,
)
from geny_executor.slash_commands.built_in import (
    ClearCommand,
    ContextCommand,
    CostCommand,
    HelpCommand,
    MemoryCommand,
    StatusCommand,
    install_built_in_commands,
)


# ── Stubs ────────────────────────────────────────────────────────────


class _FakeAccountant:
    def snapshot(self) -> dict:
        return {
            "input_tokens": 1234,
            "output_tokens": 567,
            "cached_input_tokens": 100,
            "estimated_usd": 0.1234,
        }


class _FakeHistory:
    def __init__(self):
        self.cleared = False

    async def clear(self):
        self.cleared = True


class _FakeMemory:
    def __init__(self, items: List[Any] | None = None):
        self.items = items or [
            type("N", (), {"summary": f"note {i}"})() for i in range(5)
        ]

    def recent(self, limit: int = 10):
        return self.items[:limit]


class _FakeContextLoader:
    def __init__(self, paths: List[str] | None = None):
        if paths is None:
            paths = ["/work/CLAUDE.md", "/work/AGENTS.md"]
        self._paths = paths

    def last_loaded_paths(self):
        return list(self._paths)


class _FakePipeline:
    def __init__(self, **strategies):
        self._strategies = dict(strategies)
        # Minimal manifest for /status.
        self.manifest = type(
            "M", (), {"preset_name": "worker_adaptive", "model": "claude-haiku"}
        )()
        self.stages = []

    def get_strategy(self, name: str):
        return self._strategies.get(name)


# ── /cost ────────────────────────────────────────────────────────────


class TestCost:
    @pytest.mark.asyncio
    async def test_renders_snapshot(self):
        ctx = SlashContext(pipeline=_FakePipeline(token_accountant=_FakeAccountant()))
        result = await CostCommand().execute([], ctx)
        assert result.success is True
        assert "1,234" in result.content
        assert "$0.1234" in result.content

    @pytest.mark.asyncio
    async def test_no_pipeline(self):
        result = await CostCommand().execute([], SlashContext())
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_accountant(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await CostCommand().execute([], ctx)
        assert result.success is False


# ── /clear ───────────────────────────────────────────────────────────


class TestClear:
    @pytest.mark.asyncio
    async def test_clears_history(self):
        history = _FakeHistory()
        ctx = SlashContext(pipeline=_FakePipeline(history_provider=history))
        result = await ClearCommand().execute([], ctx)
        assert result.success is True
        assert history.cleared is True

    @pytest.mark.asyncio
    async def test_no_history(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await ClearCommand().execute([], ctx)
        assert result.success is False


# ── /status ──────────────────────────────────────────────────────────


class TestStatus:
    @pytest.mark.asyncio
    async def test_renders_session_info(self):
        ctx = SlashContext(pipeline=_FakePipeline(), session_id="sess-1")
        result = await StatusCommand().execute([], ctx)
        assert "worker_adaptive" in result.content
        assert "claude-haiku" in result.content
        assert "sess-1" in result.content

    @pytest.mark.asyncio
    async def test_no_pipeline(self):
        result = await StatusCommand().execute([], SlashContext())
        assert result.success is False


# ── /help ────────────────────────────────────────────────────────────


class TestHelp:
    @pytest.mark.asyncio
    async def test_lists_registered(self):
        reg = SlashCommandRegistry()
        install_built_in_commands(reg)
        ctx = SlashContext(extras={"slash_registry": reg})
        result = await HelpCommand().execute([], ctx)
        for cmd_name in ("cost", "clear", "status", "memory", "context"):
            assert f"/{cmd_name}" in result.content

    @pytest.mark.asyncio
    async def test_groups_by_category(self):
        reg = SlashCommandRegistry()
        install_built_in_commands(reg)
        ctx = SlashContext(extras={"slash_registry": reg})
        result = await HelpCommand().execute([], ctx)
        # Markdown headers per category.
        assert "Introspection" in result.content
        assert "Control" in result.content


# ── /memory ──────────────────────────────────────────────────────────


class TestMemory:
    @pytest.mark.asyncio
    async def test_lists_recent_notes(self):
        ctx = SlashContext(pipeline=_FakePipeline(memory_provider=_FakeMemory()))
        result = await MemoryCommand().execute([], ctx)
        assert "note 0" in result.content

    @pytest.mark.asyncio
    async def test_respects_limit_arg(self):
        ctx = SlashContext(pipeline=_FakePipeline(memory_provider=_FakeMemory()))
        result = await MemoryCommand().execute(["2"], ctx)
        # Two notes shown.
        assert result.content.count("- ") == 2

    @pytest.mark.asyncio
    async def test_no_memory(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await MemoryCommand().execute([], ctx)
        assert result.success is False


# ── /context ─────────────────────────────────────────────────────────


class TestContext:
    @pytest.mark.asyncio
    async def test_lists_loaded_paths(self):
        ctx = SlashContext(pipeline=_FakePipeline(context_loader=_FakeContextLoader()))
        result = await ContextCommand().execute([], ctx)
        assert "CLAUDE.md" in result.content
        assert "AGENTS.md" in result.content

    @pytest.mark.asyncio
    async def test_no_loader(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await ContextCommand().execute([], ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_paths(self):
        ctx = SlashContext(pipeline=_FakePipeline(context_loader=_FakeContextLoader([])))
        result = await ContextCommand().execute([], ctx)
        assert "No context files loaded." in result.content


# ── install_built_in_commands ────────────────────────────────────────


class TestInstall:
    def test_installs_introspection_subset(self):
        reg = SlashCommandRegistry()
        install_built_in_commands(reg)
        # PR-A.2.2 ships these six; PR-A.2.3 adds more.
        for name in ("cost", "clear", "status", "help", "memory", "context"):
            assert reg.resolve(name) is not None

    def test_idempotent_reinstall_overwrites(self, caplog):
        reg = SlashCommandRegistry()
        install_built_in_commands(reg)
        install_built_in_commands(reg)
        assert reg.resolve("cost") is not None  # still present
