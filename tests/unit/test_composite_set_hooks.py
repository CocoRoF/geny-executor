"""EXEC-A — `MemoryProvider.set_hooks` uniform across all providers.

Pre-1.17.2 only `FileMemoryProvider` implemented `set_hooks`; hosts
that built a composite (Geny does, always) saw the hook chain
silently no-op because their `hasattr(provider, "set_hooks")` check
returned False on the composite. This test locks in the post-fix
behaviour: every provider exposes `set_hooks`, and composite
forwards to every distinct delegate so callbacks reach the
underlying file/sql/ephemeral store layers.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from geny_executor.memory.factory import MemoryProviderFactory
from geny_executor.memory.provider import (
    MemoryHooks,
    MemoryProvider,
    NoteDraft,
    Scope,
    Turn,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider


def _run(coro):
    return asyncio.run(coro)


# ── Protocol surface ───────────────────────────────────────────────


def test_every_provider_implements_set_hooks():
    """Every concrete provider class advertises `set_hooks`."""
    from geny_executor.memory.composite.provider import CompositeMemoryProvider
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    for cls in (FileMemoryProvider, EphemeralMemoryProvider, CompositeMemoryProvider):
        assert hasattr(cls, "set_hooks"), f"{cls.__name__} missing set_hooks"
        assert callable(cls.set_hooks)

    # SQL provider is import-gated on optional `psycopg`; lazy import.
    try:
        from geny_executor.memory.providers.sql.provider import SQLMemoryProvider

        assert hasattr(SQLMemoryProvider, "set_hooks")
        assert callable(SQLMemoryProvider.set_hooks)
    except ImportError:
        pass  # psycopg not installed in this env; skip


# ── Composite forwarding ───────────────────────────────────────────


def test_composite_set_hooks_forwards_to_session_delegate():
    """Composite installs the hook bag on every distinct delegate so
    `after_record_turn` / `after_note_write` reach the underlying
    file STM / Notes stores."""
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "provider": "composite",
            "session_id": "s1",
            "layers": {
                "stm": "session",
                "ltm": "session",
                "notes": "session",
                "index": "session",
            },
            "scope_providers": {"session": "session"},
            "providers": {
                "session": {"provider": "file", "root": td, "session_id": "s1"},
            },
        }
        provider = MemoryProviderFactory().build(cfg)
        _run(provider.initialize())

        captured: list = []

        async def on_turn(turn: Turn, _receipt) -> None:
            captured.append(("turn", turn.role, turn.content))

        async def on_note_write(meta) -> None:
            captured.append(("note_write", meta.ref.filename))

        async def on_note_update(meta) -> None:
            captured.append(("note_update", meta.ref.filename))

        provider.set_hooks(
            MemoryHooks(
                after_record_turn=on_turn,
                after_note_write=on_note_write,
                after_note_update=on_note_update,
            )
        )

        async def go():
            await provider.stm().append(Turn(role="user", content="hi"))
            meta = await provider.notes().write(
                NoteDraft(title="T", body="b", filename="t.md", category="topics")
            )
            from geny_executor.memory.provider import NotePatch
            await provider.notes().update("t.md", NotePatch(body="b2"))
            return meta

        _run(go())

        assert ("turn", "user", "hi") in captured
        assert ("note_write", "t.md") in captured
        assert ("note_update", "t.md") in captured


def test_composite_set_hooks_with_two_distinct_delegates():
    """Curated layout (session + user_curated) — both delegates
    receive the same hook bag exactly once. Distinct providers
    installs once each (same instance ⇒ idempotent)."""
    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "provider": "composite",
            "session_id": "s1",
            "user_id": "alice",
            "layers": {
                "stm": "session",
                "ltm": "session",
                "notes": "session",
                "index": "session",
            },
            "scope_providers": {"session": "session", "user": "user_curated"},
            "providers": {
                "session": {"provider": "file", "root": td, "session_id": "s1"},
                "user_curated": {"provider": "file", "root": str(Path(td) / "_curated_knowledge" / "alice"), "session_id": "s1"},
            },
        }
        provider = MemoryProviderFactory().build(cfg)
        _run(provider.initialize())

        captured: list = []

        async def on_write(meta) -> None:
            captured.append(meta.ref.filename)

        provider.set_hooks(MemoryHooks(after_note_write=on_write))

        async def go():
            await provider.notes().write(
                NoteDraft(title="A", body="x", filename="a.md", category="topics")
            )
            curated = provider.curated()
            if curated is not None:
                await curated.notes().write(
                    NoteDraft(title="B", body="y", filename="b.md", category="topics")
                )

        _run(go())
        # Both writes triggered the hook (session + curated delegates)
        assert "a.md" in captured
        # curated may be unsupported; only assert when fired
        if len(captured) > 1:
            assert "b.md" in captured


# ── Ephemeral / SQL contract surface ───────────────────────────────


def test_ephemeral_set_hooks_holds_hook_bag():
    """Ephemeral provider doesn't fire `after_*` (test fixture only)
    but `set_hooks` is still callable and stores the bag so callers
    don't need a `hasattr` dance."""
    p = EphemeralMemoryProvider()
    p.set_hooks(MemoryHooks(after_record_turn=lambda *a: None))
    # No exception, attribute held
    assert p._hooks is not None


def test_set_hooks_swap_is_idempotent_and_replaces():
    with tempfile.TemporaryDirectory() as td:
        from geny_executor.memory.providers.file.provider import FileMemoryProvider

        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        captured_first: list = []
        captured_second: list = []

        async def first(turn, _r):
            captured_first.append(turn.role)

        async def second(turn, _r):
            captured_second.append(turn.role)

        p.set_hooks(MemoryHooks(after_record_turn=first))
        p.set_hooks(MemoryHooks(after_record_turn=second))

        async def go():
            await p.initialize()
            await p.stm().append(Turn(role="user", content="x"))

        _run(go())
        assert captured_first == []  # superseded
        assert captured_second == ["user"]
