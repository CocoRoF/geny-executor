"""EXEC-2 — `MemoryHooks.after_*` callback wiring.

Verifies that the file provider fires `after_record_turn`,
`after_record_execution`, `after_note_write`, and `after_note_update`
once per matching operation, with the documented arguments. Hook
exceptions are swallowed (memory writes are authoritative).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from geny_executor.memory.provider import (
    ExecutionSummary,
    MemoryHooks,
    NoteDraft,
    NoteMeta,
    NotePatch,
    RecordReceipt,
    Scope,
    Turn,
)
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _make_provider_with_hooks(root: Path):
    captured: dict[str, List[Tuple]] = {
        "record_turn": [],
        "record_execution": [],
        "note_write": [],
        "note_update": [],
    }

    async def on_turn(turn: Turn, receipt: RecordReceipt) -> None:
        captured["record_turn"].append((turn, receipt))

    async def on_exec(summary: ExecutionSummary, receipt: RecordReceipt) -> None:
        captured["record_execution"].append((summary, receipt))

    async def on_write(meta: NoteMeta) -> None:
        captured["note_write"].append((meta,))

    async def on_update(meta: NoteMeta) -> None:
        captured["note_update"].append((meta,))

    hooks = MemoryHooks(
        after_record_turn=on_turn,
        after_record_execution=on_exec,
        after_note_write=on_write,
        after_note_update=on_update,
    )
    provider = FileMemoryProvider(root=root, scope=Scope.SESSION, hooks=hooks)
    return provider, captured


def _run(coro):
    return asyncio.run(coro)


def test_after_record_turn_fires_once_per_append():
    with tempfile.TemporaryDirectory() as td:
        provider, captured = _make_provider_with_hooks(Path(td))

        async def go():
            await provider.initialize()
            await provider.record_turn(Turn(role="user", content="hello"))
            await provider.record_turn(Turn(role="assistant", content="hi"))

        _run(go())
        assert len(captured["record_turn"]) == 2
        roles = [turn.role for turn, _ in captured["record_turn"]]
        assert roles == ["user", "assistant"]
        # Receipt is a default RecordReceipt() — STM-only writes don't have
        # notes / vector counts to report.
        for _, receipt in captured["record_turn"]:
            assert isinstance(receipt, RecordReceipt)


def test_after_note_write_fires_with_meta():
    with tempfile.TemporaryDirectory() as td:
        provider, captured = _make_provider_with_hooks(Path(td))

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="N",
                    body="body",
                    category="topics",
                    filename="n.md",
                    metadata={"geny.k": "v"},
                )
            )

        _run(go())
        assert len(captured["note_write"]) == 1
        (meta,) = captured["note_write"][0]
        assert meta.ref.filename == "n.md"
        assert meta.metadata == {"geny.k": "v"}


def test_after_note_update_fires_with_meta():
    with tempfile.TemporaryDirectory() as td:
        provider, captured = _make_provider_with_hooks(Path(td))

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(title="N", body="b", category="topics", filename="n.md")
            )
            await provider.notes().update("n.md", NotePatch(body="b2"))

        _run(go())
        assert len(captured["note_update"]) == 1
        (meta,) = captured["note_update"][0]
        assert meta.ref.filename == "n.md"


def test_after_record_execution_fires_with_summary_and_receipt():
    with tempfile.TemporaryDirectory() as td:
        provider, captured = _make_provider_with_hooks(Path(td))

        async def go():
            await provider.initialize()
            summary = ExecutionSummary(
                session_id="s1",
                user_input="ping",
                final_text="pong",
            )
            receipt = await provider.record_execution(summary)
            return receipt

        receipt = _run(go())
        assert len(captured["record_execution"]) == 1
        captured_summary, captured_receipt = captured["record_execution"][0]
        assert captured_summary.user_input == "ping"
        # The hook receives the same RecordReceipt instance the caller
        # gets back, so notes_written / files_updated are populated.
        assert captured_receipt.notes_written == receipt.notes_written
        assert captured_receipt.files_updated == receipt.files_updated


def test_hook_exceptions_are_swallowed():
    """A buggy hook must not abort the memory write."""
    with tempfile.TemporaryDirectory() as td:

        async def boom(*args, **kwargs):
            raise RuntimeError("hook failure")

        hooks = MemoryHooks(after_note_write=boom, after_record_turn=boom)
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, hooks=hooks)

        async def go():
            await provider.initialize()
            # Should not raise even though both hooks throw.
            await provider.record_turn(Turn(role="user", content="x"))
            meta = await provider.notes().write(
                NoteDraft(title="N", body="b", filename="n.md")
            )
            return meta

        meta = _run(go())
        assert meta.ref.filename == "n.md"


def test_no_hooks_attached_does_not_break():
    """Provider built without hooks behaves exactly as before EXEC-2."""
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.record_turn(Turn(role="user", content="x"))
            await provider.notes().write(
                NoteDraft(title="N", body="b", filename="n.md")
            )

        _run(go())  # no exception → pass


def test_set_hooks_swaps_callbacks_post_construction():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        observed = []

        async def cb(turn, receipt):
            observed.append(turn.role)

        provider.set_hooks(MemoryHooks(after_record_turn=cb))

        async def go():
            await provider.initialize()
            await provider.record_turn(Turn(role="assistant", content="x"))

        _run(go())
        assert observed == ["assistant"]
