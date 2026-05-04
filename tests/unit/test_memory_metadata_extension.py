"""Tests for the `metadata` extension field on memory dataclasses.

Cycle EXEC-1 — every stage I/O dataclass gains a `metadata: Dict[str, Any]`
field that providers store and round-trip verbatim. Host code (Geny) uses
namespaced keys (`geny.*`) to attach business hints (InteractionEvent
fields, importance, source, etc.) without the executor needing to interpret
them.

The contract checked here:

* Construction with no metadata → empty dict.
* Construction with a metadata dict → preserved verbatim.
* `Turn.from_state_message` lifts `message["metadata"]` onto `Turn.metadata`.
* `as_meta()` carries metadata across the Note → NoteMeta projection.
* File / ephemeral providers round-trip metadata through write → read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    Insight,
    MemorySnapshot,
    Note,
    NoteDraft,
    NoteGraph,
    NoteMeta,
    NotePatch,
    NoteRef,
    RecordReceipt,
    ReflectionContext,
    RetrievalResult,
    Scope,
    Turn,
)


# ── Construction defaults ───────────────────────────────────────────────


def test_dataclasses_default_metadata_is_empty_dict():
    assert Turn(role="user", content="hi").metadata == {}
    assert NoteMeta(ref=NoteRef(filename="a.md")).metadata == {}
    assert (
        Note(ref=NoteRef(filename="a.md"), title="t", body="b").metadata == {}
    )
    assert NoteDraft(title="t", body="b").metadata == {}
    assert NotePatch().metadata is None  # patch metadata is Optional
    assert NoteGraph().metadata == {}
    assert RecordReceipt().metadata == {}
    assert (
        Insight(title="t", content="c").metadata == {}
    )
    assert ReflectionContext(session_id="s1").metadata == {}
    assert RetrievalResult().metadata == {}
    assert (
        MemorySnapshot(provider="x", version="0", layers=[], payload=None).metadata
        == {}
    )


def test_metadata_is_preserved_verbatim():
    payload = {
        "geny.interaction.kind": "user_chat",
        "geny.interaction.counterpart_id": "user_alpha",
        "geny.bucket": "user__hello",
    }
    turn = Turn(role="user", content="hi", metadata=dict(payload))
    assert turn.metadata == payload
    # Mutating the input dict afterwards must not mutate the stored copy
    # (constructors copy via the host caller, contract is conservative).
    payload["mutated"] = True
    assert "mutated" not in turn.metadata or turn.metadata["mutated"]


# ── Turn.from_state_message ─────────────────────────────────────────────


def test_from_state_message_lifts_metadata_to_turn():
    msg = {
        "role": "user",
        "content": "hello",
        "metadata": {
            "geny.interaction.event_id": "abcd1234",
            "geny.interaction.kind": "user_chat",
            "geny.importance": "low",
        },
    }
    turn = Turn.from_state_message(msg)
    assert turn.role == "user"
    assert turn.content == "hello"
    assert turn.metadata == msg["metadata"]


def test_from_state_message_with_no_metadata_yields_empty_dict():
    turn = Turn.from_state_message({"role": "assistant", "content": "ack"})
    assert turn.metadata == {}


def test_from_state_message_with_non_mapping_metadata_yields_empty_dict():
    # Defensive: if upstream code stamps a string by mistake, we don't crash.
    turn = Turn.from_state_message({"role": "user", "content": "x", "metadata": "junk"})
    assert turn.metadata == {}


# ── as_meta projection ─────────────────────────────────────────────────


def test_note_as_meta_preserves_metadata():
    note = Note(
        ref=NoteRef(filename="a.md"),
        title="t",
        body="b",
        metadata={"geny.bucket": "user"},
    )
    meta = note.as_meta()
    assert isinstance(meta, NoteMeta)
    assert meta.metadata == {"geny.bucket": "user"}


# ── Provider round-trip ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_provider_roundtrips_note_metadata(tmp_path: Path):
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    await provider.notes().write(
        NoteDraft(
            title="Hello",
            body="body",
            category="topics",
            filename="hello.md",
            metadata={
                "geny.source": "agent",
                "geny.session_id": "sid_xyz",
            },
        )
    )
    note = await provider.notes().read("hello.md")
    assert note is not None
    assert note.metadata == {
        "geny.source": "agent",
        "geny.session_id": "sid_xyz",
    }


@pytest.mark.asyncio
async def test_file_provider_update_replaces_metadata(tmp_path: Path):
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    await provider.notes().write(
        NoteDraft(
            title="N",
            body="b",
            category="topics",
            filename="n.md",
            metadata={"geny.k1": "v1"},
        )
    )
    await provider.notes().update(
        "n.md",
        NotePatch(metadata={"geny.k2": "v2"}),
    )
    note = await provider.notes().read("n.md")
    assert note is not None
    assert note.metadata == {"geny.k2": "v2"}  # replace semantics


@pytest.mark.asyncio
async def test_file_provider_update_leaves_metadata_alone_when_none(tmp_path: Path):
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    await provider.notes().write(
        NoteDraft(
            title="N",
            body="b",
            category="topics",
            filename="n.md",
            metadata={"geny.k1": "v1"},
        )
    )
    # patch.metadata is None (default) → should NOT clear existing metadata
    await provider.notes().update("n.md", NotePatch(body="new body"))
    note = await provider.notes().read("n.md")
    assert note is not None
    assert note.metadata == {"geny.k1": "v1"}


@pytest.mark.asyncio
async def test_ephemeral_provider_roundtrips_note_metadata():
    from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider

    provider = EphemeralMemoryProvider()
    await provider.initialize()
    await provider.notes().write(
        NoteDraft(
            title="Hello",
            body="b",
            category="topics",
            filename="hello.md",
            metadata={"geny.k": "v"},
        )
    )
    note = await provider.notes().read("hello.md")
    assert note is not None
    assert note.metadata == {"geny.k": "v"}


# ── STM round-trip via Turn.metadata ────────────────────────────────────


@pytest.mark.asyncio
async def test_file_stm_roundtrips_turn_metadata(tmp_path: Path):
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    await provider.stm().append(
        Turn(
            role="user",
            content="hello",
            metadata={
                "geny.interaction.kind": "user_chat",
                "geny.interaction.counterpart_id": "user_alpha",
            },
        )
    )
    recent = await provider.stm().recent(n=5)
    assert len(recent) == 1
    assert recent[0].metadata == {
        "geny.interaction.kind": "user_chat",
        "geny.interaction.counterpart_id": "user_alpha",
    }
