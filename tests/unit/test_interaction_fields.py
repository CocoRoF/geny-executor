"""Interaction-field round-trip tests (EXEC-9).

Verifies the typed first-class interaction surface
(``event_id`` / ``linked_event_id`` / ``kind`` / ``direction`` /
``counterpart_id`` / ``counterpart_role`` / ``session_id``) survives
write → disk → read across every provider.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    Scope,
    Turn,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


_INTERACTION_KW = dict(
    event_id="evt-1234",
    linked_event_id="evt-prev",
    kind="agent_dm",
    direction="outbound",
    counterpart_id="user-bob",
    counterpart_role="user",
    session_id="sess-XYZ",
)


# ── Notes ────────────────────────────────────────────────────────────


def test_file_notes_round_trip_interaction_fields():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="DM",
                    body="conversation rollup body",
                    category="dms",
                    filename="dm.md",
                    importance=Importance.MEDIUM,
                    **_INTERACTION_KW,
                )
            )
            note = await p.notes().read("dm.md")
            return note

        note = _run(go())
        assert note is not None
        for key, value in _INTERACTION_KW.items():
            assert getattr(note, key) == value, f"mismatch on {key}"

        # Frontmatter should also carry the typed surface so a
        # hand-edited note stays human-readable.
        with (Path(td) / "memory" / "dms" / "dm.md").open("r", encoding="utf-8") as fh:
            text = fh.read()
        for key, value in _INTERACTION_KW.items():
            assert f"interaction.{key}: {value}" in text, f"frontmatter missing {key}"


def test_ephemeral_notes_round_trip_interaction_fields():
    p = EphemeralMemoryProvider()

    async def go():
        await p.initialize()
        await p.notes().write(
            NoteDraft(
                title="DM",
                body="body",
                category="dms",
                filename="dm.md",
                **_INTERACTION_KW,
            )
        )
        return await p.notes().read("dm.md")

    note = _run(go())
    assert note is not None
    for key, value in _INTERACTION_KW.items():
        assert getattr(note, key) == value


def test_file_notes_meta_carries_interaction():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")

        async def go():
            await p.initialize()
            meta = await p.notes().write(
                NoteDraft(
                    title="DM",
                    body="body",
                    category="dms",
                    filename="dm.md",
                    **_INTERACTION_KW,
                )
            )
            return meta

        meta = _run(go())
        for key, value in _INTERACTION_KW.items():
            assert getattr(meta, key) == value


# ── Turn / STM ───────────────────────────────────────────────────────


def test_file_stm_round_trip_interaction_fields():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")

        async def go():
            await p.initialize()
            await p.stm().append(
                Turn(
                    role="assistant",
                    content="hi from agent",
                    timestamp=datetime.now(timezone.utc),
                    **_INTERACTION_KW,
                )
            )
            turns = await p.stm().recent(n=5)
            return turns[-1]

        turn = _run(go())
        for key, value in _INTERACTION_KW.items():
            assert getattr(turn, key) == value


def test_ephemeral_stm_round_trip_interaction_fields():
    p = EphemeralMemoryProvider()

    async def go():
        await p.initialize()
        await p.stm().append(
            Turn(
                role="assistant",
                content="hi",
                timestamp=datetime.now(timezone.utc),
                **_INTERACTION_KW,
            )
        )
        turns = await p.stm().recent(n=5)
        return turns[-1]

    turn = _run(go())
    for key, value in _INTERACTION_KW.items():
        assert getattr(turn, key) == value


# ── Turn.from_state_message lifts both top-level and nested forms ───


def test_turn_from_state_message_lifts_top_level_keys():
    msg = {
        "role": "assistant",
        "content": "hi",
        "event_id": "evt-1",
        "kind": "agent_dm",
    }
    t = Turn.from_state_message(msg)
    assert t.event_id == "evt-1"
    assert t.kind == "agent_dm"


def test_turn_from_state_message_lifts_nested_metadata():
    msg = {
        "role": "user",
        "content": "hi",
        "metadata": {
            "interaction": {
                "event_id": "evt-2",
                "counterpart_id": "user-alice",
            }
        },
    }
    t = Turn.from_state_message(msg)
    assert t.event_id == "evt-2"
    assert t.counterpart_id == "user-alice"


def test_turn_from_state_message_top_level_metadata_keys():
    msg = {
        "role": "user",
        "content": "hi",
        "metadata": {
            "event_id": "evt-3",
            "kind": "user_chat",
        },
    }
    t = Turn.from_state_message(msg)
    assert t.event_id == "evt-3"
    assert t.kind == "user_chat"
