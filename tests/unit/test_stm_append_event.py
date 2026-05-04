"""EXEC-6 — `STMHandle.append_event` for non-message lines.

Hosts use this for tool-call traces, state transitions, background
trigger fires — anything that should land in the conversation
transcript jsonl alongside messages but doesn't fit `Turn`'s shape.
`recent` / `search` still scope to messages.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from geny_executor.memory.provider import Scope, Turn
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


def test_file_stm_append_event_writes_event_line():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        provider = FileMemoryProvider(root=root, scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.stm().append(Turn(role="user", content="hello"))
            await provider.stm().append_event(
                "tool_call",
                {"name": "search", "args": {"q": "rocket"}},
                metadata={"geny.tool_run_id": "abc123"},
            )
            await provider.stm().append(Turn(role="assistant", content="ok"))

        _run(go())
        jsonl = (root / "transcripts" / "session.jsonl").read_text(encoding="utf-8")
        lines = [json.loads(l) for l in jsonl.strip().splitlines()]
        assert len(lines) == 3
        assert lines[0]["type"] == "message"
        assert lines[1]["type"] == "event"
        assert lines[1]["event"] == "tool_call"
        assert lines[1]["data"] == {"name": "search", "args": {"q": "rocket"}}
        assert lines[1]["metadata"] == {"geny.tool_run_id": "abc123"}
        assert lines[2]["type"] == "message"


def test_file_stm_recent_skips_events():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.stm().append(Turn(role="user", content="m1"))
            await provider.stm().append_event("evt1", {"k": "v"})
            await provider.stm().append(Turn(role="assistant", content="m2"))
            return await provider.stm().recent(n=10)

        recent = _run(go())
        assert [t.role for t in recent] == ["user", "assistant"]
        assert [t.content for t in recent] == ["m1", "m2"]


def test_file_stm_search_skips_events():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.stm().append(Turn(role="user", content="rocket launch"))
            await provider.stm().append_event(
                "tool_call", {"q": "rocket trajectory"}
            )
            return await provider.stm().search("rocket", limit=10)

        results = _run(go())
        # Only the message line matches; the event payload is not
        # surfaced through `search`.
        assert len(results) == 1
        assert results[0].content == "rocket launch"


def test_ephemeral_stm_append_event_round_trip():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        await provider.stm().append(Turn(role="user", content="x"))
        await provider.stm().append_event("evt", {"k": "v"})
        recent = await provider.stm().recent(n=10)
        return recent

    recent = _run(go())
    assert len(recent) == 1
    assert recent[0].content == "x"
    # The event lives separately on `_events` for hosts that need it.
    assert provider.stm()._events[-1]["event"] == "evt"
