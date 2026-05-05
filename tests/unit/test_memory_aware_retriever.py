"""``MemoryAwareRetriever`` — provider-driven Stage 2 retrieval tests.

Verifies the 6-layer retrieval chain runs without host duck-typing.
The retriever consumes a ``MemoryProvider`` directly and reads policy
from a ``MemoryHooks`` bag.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import (
    Importance,
    MemoryHooks,
    NoteDraft,
    Scope,
    Turn,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider
from geny_executor.memory.retriever import MemoryAwareRetriever


def _state(query: str, session_id: str = "sess-test") -> PipelineState:
    s = PipelineState()
    s.session_id = session_id
    s.messages = [{"role": "user", "content": query}]
    return s


def _run(coro):
    return asyncio.run(coro)


# ── Construction ─────────────────────────────────────────────────────


def test_construction_requires_provider() -> None:
    with pytest.raises(ValueError):
        MemoryAwareRetriever(None)  # type: ignore[arg-type]


def test_construction_with_default_hooks() -> None:
    p = EphemeralMemoryProvider()
    r = MemoryAwareRetriever(p)
    assert r.name == "memory_aware"


# ── L0 recent turns ──────────────────────────────────────────────────


def test_recent_turns_layer_returns_tail() -> None:
    p = EphemeralMemoryProvider()

    async def go() -> List:
        await p.initialize()
        for i in range(8):
            await p.stm().append(
                Turn(role="user", content=f"msg-{i}", timestamp=datetime.now(timezone.utc))
            )
        hooks = MemoryHooks(recent_turns=4, slim_mode=True, always_render_vault_map=False)
        r = MemoryAwareRetriever(p, hooks=hooks)
        return await r.retrieve("anything", _state("anything"))

    chunks = _run(go())
    recent = [c for c in chunks if c.metadata.get("layer") == "recent_turns"]
    assert len(recent) == 1
    body = recent[0].content
    # Body holds the last 4 turns; oldest 4 trimmed.
    assert "msg-7" in body
    assert "msg-3" not in body or "[user]" in body  # tail-trimmed


def test_recent_turns_disabled_when_zero() -> None:
    p = EphemeralMemoryProvider()
    hooks = MemoryHooks(recent_turns=0, slim_mode=True, always_render_vault_map=False)

    async def go() -> List:
        await p.initialize()
        await p.stm().append(
            Turn(role="user", content="hi", timestamp=datetime.now(timezone.utc))
        )
        r = MemoryAwareRetriever(p, hooks=hooks)
        return await r.retrieve("hi", _state("hi"))

    chunks = _run(go())
    assert all(c.metadata.get("layer") != "recent_turns" for c in chunks)


# ── L1 session summary ───────────────────────────────────────────────


def test_session_summary_layer_reads_stm() -> None:
    p = EphemeralMemoryProvider()
    hooks = MemoryHooks(slim_mode=True, recent_turns=0, always_render_vault_map=False)

    async def go() -> List:
        await p.initialize()
        await p.stm().write_summary("# Session Summary\n\n- one\n- two")
        r = MemoryAwareRetriever(p, hooks=hooks)
        return await r.retrieve("anything", _state("anything"))

    chunks = _run(go())
    summary = [c for c in chunks if c.metadata.get("layer") == "session_summary"]
    assert len(summary) == 1
    assert "Session Summary" in summary[0].content


# ── L1.5 pinned facts ────────────────────────────────────────────────


def test_pinned_layer_uses_hook_category() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(
            pin_category="critical",
            recent_turns=0,
            slim_mode=True,
            always_render_vault_map=False,
        )

        async def go() -> List:
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="user-name", body="The user's name is Geny.",
                    category="critical", filename="name.md",
                    importance=Importance.CRITICAL,
                )
            )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("hi", _state("hi"))

        chunks = _run(go())
        pinned = [c for c in chunks if c.metadata.get("layer") == "pinned"]
        assert len(pinned) == 1
        assert "Geny" in pinned[0].content
        assert pinned[0].metadata.get("host_layer") == "critical"


def test_pinned_layer_skipped_when_budget_zero() -> None:
    p = EphemeralMemoryProvider()
    hooks = MemoryHooks(
        pin_category="critical",
        layer_budget_ratio={"pinned": 0.0},
        slim_mode=True,
        recent_turns=0,
        always_render_vault_map=False,
    )

    async def go() -> List:
        await p.initialize()
        r = MemoryAwareRetriever(p, hooks=hooks)
        return await r.retrieve("anything", _state("anything"))

    chunks = _run(go())
    assert all(c.metadata.get("layer") != "pinned" for c in chunks)


# ── L1.7 vault map ───────────────────────────────────────────────────


def test_vault_map_uses_hook_descriptions() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(
            vault_descriptions={"topics": "subject pages — agent free-form notes"},
            recent_turns=0,
            slim_mode=True,
            always_render_vault_map=True,
        )

        async def go() -> List:
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T", body="b", category="topics", filename="t.md")
            )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("anything", _state("anything"))

        chunks = _run(go())
        vmap = [c for c in chunks if c.metadata.get("layer") == "vault_map"]
        assert len(vmap) == 1
        assert "topics" in vmap[0].content
        assert "subject pages" in vmap[0].content


# ── L4 keyword importance + category boost ───────────────────────────


def test_keyword_layer_applies_importance_boost() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(
            recent_turns=0,
            slim_mode=False,
            always_render_vault_map=False,
            enable_vector_search=False,
            importance_boost={"critical": 10.0, "low": 0.1, "medium": 1.0, "high": 1.0},
            max_results=5,
        )

        async def go() -> List:
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="critical-note",
                    body="rocket launch fact",
                    category="topics",
                    filename="crit.md",
                    importance=Importance.CRITICAL,
                )
            )
            await p.notes().write(
                NoteDraft(
                    title="low-note",
                    body="rocket launch fact again",
                    category="topics",
                    filename="low.md",
                    importance=Importance.LOW,
                )
            )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("rocket launch", _state("rocket launch"))

        chunks = _run(go())
        keyword = [c for c in chunks if c.metadata.get("layer") == "keyword"]
        # Critical note should rank above the low one.
        keys = [c.key for c in keyword]
        assert "crit.md" in keys and "low.md" in keys
        crit_idx = keys.index("crit.md")
        low_idx = keys.index("low.md")
        assert crit_idx < low_idx


def test_keyword_layer_applies_category_boost() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(
            recent_turns=0,
            slim_mode=False,
            always_render_vault_map=False,
            enable_vector_search=False,
            category_boosts={"insights": 10.0, "topics": 0.1},
        )

        async def go() -> List:
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="topics-note",
                    body="rocket launch fact",
                    category="topics",
                    filename="t.md",
                    importance=Importance.MEDIUM,
                )
            )
            await p.notes().write(
                NoteDraft(
                    title="insights-note",
                    body="rocket launch fact",
                    category="insights",
                    filename="i.md",
                    importance=Importance.MEDIUM,
                )
            )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("rocket launch", _state("rocket launch"))

        chunks = _run(go())
        keys = [c.key for c in chunks if c.metadata.get("layer") == "keyword"]
        assert keys.index("i.md") < keys.index("t.md")


# ── slim mode short-circuits heavy layers ────────────────────────────


def test_slim_mode_stops_after_lightweight_layers() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(slim_mode=True, recent_turns=0, always_render_vault_map=True)

        async def go() -> List:
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="rocket",
                    body="rocket launch fact",
                    category="topics",
                    filename="t.md",
                )
            )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("rocket", _state("rocket"))

        chunks = _run(go())
        layers = {c.metadata.get("layer") for c in chunks}
        # Slim mode keeps L1.7 (vault_map) but drops L4 (keyword) etc.
        assert "vault_map" in layers
        assert "keyword" not in layers
        assert "vector" not in layers


# ── llm_gate predicate skips entirely ────────────────────────────────


def test_llm_gate_false_returns_empty() -> None:
    p = EphemeralMemoryProvider()

    async def gate(_q: str) -> bool:
        return False

    async def go() -> List:
        await p.initialize()
        r = MemoryAwareRetriever(p, llm_gate=gate)
        return await r.retrieve("anything", _state("anything"))

    chunks = _run(go())
    assert chunks == []


# ── budget enforcement ───────────────────────────────────────────────


def test_total_chars_does_not_exceed_max_inject_chars() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION, timezone_name="UTC")
        hooks = MemoryHooks(
            max_inject_chars=200,
            recent_turns=0,
            slim_mode=False,
            always_render_vault_map=False,
            enable_vector_search=False,
        )

        async def go() -> List:
            await p.initialize()
            big_body = "rocket launch fact " * 50  # ~950 chars
            for i in range(5):
                await p.notes().write(
                    NoteDraft(
                        title=f"n{i}",
                        body=big_body,
                        category="topics",
                        filename=f"n{i}.md",
                        importance=Importance.HIGH,
                    )
                )
            r = MemoryAwareRetriever(p, hooks=hooks)
            return await r.retrieve("rocket launch", _state("rocket launch"))

        chunks = _run(go())
        total = sum(len(c.content) for c in chunks)
        assert total <= 200, f"budget breached: {total} > 200"
