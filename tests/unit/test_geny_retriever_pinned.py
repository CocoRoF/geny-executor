"""Tests for the L1.5 pinned-facts retriever layer (Memory v2 PR 12).

The pinned-facts layer pulls content from the host's
``mgr.load_pinned(max_chars)`` duck-typed surface and injects it into
``state.metadata`` so the system-prompt builder can render it under
``# Pinned Facts``. The layer is host-agnostic — Geny pins from
``memory/critical/``, but other hosts can wire any source.

These tests exercise the executor side only with stubbed managers to
verify the duck-typing contract holds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory import GenyMemoryRetriever


# ── Test doubles ────────────────────────────────────────────────────


@dataclass
class _STM:
    summary: Optional[str] = None
    recent: List[Any] = field(default_factory=list)

    def get_summary(self) -> Optional[str]:
        return self.summary

    def get_recent(self, n: int) -> List[Any]:
        return self.recent[-n:] if n > 0 else []


@dataclass
class _PinnedEntry:
    content: str
    char_count: int = 0
    source: str = "pinned"

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.content)


class _MgrWithPinned:
    """Minimal duck-typed manager exposing the pinned-facts surface."""

    def __init__(self, *, pinned_text: Optional[str] = None) -> None:
        self.short_term = _STM()
        self._pinned_text = pinned_text
        self.last_max_chars: Optional[int] = None

    def load_pinned(self, *, max_chars: int):
        self.last_max_chars = max_chars
        if self._pinned_text is None:
            return None
        return _PinnedEntry(content=self._pinned_text)

    # The retriever defensively probes additional methods; stub them
    # as no-ops so the layer-by-layer code path stays exercised.
    @property
    def long_term(self):
        return None

    @property
    def vector_memory(self):
        return None

    def search(self, *_args, **_kwargs):
        return []

    def read_note(self, *_args, **_kwargs):
        return None


class _MgrWithoutPinned:
    """Manager that does NOT implement ``load_pinned`` — layer must no-op."""

    def __init__(self) -> None:
        self.short_term = _STM()

    @property
    def long_term(self):
        return None

    @property
    def vector_memory(self):
        return None

    def search(self, *_args, **_kwargs):
        return []

    def read_note(self, *_args, **_kwargs):
        return None


def _state(query: str = "hi") -> PipelineState:
    """Build a minimal PipelineState carrying ``query`` as user input."""
    state = PipelineState()
    state.session_id = "test-session"
    state.messages = [{"role": "user", "content": query}]
    return state


# ── L1.5 pinned layer ───────────────────────────────────────────────


def test_pinned_layer_calls_load_pinned_and_emits_chunk():
    mgr = _MgrWithPinned(pinned_text="User wants to be addressed as 주인님.")
    retr = GenyMemoryRetriever(
        mgr,
        max_inject_chars=2000,
        enable_vector_search=False,
        recent_turns=0,
        always_render_vault_map=False,
    )
    chunks = asyncio.run(retr.retrieve("뭐하고 있었어", _state("뭐하고 있었어")))

    pinned = [c for c in chunks if c.source == "pinned"]
    assert len(pinned) == 1
    assert "주인님" in pinned[0].content
    # Budget cap = 30% of 2000 by default.
    assert mgr.last_max_chars == 600


def test_pinned_layer_noops_when_host_lacks_load_pinned():
    mgr = _MgrWithoutPinned()
    retr = GenyMemoryRetriever(
        mgr,
        max_inject_chars=2000,
        enable_vector_search=False,
        recent_turns=0,
        always_render_vault_map=False,
    )
    chunks = asyncio.run(retr.retrieve("hello", _state("hello")))
    assert all(c.source != "pinned" for c in chunks)


def test_pinned_disabled_when_ratio_zero():
    mgr = _MgrWithPinned(pinned_text="should-not-load")
    retr = GenyMemoryRetriever(
        mgr,
        max_inject_chars=2000,
        enable_vector_search=False,
        recent_turns=0,
        pin_budget_ratio=0.0,
        always_render_vault_map=False,
    )
    chunks = asyncio.run(retr.retrieve("hello", _state("hello")))
    assert all(c.source != "pinned" for c in chunks)
    # The loader was never called either — verifies the early-return.
    assert mgr.last_max_chars is None


def test_pinned_chunk_carries_layer_metadata():
    mgr = _MgrWithPinned(pinned_text="Pinned body.")
    retr = GenyMemoryRetriever(
        mgr,
        max_inject_chars=2000,
        enable_vector_search=False,
        recent_turns=0,
        always_render_vault_map=False,
    )
    chunks = asyncio.run(retr.retrieve("hi", _state("hi")))
    pinned = [c for c in chunks if c.source == "pinned"]
    assert pinned and pinned[0].metadata.get("layer") == "pinned"


def test_breakdown_event_includes_pinned_count():
    mgr = _MgrWithPinned(pinned_text="Pinned body.")
    retr = GenyMemoryRetriever(
        mgr,
        max_inject_chars=2000,
        enable_vector_search=False,
        recent_turns=0,
        always_render_vault_map=False,
    )
    state = _state("hi")
    asyncio.run(retr.retrieve("hi", state))
    breakdowns = [
        ev for ev in state.events if ev.get("type") == "memory.retrieve_breakdown"
    ]
    assert breakdowns, "expected memory.retrieve_breakdown event"
    layers = breakdowns[-1]["data"]["layers"]
    assert layers.get("pinned") == 1
