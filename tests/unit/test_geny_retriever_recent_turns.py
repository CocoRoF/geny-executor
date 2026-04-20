"""Regression tests for GenyMemoryRetriever L0 recent-turns layer (0.28.0).

The L0 layer injects the tail of the STM transcript verbatim before
any semantic/keyword matching runs. Its job is to restore conversation
continuity on trigger-style turns whose query text has no lexical
overlap with the prior dialogue — cycle 20260420_8 Bug 2b-β.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory import GenyMemoryRetriever


@dataclass
class _STMEntry:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class _STMWithRecent:
    """Duck-typed short_term supporting both get_recent() and get_summary()."""

    def __init__(self, messages: Optional[List[_STMEntry]] = None, summary: Optional[str] = None):
        self._messages = messages or []
        self._summary = summary

    def get_recent(self, n: int) -> List[_STMEntry]:
        return self._messages[-n:] if n > 0 else []

    def get_summary(self) -> Optional[str]:
        return self._summary


class _STMNoRecent:
    """Legacy STM that only exposes get_summary — used to verify the
    L0 layer skips cleanly when get_recent is unavailable."""

    def __init__(self, summary: Optional[str] = None):
        self._summary = summary

    def get_summary(self) -> Optional[str]:
        return self._summary


class _MemoryManager:
    """Minimum surface for GenyMemoryRetriever to run without blowing
    up on attributes the other layers touch."""

    def __init__(self, short_term: Any = None):
        self.short_term = short_term
        self.long_term = None
        self.vector_memory = None

    def search(self, query: str, max_results: int = 5):
        return []

    def read_note(self, filename: str):
        return None


@pytest.mark.asyncio
async def test_recent_turns_injected_as_l0() -> None:
    """With 8 STM messages and recent_turns=6, the retriever must
    inject the last six as the first chunk."""
    msgs = [
        _STMEntry(content=f"m{i}", metadata={"role": "user" if i % 2 == 0 else "assistant"})
        for i in range(8)
    ]
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=msgs))
    retriever = GenyMemoryRetriever(mgr, recent_turns=6, enable_vector_search=False)

    chunks = await retriever.retrieve("any query", PipelineState())
    recent = [c for c in chunks if c.metadata.get("layer") == "recent_turns"]
    assert len(recent) == 1
    # First chunk overall should be recent_turns (L0 before L1 summary)
    assert chunks[0].metadata.get("layer") == "recent_turns"

    body = recent[0].content
    # Last 6 messages are m2..m7
    assert "m2" in body and "m7" in body
    assert "m0" not in body and "m1" not in body
    # Role labels must be present
    assert "[user]" in body and "[assistant]" in body


@pytest.mark.asyncio
async def test_recent_turns_disabled_when_zero() -> None:
    msgs = [_STMEntry(content="hi", metadata={"role": "user"})]
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=msgs))
    retriever = GenyMemoryRetriever(mgr, recent_turns=0, enable_vector_search=False)

    chunks = await retriever.retrieve("q", PipelineState())
    assert not any(c.metadata.get("layer") == "recent_turns" for c in chunks)


@pytest.mark.asyncio
async def test_recent_turns_budget_capped() -> None:
    """A very long message must be truncated so recent_turns does not
    consume more than 40% of the budget."""
    huge = _STMEntry(content="x" * 50_000, metadata={"role": "user"})
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=[huge]))
    retriever = GenyMemoryRetriever(
        mgr, recent_turns=6, max_inject_chars=1000, enable_vector_search=False
    )

    chunks = await retriever.retrieve("q", PipelineState())
    recent = [c for c in chunks if c.metadata.get("layer") == "recent_turns"]
    assert len(recent) == 1
    # 40% of 1000 = 400
    assert len(recent[0].content) <= 400


@pytest.mark.asyncio
async def test_recent_turns_missing_get_recent_skipped() -> None:
    """Legacy STM (no get_recent) quietly disables L0 but other layers
    still run."""
    mgr = _MemoryManager(short_term=_STMNoRecent(summary="cached summary"))
    retriever = GenyMemoryRetriever(mgr, recent_turns=6, enable_vector_search=False)

    chunks = await retriever.retrieve("q", PipelineState())
    assert not any(c.metadata.get("layer") == "recent_turns" for c in chunks)
    # Session summary layer still fires
    assert any(c.metadata.get("layer") == "session_summary" for c in chunks)


@pytest.mark.asyncio
async def test_recent_turns_precedes_session_summary_in_chunks() -> None:
    """Ordering contract: L0 recent_turns chunk must come before L1
    session_summary chunk — downstream prompt assembly relies on this."""
    msgs = [_STMEntry(content="hi", metadata={"role": "user"})]
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=msgs, summary="a summary line"))
    retriever = GenyMemoryRetriever(mgr, recent_turns=6, enable_vector_search=False)

    chunks = await retriever.retrieve("q", PipelineState())
    layers = [c.metadata.get("layer") for c in chunks]
    i_recent = layers.index("recent_turns")
    i_summary = layers.index("session_summary")
    assert i_recent < i_summary


@pytest.mark.asyncio
async def test_trigger_style_query_finds_prior_subworker_result() -> None:
    """End-to-end for Bug 2b-β: a THINKING_TRIGGER query with no
    lexical overlap with earlier dialogue still surfaces the prior
    [SUB_WORKER_RESULT] turn via the L0 tail."""
    msgs = [
        _STMEntry(
            content="[SUB_WORKER_RESULT] Task completed: test.txt created",
            metadata={"role": "assistant_dm"},
        ),
        _STMEntry(
            content="와! Sub-Worker가 파일을 만들었네!",
            metadata={"role": "assistant"},
        ),
    ]
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=msgs))
    retriever = GenyMemoryRetriever(mgr, recent_turns=6, enable_vector_search=False)

    chunks = await retriever.retrieve(
        "[THINKING_TRIGGER:continued_idle] 여전히 조용하다",
        PipelineState(),
    )
    recent = next(c for c in chunks if c.metadata.get("layer") == "recent_turns")
    assert "SUB_WORKER_RESULT" in recent.content
    assert "assistant_dm" in recent.content  # new role label flows through
    assert "test.txt" in recent.content


@pytest.mark.asyncio
async def test_recent_turns_skips_empty_content() -> None:
    """Whitespace-only STM entries are not injected as blank lines."""
    msgs = [
        _STMEntry(content="hello", metadata={"role": "user"}),
        _STMEntry(content="   \n  ", metadata={"role": "assistant"}),
        _STMEntry(content="there", metadata={"role": "user"}),
    ]
    mgr = _MemoryManager(short_term=_STMWithRecent(messages=msgs))
    retriever = GenyMemoryRetriever(mgr, recent_turns=6, enable_vector_search=False)

    chunks = await retriever.retrieve("q", PipelineState())
    recent = next(c for c in chunks if c.metadata.get("layer") == "recent_turns")
    # Exactly two non-empty lines made it in
    assert recent.metadata["turns"] == 2
    assert "hello" in recent.content and "there" in recent.content
