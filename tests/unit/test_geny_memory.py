"""Tests for Geny memory integration (retriever, strategy, persistence, presets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory import (
    GenyMemoryRetriever,
    GenyMemoryStrategy,
    GenyPersistence,
    GenyPresets,
)


# ── Mock Memory Manager (duck-typed to match Geny's SessionMemoryManager) ──


@dataclass
class MockMemoryEntry:
    content: str = ""
    char_count: int = 0
    filename: str = ""
    source: str = "long_term"
    importance: str = "medium"
    tags: list = field(default_factory=list)


@dataclass
class MockSearchResult:
    entry: MockMemoryEntry = field(default_factory=MockMemoryEntry)
    score: float = 1.0
    snippet: str = ""


class MockShortTermMemory:
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._summary: Optional[str] = None

    def get_summary(self) -> Optional[str]:
        return self._summary

    def add_message(self, role: str, content: str, metadata=None):
        self._messages.append({"role": role, "content": content})

    def write_summary(self, summary: str):
        self._summary = summary

    def load_all(self):
        return []


class MockLongTermMemory:
    def __init__(self, main_content: str = ""):
        self._main_content = main_content

    def load_main(self) -> Optional[MockMemoryEntry]:
        if not self._main_content:
            return None
        return MockMemoryEntry(
            content=self._main_content,
            char_count=len(self._main_content),
            filename="memory/MEMORY.md",
        )


class MockVectorMemory:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def search(self, query: str, top_k: int = 5):
        return []


class MockMemoryManager:
    """Duck-typed mock of Geny's SessionMemoryManager."""

    def __init__(
        self,
        main_memory: str = "",
        summary: str = "",
        search_results: Optional[List[MockSearchResult]] = None,
    ):
        self.short_term = MockShortTermMemory()
        self.short_term._summary = summary or None
        self.long_term = MockLongTermMemory(main_memory)
        self.vector_memory = MockVectorMemory(enabled=False)
        self._search_results = search_results or []
        self._recorded_messages: List[Dict] = []
        self._notes: List[Dict] = []
        self._dated_entries: List[str] = []

    def search(self, query: str, max_results: int = 5) -> List[MockSearchResult]:
        return self._search_results[:max_results]

    def record_message(self, role: str, content: str, **metadata):
        self._recorded_messages.append({"role": role, "content": content})

    def remember_dated(self, text: str):
        self._dated_entries.append(text)

    def write_note(self, title, content, **kwargs) -> Optional[str]:
        self._notes.append({"title": title, "content": content, **kwargs})
        return f"{title.lower().replace(' ', '-')}.md"

    def read_note(self, filename: str):
        return None


# ══════════════════════════════════════════════════════════════════════
# GenyMemoryRetriever Tests
# ══════════════════════════════════════════════════════════════════════


class TestGenyMemoryRetriever:
    """Test GenyMemoryRetriever (S02 Context Strategy)."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        mgr = MockMemoryManager()
        retriever = GenyMemoryRetriever(mgr)
        result = await retriever.retrieve("", PipelineState())
        assert result == []

    @pytest.mark.asyncio
    async def test_no_manager_returns_empty(self):
        retriever = GenyMemoryRetriever(None)
        result = await retriever.retrieve("test query", PipelineState())
        assert result == []

    @pytest.mark.asyncio
    async def test_loads_session_summary(self):
        mgr = MockMemoryManager(summary="Previous session context here")
        retriever = GenyMemoryRetriever(mgr)
        result = await retriever.retrieve("test", PipelineState())

        summary_chunks = [c for c in result if c.source == "short_term"]
        assert len(summary_chunks) == 1
        assert "Previous session context" in summary_chunks[0].content

    @pytest.mark.asyncio
    async def test_loads_main_memory(self):
        mgr = MockMemoryManager(main_memory="# Important Knowledge\nFoo bar baz")
        retriever = GenyMemoryRetriever(mgr)
        result = await retriever.retrieve("test", PipelineState())

        ltm_chunks = [c for c in result if c.source == "long_term"]
        assert len(ltm_chunks) == 1
        assert "Important Knowledge" in ltm_chunks[0].content

    @pytest.mark.asyncio
    async def test_loads_keyword_results(self):
        mgr = MockMemoryManager(
            search_results=[
                MockSearchResult(
                    entry=MockMemoryEntry(
                        content="Found result",
                        filename="topics/test.md",
                        importance="high",
                    ),
                    score=0.8,
                    snippet="Found result snippet",
                ),
            ]
        )
        retriever = GenyMemoryRetriever(mgr)
        result = await retriever.retrieve("test query", PipelineState())

        keyword_chunks = [c for c in result if c.metadata.get("layer") == "keyword"]
        assert len(keyword_chunks) == 1

    @pytest.mark.asyncio
    async def test_respects_budget(self):
        # Create a large main memory that exceeds budget
        large_content = "x" * 5000
        mgr = MockMemoryManager(
            main_memory=large_content,
            summary="summary " * 100,
        )
        retriever = GenyMemoryRetriever(mgr, max_inject_chars=3000)
        result = await retriever.retrieve("test", PipelineState())

        total_chars = sum(len(c.content) for c in result)
        assert total_chars <= 3000

    @pytest.mark.asyncio
    async def test_llm_gate_skip(self):
        """When LLM gate says no, skip retrieval."""
        mgr = MockMemoryManager(main_memory="important stuff")

        async def gate(query: str) -> bool:
            return False

        retriever = GenyMemoryRetriever(mgr, llm_gate=gate)
        result = await retriever.retrieve("hello", PipelineState())
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_gate_proceed(self):
        """When LLM gate says yes, proceed with retrieval."""
        mgr = MockMemoryManager(main_memory="important stuff")

        async def gate(query: str) -> bool:
            return True

        retriever = GenyMemoryRetriever(mgr, llm_gate=gate)
        result = await retriever.retrieve("complex query", PipelineState())
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_llm_gate_error_proceeds(self):
        """When LLM gate fails, proceed with retrieval (safe side)."""
        mgr = MockMemoryManager(main_memory="important stuff")

        async def gate(query: str) -> bool:
            raise RuntimeError("gate broken")

        retriever = GenyMemoryRetriever(mgr, llm_gate=gate)
        result = await retriever.retrieve("query", PipelineState())
        assert len(result) > 0

    def test_name_and_description(self):
        retriever = GenyMemoryRetriever(None)
        assert retriever.name == "geny_memory"
        assert "5-layer" in retriever.description


# ══════════════════════════════════════════════════════════════════════
# GenyMemoryStrategy Tests
# ══════════════════════════════════════════════════════════════════════


class TestGenyMemoryStrategy:
    """Test GenyMemoryStrategy (S15 Memory Strategy)."""

    @pytest.mark.asyncio
    async def test_records_transcript(self):
        mgr = MockMemoryManager()
        strategy = GenyMemoryStrategy(mgr, enable_reflection=False)

        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        state.final_text = "Python is a programming language."

        await strategy.update(state)

        # Both user and assistant messages should be recorded
        assert len(mgr._recorded_messages) == 2
        assert mgr._recorded_messages[0]["role"] == "user"
        assert mgr._recorded_messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_records_transcript_no_duplicate(self):
        """Second call should not re-record already recorded messages."""
        mgr = MockMemoryManager()
        strategy = GenyMemoryStrategy(mgr, enable_reflection=False)

        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        state.final_text = "reply1"

        await strategy.update(state)
        assert len(mgr._recorded_messages) == 2

        # Simulate next turn: new messages appended
        state.messages.append({"role": "user", "content": "msg2"})
        state.messages.append({"role": "assistant", "content": "reply2"})
        state.final_text = "reply2"
        state.iteration = 1

        await strategy.update(state)
        # Only new messages should be recorded (not the old 2)
        assert len(mgr._recorded_messages) == 4
        assert mgr._recorded_messages[2]["content"] == "msg2"
        assert mgr._recorded_messages[3]["content"] == "reply2"

    @pytest.mark.asyncio
    async def test_records_execution_to_ltm(self):
        mgr = MockMemoryManager()
        strategy = GenyMemoryStrategy(mgr, enable_reflection=False)

        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "Build a complex app"},
            {"role": "assistant", "content": "Step 1..."},
        ]
        state.final_text = "Done building the app"
        state.iteration = 3  # multi-turn
        state.total_cost_usd = 0.05

        await strategy.update(state)

        assert len(mgr._dated_entries) == 1
        assert "Build a complex app" in mgr._dated_entries[0]

    @pytest.mark.asyncio
    async def test_skips_ltm_for_trivial(self):
        """Single-turn with no tools should not record to LTM."""
        mgr = MockMemoryManager()
        strategy = GenyMemoryStrategy(mgr, enable_reflection=False)

        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        state.final_text = "Hello!"
        state.iteration = 0

        await strategy.update(state)
        assert len(mgr._dated_entries) == 0

    @pytest.mark.asyncio
    async def test_reflection_flag_without_llm(self):
        """Without llm_reflect callable, just sets flag."""
        mgr = MockMemoryManager()
        strategy = GenyMemoryStrategy(mgr, enable_reflection=True)

        state = PipelineState()
        state.messages = [{"role": "assistant", "content": "result"}]
        state.final_text = "result"

        await strategy.update(state)
        assert state.metadata.get("needs_reflection") is True

    @pytest.mark.asyncio
    async def test_reflection_with_llm(self):
        """With llm_reflect callable, extracts and saves insights.

        1.10.0 raised the default ``min_insight_importance`` to
        ``"high"``. The mock here emits a ``"high"`` reflection so
        it survives the gate; the gate behaviour itself is covered
        by ``test_reflection_gate_drops_below_threshold``.
        """
        mgr = MockMemoryManager()

        async def mock_reflect(input_text: str, output_text: str):
            return [
                {
                    "title": "Test Insight",
                    "content": "Learned something",
                    "category": "insights",
                    "tags": ["test"],
                    "importance": "high",
                }
            ]

        strategy = GenyMemoryStrategy(mgr, enable_reflection=True, llm_reflect=mock_reflect)

        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "Do something complex"},
            {"role": "assistant", "content": "Done with the task"},
        ]
        state.final_text = "Done with the task"

        await strategy.update(state)

        assert len(mgr._notes) == 1
        assert mgr._notes[0]["title"] == "Test Insight"

    @pytest.mark.asyncio
    async def test_reflection_gate_drops_below_threshold(self):
        """1.10.0 — ``min_insight_importance`` (default ``high``)
        rejects medium / low reflections silently.
        """
        mgr = MockMemoryManager()

        async def mock_reflect(input_text: str, output_text: str):
            return [
                {
                    "title": "Trivial Pattern",
                    "content": "greet warmly",
                    "category": "insights",
                    "tags": [],
                    "importance": "medium",
                },
                {
                    "title": "Real Fact",
                    "content": "user prefers Korean",
                    "category": "insights",
                    "tags": [],
                    "importance": "high",
                },
            ]

        strategy = GenyMemoryStrategy(
            mgr, enable_reflection=True, llm_reflect=mock_reflect,
        )
        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요"},
        ]
        state.final_text = "안녕하세요"

        await strategy.update(state)

        # Only the ``high`` reflection survives.
        assert len(mgr._notes) == 1
        assert mgr._notes[0]["title"] == "Real Fact"

    @pytest.mark.asyncio
    async def test_reflection_gate_low_lets_everything_through(self):
        """``min_insight_importance="low"`` restores legacy permissive
        behaviour (every reflection saved regardless of importance).
        """
        mgr = MockMemoryManager()

        async def mock_reflect(input_text: str, output_text: str):
            return [
                {"title": "A", "content": "x", "category": "insights",
                 "tags": [], "importance": "low"},
                {"title": "B", "content": "y", "category": "insights",
                 "tags": [], "importance": "medium"},
            ]

        strategy = GenyMemoryStrategy(
            mgr, enable_reflection=True, llm_reflect=mock_reflect,
            min_insight_importance="low",
        )
        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        state.final_text = "a"
        await strategy.update(state)

        # Both saved (capped by max_insights=3).
        assert len(mgr._notes) == 2

    @pytest.mark.asyncio
    async def test_no_manager_noop(self):
        strategy = GenyMemoryStrategy(None)
        state = PipelineState()
        await strategy.update(state)  # should not raise


# ══════════════════════════════════════════════════════════════════════
# GenyPersistence Tests
# ══════════════════════════════════════════════════════════════════════


class TestGenyPersistence:
    """Test GenyPersistence (S15 Memory Persistence)."""

    @pytest.mark.asyncio
    async def test_save_writes_summary(self):
        mgr = MockMemoryManager()
        persistence = GenyPersistence(mgr)

        messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "reply2"},
        ]

        await persistence.save("test-session", messages)
        assert mgr.short_term._summary is not None
        assert "Session Summary" in mgr.short_term._summary

    @pytest.mark.asyncio
    async def test_save_skips_short_conversations(self):
        mgr = MockMemoryManager()
        persistence = GenyPersistence(mgr)

        await persistence.save("test-session", [{"role": "user", "content": "hi"}])
        assert mgr.short_term._summary is None

    @pytest.mark.asyncio
    async def test_clear_is_noop(self):
        mgr = MockMemoryManager()
        persistence = GenyPersistence(mgr)
        await persistence.clear("test-session")  # should not raise

    @pytest.mark.asyncio
    async def test_load_returns_empty_for_fresh(self):
        mgr = MockMemoryManager()
        persistence = GenyPersistence(mgr)
        result = await persistence.load("test-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_manager_noop(self):
        persistence = GenyPersistence(None)
        await persistence.save("x", [])
        result = await persistence.load("x")
        assert result == []


# ══════════════════════════════════════════════════════════════════════
# GenyPresets Tests
# ══════════════════════════════════════════════════════════════════════


class TestGenyPresets:
    """Test GenyPresets pipeline factory methods."""

    def test_worker_easy_creates_pipeline(self):
        mgr = MockMemoryManager()
        pipeline = GenyPresets.worker_easy("test-key", mgr)

        assert pipeline is not None
        stages = pipeline.stages
        stage_names = [s.name for s in stages]

        assert "input" in stage_names
        assert "context" in stage_names
        assert "system" in stage_names
        assert "api" in stage_names
        assert "memory" in stage_names
        assert "yield" in stage_names
        # Easy mode: no loop, no evaluate
        assert "loop" not in stage_names

    def test_worker_full_creates_pipeline(self):
        mgr = MockMemoryManager()
        pipeline = GenyPresets.worker_full("test-key", mgr)

        stages = pipeline.stages
        stage_names = [s.name for s in stages]

        assert "input" in stage_names
        assert "context" in stage_names
        assert "guard" in stage_names
        assert "api" in stage_names
        assert "think" in stage_names
        assert "evaluate" in stage_names
        assert "loop" in stage_names
        assert "memory" in stage_names

    def test_worker_full_with_tools(self):
        from geny_executor.tools.base import Tool, ToolResult
        from geny_executor.tools.registry import ToolRegistry

        class DummyTool(Tool):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "A dummy tool"

            @property
            def input_schema(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, input, context=None) -> ToolResult:
                return ToolResult(content="ok")

        mgr = MockMemoryManager()
        tools = ToolRegistry()
        tools.register(DummyTool())
        pipeline = GenyPresets.worker_full("test-key", mgr, tools=tools)

        stage_names = [s.name for s in pipeline.stages]
        assert "tool" in stage_names

    def test_vtuber_creates_pipeline(self):
        mgr = MockMemoryManager()
        pipeline = GenyPresets.vtuber("test-key", mgr)

        stages = pipeline.stages
        stage_names = [s.name for s in stages]

        assert "input" in stage_names
        assert "context" in stage_names
        assert "system" in stage_names
        assert "memory" in stage_names
        # VTuber: has loop/evaluate for tool call support
        assert "loop" in stage_names
        assert "evaluate" in stage_names

    def test_vtuber_custom_persona(self):
        mgr = MockMemoryManager()
        pipeline = GenyPresets.vtuber("test-key", mgr, persona_prompt="I am a custom VTuber!")
        assert pipeline is not None

    def test_all_presets_have_memory_stage(self):
        """All Geny presets must have memory integration."""
        mgr = MockMemoryManager()

        for preset_fn in [
            lambda: GenyPresets.worker_easy("k", mgr),
            lambda: GenyPresets.worker_full("k", mgr),
            lambda: GenyPresets.vtuber("k", mgr),
        ]:
            pipeline = preset_fn()
            stage_names = [s.name for s in pipeline.stages]
            assert "memory" in stage_names, f"Missing memory stage in {pipeline}"
            assert "context" in stage_names, f"Missing context stage in {pipeline}"
