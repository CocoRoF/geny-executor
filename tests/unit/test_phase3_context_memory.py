"""Phase 3 tests — context, system, cache, memory, session."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s02_context import (
    ContextStage,
    SimpleLoadStrategy,
    HybridStrategy,
    TruncateCompactor,
)
from geny_executor.stages.s02_context.retrievers import StaticRetriever
from geny_executor.stages.s03_system import (
    SystemStage,
    ComposablePromptBuilder,
    PersonaBlock,
    RulesBlock,
    DateTimeBlock,
)
from geny_executor.stages.s05_cache import (
    SystemCacheStrategy,
    AggressiveCacheStrategy,
)
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s06_api.retry import NoRetry
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s18_memory import (
    MemoryStage,
    InMemoryPersistence,
)
from geny_executor.stages.s21_yield import YieldStage
from geny_executor.session import Session, SessionManager


# ── Context Stage ──


@pytest.mark.asyncio
async def test_context_stage_simple():
    """Simple context strategy passes through."""
    provider = MockProvider(default_text="Reply")
    pipeline = Pipeline(PipelineConfig(name="ctx"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(ContextStage(strategy=SimpleLoadStrategy()))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello")
    assert result.success is True
    assert result.text == "Reply"


@pytest.mark.asyncio
async def test_context_with_memory_retriever():
    """Memory retriever injects chunks."""
    retriever = StaticRetriever()
    retriever.add_chunk("fact1", "The sky is blue", source="long_term")
    retriever.add_chunk("fact2", "Water is wet", source="long_term")

    provider = MockProvider(default_text="Thanks for the context")
    pipeline = Pipeline(PipelineConfig(name="mem"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(ContextStage(retriever=retriever))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    result = await pipeline.run("Tell me facts", state)
    assert result.success is True
    assert len(state.memory_refs) == 2


@pytest.mark.asyncio
async def test_hybrid_strategy_trims():
    """HybridStrategy trims to max recent turns."""
    strategy = HybridStrategy(max_recent_turns=2)
    state = PipelineState()
    # Simulate 10 turns of history
    for i in range(10):
        state.messages.append({"role": "user", "content": f"Q{i}"})
        state.messages.append({"role": "assistant", "content": f"A{i}"})

    await strategy.build_context(state)
    assert len(state.messages) == 4  # 2 turns * 2 messages


@pytest.mark.asyncio
async def test_truncate_compactor():
    """TruncateCompactor keeps last N messages."""
    compactor = TruncateCompactor(keep_last=4)
    state = PipelineState()
    for i in range(10):
        state.messages.append({"role": "user", "content": f"msg{i}"})
    await compactor.compact(state)
    assert len(state.messages) == 4


# ── System Stage ──


@pytest.mark.asyncio
async def test_system_static_prompt():
    """SystemStage sets static prompt."""
    provider = MockProvider(default_text="OK")
    pipeline = Pipeline(PipelineConfig(name="sys"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(SystemStage(prompt="You are Geny VTuber."))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    await pipeline.run("Hi", state)
    assert state.system == "You are Geny VTuber."


@pytest.mark.asyncio
async def test_composable_prompt_builder():
    """ComposablePromptBuilder assembles blocks."""
    builder = ComposablePromptBuilder(
        blocks=[
            PersonaBlock("You are Geny, a friendly VTuber."),
            RulesBlock(["Be helpful", "Be concise"]),
            DateTimeBlock(),
        ]
    )

    state = PipelineState()
    prompt = builder.build(state)
    assert "Geny" in prompt
    assert "Be helpful" in prompt
    assert "Current date" in prompt


# ── Cache Stage ──


@pytest.mark.asyncio
async def test_system_cache_strategy():
    """SystemCacheStrategy adds cache_control to system."""
    strategy = SystemCacheStrategy()
    state = PipelineState()
    state.system = "You are helpful."

    strategy.apply_cache_markers(state)

    assert isinstance(state.system, list)
    assert state.system[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_aggressive_cache_strategy():
    """AggressiveCacheStrategy caches system + history."""
    strategy = AggressiveCacheStrategy(stable_history_offset=2)
    state = PipelineState()
    state.system = "System prompt"
    for i in range(6):
        state.messages.append({"role": "user", "content": f"msg{i}"})

    strategy.apply_cache_markers(state)

    # System should be cached
    assert isinstance(state.system, list)
    assert "cache_control" in state.system[0]


# ── Memory Stage ──


@pytest.mark.asyncio
async def test_memory_persistence():
    """MemoryStage persists messages."""
    persistence = InMemoryPersistence()
    provider = MockProvider(default_text="Stored")
    pipeline = Pipeline(PipelineConfig(name="mem"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(MemoryStage(persistence=persistence))
    pipeline.register_stage(YieldStage())

    state = PipelineState(session_id="test-123")
    await pipeline.run("Remember this", state)

    loaded = await persistence.load("test-123")
    assert len(loaded) >= 2  # at least user + assistant


# ── Session ──


@pytest.mark.asyncio
async def test_session_preserves_state():
    """Session preserves state across runs."""
    provider = MockProvider(default_text="Reply")
    pipeline = Pipeline(PipelineConfig(name="session"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    session = Session(pipeline=pipeline)
    await session.run("First")
    await session.run("Second")

    # State should accumulate messages from both runs
    assert len(session.state.messages) == 4  # 2 per run


@pytest.mark.asyncio
async def test_session_manager():
    """SessionManager creates and lists sessions."""
    pipeline = Pipeline(PipelineConfig(name="mgr"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(), retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    mgr = SessionManager()
    s1 = mgr.create(pipeline)
    mgr.create(pipeline)

    assert len(mgr) == 2
    assert mgr.get(s1.id) is s1

    sessions = mgr.list_sessions()
    assert len(sessions) == 2

    mgr.delete(s1.id)
    assert len(mgr) == 1
