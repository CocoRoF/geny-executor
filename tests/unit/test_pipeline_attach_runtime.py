"""Pipeline.attach_runtime() tests (v0.24.0).

Exercises the runtime injection helper used by hosts that build
pipelines from EnvironmentManifest (where runtime objects like
memory managers and LLM callbacks cannot live) and need to wire
session-scoped runtime objects in before the first run.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from geny_executor import Pipeline, PipelineConfig
from geny_executor.stages.s02_context import ContextStage
from geny_executor.stages.s02_context.interface import MemoryRetriever
from geny_executor.stages.s02_context.types import MemoryChunk
from geny_executor.stages.s15_memory import MemoryStage
from geny_executor.stages.s15_memory.interface import (
    ConversationPersistence,
    MemoryUpdateStrategy,
)


# ─── Test doubles ───


class _TrackedRetriever(MemoryRetriever):
    """MemoryRetriever stub that records whether it was invoked."""

    def __init__(self, label: str):
        self._label = label
        self.called = False

    @property
    def name(self) -> str:
        return f"tracked:{self._label}"

    @property
    def description(self) -> str:
        return "test retriever"

    async def retrieve(self, state: Any, query: str) -> List[MemoryChunk]:
        self.called = True
        return []


class _TrackedStrategy(MemoryUpdateStrategy):
    def __init__(self, label: str):
        self._label = label

    @property
    def name(self) -> str:
        return f"tracked_strategy:{self._label}"

    @property
    def description(self) -> str:
        return "test strategy"

    async def update(self, state: Any) -> None:  # pragma: no cover - interface signature only
        pass


class _TrackedPersistence(ConversationPersistence):
    def __init__(self, label: str):
        self._label = label

    @property
    def name(self) -> str:
        return f"tracked_persistence:{self._label}"

    @property
    def description(self) -> str:
        return "test persistence"

    async def save(self, session_id: str, messages: Any) -> None:  # pragma: no cover
        pass

    async def load(self, session_id: str) -> Any:  # pragma: no cover
        return []

    async def clear(self, session_id: str) -> None:  # pragma: no cover
        pass


def _make_pipeline_with_context_and_memory() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="attach-runtime-test"))
    pipeline.register_stage(ContextStage())
    pipeline.register_stage(MemoryStage())
    return pipeline


# ─── Tests ───


def test_attach_runtime_replaces_context_retriever():
    pipeline = _make_pipeline_with_context_and_memory()
    context_stage = pipeline.get_stage(2)

    baseline = context_stage.get_strategy_slots()["retriever"].strategy
    assert baseline.name == "null"

    retriever = _TrackedRetriever("ctx")
    pipeline.attach_runtime(memory_retriever=retriever)

    slot_strategy = context_stage.get_strategy_slots()["retriever"].strategy
    assert slot_strategy is retriever
    assert context_stage._retriever is retriever


def test_attach_runtime_replaces_memory_strategy_and_persistence():
    pipeline = _make_pipeline_with_context_and_memory()
    memory_stage = pipeline.get_stage(15)

    strategy = _TrackedStrategy("mem")
    persistence = _TrackedPersistence("mem")
    pipeline.attach_runtime(
        memory_strategy=strategy,
        memory_persistence=persistence,
    )

    assert memory_stage.get_strategy_slots()["strategy"].strategy is strategy
    assert memory_stage.get_strategy_slots()["persistence"].strategy is persistence
    assert memory_stage._strategy is strategy
    assert memory_stage._persistence is persistence


def test_attach_runtime_all_three_kwargs():
    pipeline = _make_pipeline_with_context_and_memory()
    retriever = _TrackedRetriever("full")
    strategy = _TrackedStrategy("full")
    persistence = _TrackedPersistence("full")

    pipeline.attach_runtime(
        memory_retriever=retriever,
        memory_strategy=strategy,
        memory_persistence=persistence,
    )

    assert pipeline.get_stage(2)._retriever is retriever
    assert pipeline.get_stage(15)._strategy is strategy
    assert pipeline.get_stage(15)._persistence is persistence


def test_attach_runtime_is_idempotent_before_first_run():
    """Calling attach_runtime repeatedly before run() is fine — last one wins."""
    pipeline = _make_pipeline_with_context_and_memory()

    first = _TrackedRetriever("first")
    second = _TrackedRetriever("second")

    pipeline.attach_runtime(memory_retriever=first)
    assert pipeline.get_stage(2)._retriever is first

    pipeline.attach_runtime(memory_retriever=second)
    assert pipeline.get_stage(2)._retriever is second


def test_attach_runtime_missing_stage_is_silent_noop():
    """A pipeline without a Memory stage can still attach_runtime — Memory args just skip."""
    pipeline = Pipeline(PipelineConfig(name="context-only"))
    pipeline.register_stage(ContextStage())
    # No MemoryStage.

    retriever = _TrackedRetriever("ctx")
    strategy = _TrackedStrategy("ignored")

    pipeline.attach_runtime(memory_retriever=retriever, memory_strategy=strategy)

    assert pipeline.get_stage(2)._retriever is retriever
    assert pipeline.get_stage(15) is None


def test_attach_runtime_none_kwargs_dont_overwrite():
    """Omitting a kwarg leaves the corresponding slot untouched."""
    pipeline = _make_pipeline_with_context_and_memory()

    first = _TrackedRetriever("first")
    pipeline.attach_runtime(memory_retriever=first)

    # Second call attaches only strategy — retriever should stay.
    strategy = _TrackedStrategy("mem")
    pipeline.attach_runtime(memory_strategy=strategy)

    assert pipeline.get_stage(2)._retriever is first
    assert pipeline.get_stage(15)._strategy is strategy


@pytest.mark.asyncio
async def test_attach_runtime_after_run_raises():
    """Once _init_state has fired, attach_runtime must refuse — prior state
    has already captured slot references and mixing them produces hard-to-
    reason-about runtime behavior."""
    pipeline = _make_pipeline_with_context_and_memory()

    # Simulate a run having happened — _init_state flips the flag.
    pipeline._init_state(None)

    with pytest.raises(RuntimeError, match="attach_runtime"):
        pipeline.attach_runtime(memory_retriever=_TrackedRetriever("late"))


def test_attach_runtime_no_args_is_valid_noop():
    """Calling with no kwargs is valid — nothing changes."""
    pipeline = _make_pipeline_with_context_and_memory()
    baseline_retriever = pipeline.get_stage(2)._retriever
    baseline_strategy = pipeline.get_stage(15)._strategy

    pipeline.attach_runtime()

    assert pipeline.get_stage(2)._retriever is baseline_retriever
    assert pipeline.get_stage(15)._strategy is baseline_strategy
