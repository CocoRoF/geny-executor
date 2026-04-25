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
from geny_executor.stages.s03_system import SystemStage
from geny_executor.stages.s03_system.interface import PromptBuilder
from geny_executor.stages.s10_tool import ToolStage
from geny_executor.stages.s18_memory import MemoryStage
from geny_executor.stages.s18_memory.interface import (
    ConversationPersistence,
    MemoryUpdateStrategy,
)
from geny_executor.tools.base import ToolContext


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
    memory_stage = pipeline.get_stage(18)

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
    assert pipeline.get_stage(18)._strategy is strategy
    assert pipeline.get_stage(18)._persistence is persistence


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
    assert pipeline.get_stage(18) is None


def test_attach_runtime_none_kwargs_dont_overwrite():
    """Omitting a kwarg leaves the corresponding slot untouched."""
    pipeline = _make_pipeline_with_context_and_memory()

    first = _TrackedRetriever("first")
    pipeline.attach_runtime(memory_retriever=first)

    # Second call attaches only strategy — retriever should stay.
    strategy = _TrackedStrategy("mem")
    pipeline.attach_runtime(memory_strategy=strategy)

    assert pipeline.get_stage(2)._retriever is first
    assert pipeline.get_stage(18)._strategy is strategy


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
    baseline_strategy = pipeline.get_stage(18)._strategy

    pipeline.attach_runtime()

    assert pipeline.get_stage(2)._retriever is baseline_retriever
    assert pipeline.get_stage(18)._strategy is baseline_strategy


# ─── v0.26.0: system_builder / tool_context ───


class _TrackedSystemBuilder(PromptBuilder):
    """PromptBuilder stub that records invocation."""

    def __init__(self, label: str):
        self._label = label
        self.built = False

    @property
    def name(self) -> str:
        return f"tracked_builder:{self._label}"

    @property
    def description(self) -> str:
        return "test system builder"

    def build(self, state: Any) -> str:
        self.built = True
        return f"built by {self._label}"


def _make_pipeline_with_system_and_tool() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="attach-runtime-v26-test"))
    pipeline.register_stage(SystemStage())
    pipeline.register_stage(ToolStage())
    return pipeline


def test_attach_runtime_replaces_system_builder():
    pipeline = _make_pipeline_with_system_and_tool()
    system_stage = pipeline.get_stage(3)

    baseline = system_stage.get_strategy_slots()["builder"].strategy
    assert baseline.name == "static"

    builder = _TrackedSystemBuilder("sys")
    pipeline.attach_runtime(system_builder=builder)

    slot_strategy = system_stage.get_strategy_slots()["builder"].strategy
    assert slot_strategy is builder
    assert system_stage._builder is builder


def test_attach_runtime_replaces_tool_context():
    pipeline = _make_pipeline_with_system_and_tool()
    tool_stage = pipeline.get_stage(10)

    baseline_ctx = tool_stage._context
    assert isinstance(baseline_ctx, ToolContext)

    new_ctx = ToolContext(
        working_dir="/tmp/session-foo",
        storage_path="/tmp/session-foo/storage",
        metadata={"session_label": "sys-test"},
    )
    pipeline.attach_runtime(tool_context=new_ctx)

    assert tool_stage._context is new_ctx
    assert tool_stage._context.working_dir == "/tmp/session-foo"
    assert tool_stage._context.storage_path == "/tmp/session-foo/storage"
    assert tool_stage._context.metadata == {"session_label": "sys-test"}


def test_attach_runtime_system_builder_missing_stage_noop():
    """system_builder silently skipped when no SystemStage is registered."""
    pipeline = Pipeline(PipelineConfig(name="no-system"))
    pipeline.register_stage(ContextStage())  # no SystemStage

    pipeline.attach_runtime(system_builder=_TrackedSystemBuilder("nope"))

    assert pipeline.get_stage(3) is None


def test_attach_runtime_tool_context_missing_stage_noop():
    """tool_context silently skipped when no ToolStage is registered."""
    pipeline = Pipeline(PipelineConfig(name="no-tool"))
    pipeline.register_stage(ContextStage())  # no ToolStage

    pipeline.attach_runtime(tool_context=ToolContext(working_dir="/tmp/x"))

    assert pipeline.get_stage(10) is None


def test_attach_runtime_all_five_kwargs_together():
    """Attaching memory + system + tool runtime in one call wires them all."""
    pipeline = Pipeline(PipelineConfig(name="attach-runtime-full"))
    pipeline.register_stage(ContextStage())
    pipeline.register_stage(SystemStage())
    pipeline.register_stage(ToolStage())
    pipeline.register_stage(MemoryStage())

    retriever = _TrackedRetriever("full")
    strategy = _TrackedStrategy("full")
    persistence = _TrackedPersistence("full")
    builder = _TrackedSystemBuilder("full")
    ctx = ToolContext(working_dir="/tmp/full", storage_path="/tmp/full/store")

    pipeline.attach_runtime(
        memory_retriever=retriever,
        memory_strategy=strategy,
        memory_persistence=persistence,
        system_builder=builder,
        tool_context=ctx,
    )

    assert pipeline.get_stage(2)._retriever is retriever
    assert pipeline.get_stage(3)._builder is builder
    assert pipeline.get_stage(10)._context is ctx
    assert pipeline.get_stage(18)._strategy is strategy
    assert pipeline.get_stage(18)._persistence is persistence


@pytest.mark.asyncio
async def test_attach_runtime_after_run_raises_for_v26_kwargs():
    """system_builder / tool_context attach is also refused once pipeline has started."""
    pipeline = _make_pipeline_with_system_and_tool()
    pipeline._init_state(None)

    with pytest.raises(RuntimeError, match="attach_runtime"):
        pipeline.attach_runtime(system_builder=_TrackedSystemBuilder("late"))

    with pytest.raises(RuntimeError, match="attach_runtime"):
        pipeline.attach_runtime(tool_context=ToolContext(working_dir="/tmp/late"))
