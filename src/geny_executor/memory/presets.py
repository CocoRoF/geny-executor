"""GenyPresets — Pre-configured pipelines for Geny execution modes.

Provides two production presets that replace Geny's execution logic:
  - worker: Autonomous agent with tools, loop, and full memory
  - vtuber: Conversational agent with memory reflection

Both presets integrate with Geny's SessionMemoryManager for
5-layer memory retrieval and structured note persistence.

CRITICAL: Uses ComposablePromptBuilder (not StaticPromptBuilder)
to ensure memory context from S02 Context is injected into the
system prompt via MemoryContextBlock.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional

from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.pipeline import Pipeline
from geny_executor.memory.retriever import GenyMemoryRetriever
from geny_executor.memory.strategy import GenyMemoryStrategy
from geny_executor.memory.persistence import GenyPersistence
from geny_executor.tools.registry import ToolRegistry


def _build_system_builder(
    prompt: str,
    include_memory: bool = True,
    include_datetime: bool = True,
):
    """Build a ComposablePromptBuilder that includes memory context.

    Unlike StaticPromptBuilder, this builder reads state.metadata["memory_context"]
    and injects it into the system prompt so the LLM can see retrieved memory.
    """
    from geny_executor.stages.s03_system.artifact.default.builders import (
        ComposablePromptBuilder,
        DateTimeBlock,
        MemoryContextBlock,
        PersonaBlock,
    )

    blocks = [PersonaBlock(prompt)]

    if include_datetime:
        blocks.append(DateTimeBlock())

    if include_memory:
        blocks.append(MemoryContextBlock())

    return ComposablePromptBuilder(blocks=blocks)


class GenyPresets:
    """Pre-configured pipelines for Geny execution modes."""

    @staticmethod
    def worker_easy(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "",
        max_inject_chars: int = 10000,
    ) -> Pipeline:
        """Worker (easy) — single-turn Q&A with memory context.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Parse → Memory → Yield

        Best for: simple questions that need memory context but no tools or loops.
        """
        retriever = GenyMemoryRetriever(
            memory_manager,
            max_inject_chars=max_inject_chars,
            enable_vector_search=True,
        )
        strategy = GenyMemoryStrategy(
            memory_manager,
            enable_reflection=False,
        )
        persistence = GenyPersistence(memory_manager)

        builder = _build_system_builder(system_prompt or _DEFAULT_WORKER_PROMPT)

        return (
            PipelineBuilder("worker-easy", api_key=api_key, model=model)
            .with_context(retriever=retriever)
            .with_system(builder=builder)
            .with_guard()
            .with_cache(strategy="system")
            .with_memory(strategy=strategy, persistence=persistence)
            .build()
        )

    @staticmethod
    def worker_full(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "",
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 50,
        max_inject_chars: int = 10000,
        enable_reflection: bool = True,
        llm_reflect: Optional[Callable[[str, str], Awaitable[List[Dict[str, Any]]]]] = None,
        llm_gate: Optional[Callable[[str], Awaitable[bool]]] = None,
        curated_knowledge_manager: Any = None,
    ) -> Pipeline:
        """Worker (full) — autonomous agent with all stages.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Think → Parse → Tool
                       → Evaluate → Loop → Memory → Yield

        Best for: complex tasks that require tools, multi-turn loops,
        and full memory integration.
        """
        retriever = GenyMemoryRetriever(
            memory_manager,
            max_inject_chars=max_inject_chars,
            enable_vector_search=True,
            llm_gate=llm_gate,
            curated_knowledge_manager=curated_knowledge_manager,
        )
        strategy = GenyMemoryStrategy(
            memory_manager,
            enable_reflection=enable_reflection,
            llm_reflect=llm_reflect,
            curated_knowledge_manager=curated_knowledge_manager,
        )
        persistence = GenyPersistence(memory_manager)

        sys_builder = _build_system_builder(system_prompt or _DEFAULT_WORKER_PROMPT)

        pipeline_builder = (
            PipelineBuilder("worker-full", api_key=api_key, model=model)
            .with_context(retriever=retriever)
            .with_system(builder=sys_builder)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_evaluate()
            .with_loop(max_turns=max_turns)
            .with_memory(strategy=strategy, persistence=persistence)
        )

        if tools:
            pipeline_builder = pipeline_builder.with_tools(registry=tools)

        return pipeline_builder.build()

    @staticmethod
    def vtuber(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        persona_prompt: str = "",
        max_inject_chars: int = 8000,
        enable_reflection: bool = True,
        llm_reflect: Optional[Callable[[str, str], Awaitable[List[Dict[str, Any]]]]] = None,
        tools: Optional[ToolRegistry] = None,
        curated_knowledge_manager: Any = None,
    ) -> Pipeline:
        """VTuber — conversational agent with persona and memory reflection.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Parse → [Tool] → Memory → Yield

        Best for: VTuber persona with conversation memory and
        post-execution insight extraction.
        """
        retriever = GenyMemoryRetriever(
            memory_manager,
            max_inject_chars=max_inject_chars,
            enable_vector_search=True,
            curated_knowledge_manager=curated_knowledge_manager,
        )
        strategy = GenyMemoryStrategy(
            memory_manager,
            enable_reflection=enable_reflection,
            llm_reflect=llm_reflect,
            curated_knowledge_manager=curated_knowledge_manager,
        )
        persistence = GenyPersistence(memory_manager)

        sys_builder = _build_system_builder(persona_prompt or _DEFAULT_VTUBER_PROMPT)

        pipeline_builder = (
            PipelineBuilder("vtuber", api_key=api_key, model=model)
            .with_context(retriever=retriever)
            .with_system(builder=sys_builder)
            .with_guard()
            .with_cache(strategy="system")
            .with_memory(strategy=strategy, persistence=persistence)
        )

        if tools:
            pipeline_builder = pipeline_builder.with_tools(registry=tools)

        return pipeline_builder.build()


# ── Default Prompts ──────────────────────────────────────────────────

_DEFAULT_WORKER_PROMPT = """\
You are an autonomous AI agent. Complete the user's task step by step.

When you have finished the task, end your response with [TASK_COMPLETE].
If you need to continue working, end with [CONTINUE: next action].
If you are blocked and cannot proceed, end with [BLOCKED: reason].

Be thorough, accurate, and concise."""

_DEFAULT_VTUBER_PROMPT = """\
You are a friendly AI VTuber assistant. Engage in natural conversation
while being helpful and knowledgeable.

When the user asks a complex task that requires tools or multi-step work,
indicate that you will delegate it.

Keep responses conversational and natural."""
