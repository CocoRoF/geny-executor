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
from geny_executor.tools.base import ToolContext
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
        model: str = "claude-sonnet-4-6",
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
            .with_tool_review()
            .with_task_registry()
            .with_hitl()
            .with_memory(strategy=strategy, persistence=persistence)
            .with_summarize()
            .with_persist()
            .build()
        )

    @staticmethod
    def worker_full(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-6",
        system_prompt: str = "",
        tools: Optional[ToolRegistry] = None,
        tool_context: Optional[ToolContext] = None,
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
            .with_tool_review()
            .with_task_registry()
            .with_hitl()
            .with_evaluate()
            .with_loop(max_turns=max_turns)
            .with_memory(strategy=strategy, persistence=persistence)
            .with_summarize()
            .with_persist()
        )

        if tools:
            tool_kwargs: Dict[str, Any] = {}
            if tool_context:
                tool_kwargs["context"] = tool_context
            pipeline_builder = pipeline_builder.with_tools(registry=tools, **tool_kwargs)

        return pipeline_builder.build()

    @staticmethod
    def worker_adaptive(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-6",
        system_prompt: str = "",
        tools: Optional[ToolRegistry] = None,
        tool_context: Optional[ToolContext] = None,
        max_turns: int = 30,
        easy_max_turns: int = 1,
        max_inject_chars: int = 10000,
        enable_reflection: bool = True,
        llm_reflect: Optional[Callable[[str, str], Awaitable[List[Dict[str, Any]]]]] = None,
        llm_gate: Optional[Callable[[str], Awaitable[bool]]] = None,
        curated_knowledge_manager: Any = None,
    ) -> Pipeline:
        """Worker (adaptive) — binary classify + autonomous execution.

        Auto-classifies tasks on the first turn:
          - easy: 1-turn direct answer (no tools, minimal tokens)
          - not_easy: multi-turn loop with TODO decomposition + tool use

        Replaces the old template-optimized-autonomous workflow graph
        with a single Pipeline using BinaryClassifyEvaluation strategy.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Think → Parse → Tool
                       → Evaluate(BinaryClassify) → Loop → Memory → Yield
        """
        from geny_executor.stages.s14_evaluate.artifact.adaptive.strategy import (
            BinaryClassifyConfig,
            BinaryClassifyEvaluation,
        )

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

        # Combine user prompt with adaptive execution instructions
        full_prompt = (system_prompt or _DEFAULT_WORKER_PROMPT) + "\n\n" + _ADAPTIVE_PROMPT
        sys_builder = _build_system_builder(full_prompt)

        classify_config = BinaryClassifyConfig(
            easy_max_turns=easy_max_turns,
            not_easy_max_turns=max_turns,
        )
        eval_strategy = BinaryClassifyEvaluation(classify_config)

        pipeline_builder = (
            PipelineBuilder("worker-adaptive", api_key=api_key, model=model)
            .with_context(retriever=retriever)
            .with_system(builder=sys_builder)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_tool_review()
            .with_task_registry()
            .with_hitl()
            .with_evaluate(strategy=eval_strategy)
            .with_loop(max_turns=max_turns)
            .with_memory(strategy=strategy, persistence=persistence)
            .with_summarize()
            .with_persist()
        )

        if tools:
            tool_kwargs: Dict[str, Any] = {}
            if tool_context:
                tool_kwargs["context"] = tool_context
            pipeline_builder = pipeline_builder.with_tools(registry=tools, **tool_kwargs)

        return pipeline_builder.build()

    @staticmethod
    def vtuber(
        api_key: str,
        memory_manager: Any,
        *,
        model: str = "claude-sonnet-4-6",
        persona_prompt: str = "",
        max_inject_chars: int = 8000,
        enable_reflection: bool = True,
        llm_reflect: Optional[Callable[[str, str], Awaitable[List[Dict[str, Any]]]]] = None,
        tools: Optional[ToolRegistry] = None,
        tool_context: Optional[ToolContext] = None,
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
            .with_tool_review()
            .with_task_registry()
            .with_hitl()
            .with_evaluate()
            .with_loop(max_turns=10)
            .with_memory(strategy=strategy, persistence=persistence)
            .with_summarize()
            .with_persist()
        )

        if tools:
            tool_kwargs: Dict[str, Any] = {}
            if tool_context:
                tool_kwargs["context"] = tool_context
            pipeline_builder = pipeline_builder.with_tools(registry=tools, **tool_kwargs)

        return pipeline_builder.build()


# ── Default Prompts ──────────────────────────────────────────────────

# Memory v2 PR 12 — generic memory-usage clause appended to every
# default preset. PR 14 (cycle 20260503_6) refines the clause around
# **progressive disclosure**: the agent doesn't get the entire
# vault — it gets a category map (vault map) plus a tool ladder it
# walks down only as needed.
#
# The clause stays tool-name-driven and category-agnostic so the
# executor stays generic; concrete hosts (Geny) decide what lives
# in their pinned surface and which categories exist.
_MEMORY_USAGE_CLAUSE = """\
## Memory Usage

You can recall and grow long-term memory through these tools the
host exposes:

  - `memory_categories` — Tier 1: list every memory category with
    a 1-line description, file count, and last-modified timestamp.
    This is your *map of the vault*.
  - `memory_list(category=…)` — Tier 2: see the files inside one
    category (filename, title, summary, importance, modified).
  - `memory_read(filename=…)` — Tier 3: open a specific note's
    full body.
  - `memory_search(query=…)` — fuzzy / semantic search when you
    have a query but don't know which folder owns the answer.
  - `memory_write` / `memory_pin` / `memory_update` — write paths.

**Progressive disclosure rule.** When you need to recall something,
walk the ladder: read the system-prompt vault map first, pick a
category, call `memory_list`, then `memory_read` on the matching
file. Reach for `memory_search` only when the vault map doesn't
make the right folder obvious.

The "Pinned Facts" section in this prompt — when present — holds
the must-know facts about the user, the agent, and the ongoing
work. Treat it as authoritative; never claim ignorance of anything
stated there.

When the user's intent is ambiguous and the answer might already
be remembered, **walk the memory ladder BEFORE asking the user a
clarification question they may have answered before.**

Do not announce the search; just use it."""


_DEFAULT_WORKER_PROMPT = """\
You are an autonomous AI agent. Complete the user's task step by step.

When you have finished the task, end your response with [TASK_COMPLETE].
If you need to continue working, end with [CONTINUE: next action].
If you are blocked and cannot proceed, end with [BLOCKED: reason].

Be thorough, accurate, and concise.

""" + _MEMORY_USAGE_CLAUSE

_ADAPTIVE_PROMPT = """\
## Execution Strategy

Classify the task and act accordingly:

**Easy tasks** (factual Q&A, simple lookups, greetings, short explanations):
Answer directly in one response. Do not use tools unless absolutely necessary.

**Complex tasks** (coding, research, multi-step work, file operations):
1. Plan: Decompose into clear steps
2. Execute: Use tools to complete each step
3. Verify: Check your work
4. Signal [CONTINUE: next step] after each step
5. Signal [TASK_COMPLETE] when all steps are done"""

_DEFAULT_VTUBER_PROMPT = """\
You are a friendly AI VTuber assistant. Engage in natural conversation
while being helpful and knowledgeable.

When the user asks a complex task that requires tools or multi-step work,
indicate that you will delegate it.

Keep responses conversational and natural.

""" + _MEMORY_USAGE_CLAUSE
