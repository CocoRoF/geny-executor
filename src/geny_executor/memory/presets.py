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


def _compose_persona_prompt(
    base_prompt: str,
    *,
    host_memory_clause: Optional[str] = None,
) -> str:
    """Combine the caller's persona prompt with the executor's
    generic memory-usage clause and the host-supplied tool catalogue.

    The executor's :data:`_MEMORY_USAGE_CLAUSE` is **tool-name free**
    by construction — it states the policy ("consult memory before
    asking; trust Pinned Facts") without naming any specific tools,
    because the executor cannot know which tools a host has wired
    in. The host provides its own catalogue verbatim via
    ``host_memory_clause`` (typically a 5-line bullet list of
    ``memory_*`` tool names with one-line semantics) and this
    function appends it after the executor's policy clause so the
    final system prompt reads:

        <user's prompt>

        ## Memory Usage         <- executor (generic policy)
        ...

        <host_memory_clause>    <- Geny (concrete tool catalogue)

    Hosts that don't supply ``host_memory_clause`` get just the
    policy. The agent then has to discover tools from its tool
    catalogue alone — degraded but not broken.
    """
    parts = [base_prompt.rstrip()] if base_prompt else []
    parts.append(_MEMORY_USAGE_CLAUSE)
    if host_memory_clause and host_memory_clause.strip():
        parts.append(host_memory_clause.strip())
    return "\n\n".join(parts)


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
        host_memory_clause: Optional[str] = None,
    ) -> Pipeline:
        """Worker (easy) — single-turn Q&A with memory context.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Parse → Memory → Yield

        Best for: simple questions that need memory context but no tools or loops.

        Args:
            host_memory_clause: Concrete tool catalogue + ladder
                description the host wants appended after the
                executor's generic memory-usage clause. Hosts that
                don't pass anything still get the policy half.
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

        composed = _compose_persona_prompt(
            system_prompt or _DEFAULT_WORKER_PROMPT,
            host_memory_clause=host_memory_clause,
        )
        builder = _build_system_builder(composed)

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
        host_memory_clause: Optional[str] = None,
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

        composed = _compose_persona_prompt(
            system_prompt or _DEFAULT_WORKER_PROMPT,
            host_memory_clause=host_memory_clause,
        )
        sys_builder = _build_system_builder(composed)

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
        host_memory_clause: Optional[str] = None,
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
        # then layer the executor's memory policy + host's tool
        # catalogue on top.
        full_prompt = (system_prompt or _DEFAULT_WORKER_PROMPT) + "\n\n" + _ADAPTIVE_PROMPT
        composed = _compose_persona_prompt(
            full_prompt, host_memory_clause=host_memory_clause,
        )
        sys_builder = _build_system_builder(composed)

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
        host_memory_clause: Optional[str] = None,
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

        composed = _compose_persona_prompt(
            persona_prompt or _DEFAULT_VTUBER_PROMPT,
            host_memory_clause=host_memory_clause,
        )
        sys_builder = _build_system_builder(composed)

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

# Memory v2 PR 15 (cycle 20260503_7). The previous incarnation of
# this clause hard-coded host-specific tool names (``memory_search``
# / ``memory_read`` / ``memory_categories`` / …) directly into the
# executor preset. That violated the boundary the rest of this
# package keeps: the executor ships generic mechanisms, the host
# (Geny etc.) ships concrete categorisation and tool naming.
#
# The rewrite carries only **policy** here — what to do, not which
# tools to call. The host injects its own tool catalogue verbatim
# via the ``host_memory_clause`` preset kwarg; ``_compose_persona_prompt``
# appends it after this clause so the final system prompt reads:
#
#     <user's prompt>
#
#     ## Memory Usage          <- this constant (policy only)
#     ...
#
#     <host_memory_clause>     <- host (concrete tool catalogue)
#
# A host that doesn't supply a clause still gets the policy; the
# agent then discovers concrete tools from its tool catalogue
# alone (degraded but functional).
_MEMORY_USAGE_CLAUSE = """\
## Memory Usage

The host maintains a long-term memory for you. The "Pinned Facts"
section in this prompt — when present — holds must-know facts
about the user, the agent, and the ongoing work. Treat them as
authoritative; never claim ignorance of anything stated there.

When the user's intent is ambiguous and the answer might already
be remembered, **consult memory before asking a clarification
question the user may have already answered.** Your tool
catalogue lists the read / search / write tools the host has
wired in for this — use them.

Do not announce the lookup; just do it."""


_DEFAULT_WORKER_PROMPT = """\
You are an autonomous AI agent. Complete the user's task step by step.

When you have finished the task, end your response with [TASK_COMPLETE].
If you need to continue working, end with [CONTINUE: next action].
If you are blocked and cannot proceed, end with [BLOCKED: reason].

Be thorough, accurate, and concise."""


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

Keep responses conversational and natural."""
