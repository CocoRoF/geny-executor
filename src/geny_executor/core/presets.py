"""Pipeline presets — pre-configured pipeline patterns."""

from __future__ import annotations

from typing import Optional

from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.registry import ToolRegistry


class PipelinePresets:
    """Pre-configured pipeline patterns for common use cases."""

    @staticmethod
    def minimal(api_key: str, model: str = "claude-sonnet-4-20250514") -> Pipeline:
        """Minimal pipeline — simple Q&A.

        Active stages: Input → API → Parse → Yield
        """
        return PipelineBuilder("minimal", api_key=api_key, model=model).build()

    @staticmethod
    def chat(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[ToolRegistry] = None,
    ) -> Pipeline:
        """Chat pipeline — history, system prompt, optional tools.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Parse → Tool → Loop → Memory → Yield
        """
        builder = (
            PipelineBuilder("chat", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=system_prompt)
            .with_guard()
            .with_cache(strategy="system")
            .with_loop(max_turns=20)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()

    @staticmethod
    def agent(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "You are an autonomous agent. Complete the task step by step.",
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 50,
    ) -> Pipeline:
        """Agent pipeline — full autonomous agent with all stages.

        Active stages: All 16 stages
        """
        builder = (
            PipelineBuilder("agent", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=system_prompt)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_evaluate()
            .with_loop(max_turns=max_turns)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()

    @staticmethod
    def evaluator(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        evaluation_prompt: str = "Evaluate the following response for quality, accuracy, and completeness.",
    ) -> Pipeline:
        """Evaluator pipeline — lightweight evaluation for Generator/Evaluator pattern.

        Active stages: Input → System → API → Parse → Evaluate → Yield
        """
        return (
            PipelineBuilder("evaluator", api_key=api_key, model=model)
            .with_system(prompt=evaluation_prompt)
            .with_evaluate()
            .build()
        )

    @staticmethod
    def geny_vtuber(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        persona: str = "You are Geny, a friendly AI VTuber.",
        tools: Optional[ToolRegistry] = None,
    ) -> Pipeline:
        """Geny VTuber pipeline — full Geny system reproduction.

        Active stages: All 16 stages + VTuber/TTS emitters
        """
        builder = (
            PipelineBuilder("geny-vtuber", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=persona)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_evaluate()
            .with_loop(max_turns=50)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()
