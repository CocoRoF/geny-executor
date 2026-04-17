"""Pipeline presets — pre-configured pipeline patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from geny_executor.core.environment import EnvironmentManager


@dataclass
class PresetInfo:
    """Metadata about a preset (built-in or user-defined)."""

    name: str
    description: str = ""
    preset_type: str = "built_in"  # "built_in" | "user"
    tags: List[str] = field(default_factory=list)
    environment_id: Optional[str] = None


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


# ═══════════════════════════════════════════════════════════
#  PresetManager — built-in + user presets
# ═══════════════════════════════════════════════════════════


class PresetManager:
    """Manages pipeline presets (built-in + user-defined from environments)."""

    _BUILT_IN_DESCRIPTIONS: Dict[str, str] = {
        "minimal": "Minimal Q&A pipeline — Input → API → Parse → Yield",
        "chat": "Chat pipeline — history, system prompt, optional tools",
        "agent": "Agent pipeline — full autonomous agent with all stages",
        "evaluator": "Evaluator pipeline — lightweight evaluation",
        "geny_vtuber": "Geny VTuber pipeline — full Geny system reproduction",
    }

    def __init__(self, env_manager: EnvironmentManager) -> None:
        self._env_manager = env_manager
        self._built_in_factories = {
            "minimal": PipelinePresets.minimal,
            "chat": PipelinePresets.chat,
            "agent": PipelinePresets.agent,
            "evaluator": PipelinePresets.evaluator,
            "geny_vtuber": PipelinePresets.geny_vtuber,
        }

    def list_all(self) -> List[PresetInfo]:
        """List built-in + user-defined presets."""
        presets: List[PresetInfo] = []

        # Built-in
        for name in self._built_in_factories:
            presets.append(
                PresetInfo(
                    name=name,
                    description=self._BUILT_IN_DESCRIPTIONS.get(name, ""),
                    preset_type="built_in",
                )
            )

        # User presets (environments tagged "preset")
        for env in self._env_manager.list_all():
            if "preset" in env.tags:
                presets.append(
                    PresetInfo(
                        name=f"user:{env.id}",
                        description=env.description,
                        preset_type="user",
                        tags=env.tags,
                        environment_id=env.id,
                    )
                )

        return presets

    def save_as_preset(self, env_id: str) -> None:
        """Mark an environment as a reusable preset."""
        manifest = self._env_manager.load(env_id)
        if "preset" not in manifest.metadata.tags:
            manifest.metadata.tags.append("preset")
            self._env_manager.update(env_id, {"metadata": {"tags": manifest.metadata.tags}})

    def remove_preset_flag(self, env_id: str) -> None:
        """Un-mark an environment as a preset."""
        manifest = self._env_manager.load(env_id)
        if "preset" in manifest.metadata.tags:
            manifest.metadata.tags.remove("preset")
            self._env_manager.update(env_id, {"metadata": {"tags": manifest.metadata.tags}})
