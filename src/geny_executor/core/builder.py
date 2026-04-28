"""PipelineBuilder — declarative pipeline construction."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Optional

from geny_executor.core.config import ModelConfig, PipelineConfig
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.registry import ToolRegistry

# ModelConfig field names — used to route kwargs correctly in build().
_MODEL_CONFIG_FIELDS: set[str] = {f.name for f in dataclass_fields(ModelConfig)}


class PipelineBuilder:
    """Declarative pipeline builder.

    Usage::

        pipeline = (
            PipelineBuilder("my-agent", api_key="sk-...")
            .with_model("claude-sonnet-4-6",
                         max_tokens=4096, temperature=0.7)
            .with_system(prompt="You are helpful.")
            .with_tools(registry=my_tools)
            .with_cache(strategy="system")
            .with_loop(max_turns=20)
            .build()
        )
    """

    def __init__(self, name: str = "default", *, api_key: str = "", model: str = ""):
        self._name = name
        self._api_key = api_key
        self._model = model or "claude-sonnet-4-6"
        self._model_kwargs: Dict[str, Any] = {}
        self._config_kwargs: Dict[str, Any] = {}
        self._stage_configs: Dict[str, Dict[str, Any]] = {}
        self._tool_registry: Optional[ToolRegistry] = None
        self._artifact_overrides: Dict[str, str] = {}

    def with_artifact(self, stage: str, artifact: str) -> PipelineBuilder:
        """Select a specific artifact for a stage.

        Args:
            stage: Stage identifier (e.g., "s06_api", "api", "6").
            artifact: Artifact name (folder under artifact/).
        """
        self._artifact_overrides[stage] = artifact
        return self

    def with_model(self, model: str, **kwargs: Any) -> PipelineBuilder:
        """Set model name and optional ModelConfig overrides.

        ModelConfig fields (max_tokens, temperature, top_p, top_k,
        stop_sequences, thinking_enabled, thinking_budget_tokens,
        thinking_type, thinking_display) are routed to ModelConfig.

        Other kwargs are routed to PipelineConfig.
        """
        self._model = model
        for key, value in kwargs.items():
            if key in _MODEL_CONFIG_FIELDS:
                self._model_kwargs[key] = value
            else:
                self._config_kwargs[key] = value
        return self

    def with_system(self, prompt: str = "", **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["system"] = {"prompt": prompt, **kwargs}
        return self

    def with_tools(self, registry: ToolRegistry, **kwargs: Any) -> PipelineBuilder:
        self._tool_registry = registry
        self._stage_configs["tool"] = kwargs
        return self

    def with_guard(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["guard"] = kwargs
        return self

    def with_cache(self, strategy: str = "system", **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["cache"] = {"strategy": strategy, **kwargs}
        return self

    def with_context(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["context"] = kwargs
        return self

    def with_memory(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["memory"] = kwargs
        return self

    def with_loop(self, max_turns: int = 50, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["loop"] = {"max_turns": max_turns, **kwargs}
        return self

    def with_think(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["think"] = kwargs
        return self

    def with_agent(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["agent"] = kwargs
        return self

    def with_evaluate(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["evaluate"] = kwargs
        return self

    def with_emit(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["emit"] = kwargs
        return self

    # ── Sub-phase 9a scaffolds (S9a.5) ──
    # The five new stages added by S9a.2/3 default to pass-through /
    # bypass behaviour. Calling these builder methods is *opt-in*
    # registration — they wire the scaffold into the pipeline so it
    # appears in describe()/introspection and is ready to swap in a
    # real implementation when Sub-phase 9b lands. Without calling
    # them the pipeline is identical to pre-9a behaviour.

    def with_tool_review(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["tool_review"] = kwargs
        return self

    def with_task_registry(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["task_registry"] = kwargs
        return self

    def with_hitl(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["hitl"] = kwargs
        return self

    def with_summarize(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["summarize"] = kwargs
        return self

    def with_persist(self, **kwargs: Any) -> PipelineBuilder:
        self._stage_configs["persist"] = kwargs
        return self

    def build(self) -> Pipeline:
        """Build the pipeline with all configured stages."""
        model_config = ModelConfig(model=self._model, **self._model_kwargs)

        config = PipelineConfig(
            name=self._name,
            api_key=self._api_key,
            model=model_config,
            artifacts=dict(self._artifact_overrides),
            **self._config_kwargs,
        )

        pipeline = Pipeline(config)

        # Always register: Input, API, Parse, Yield
        from geny_executor.stages.s01_input import InputStage
        from geny_executor.stages.s07_token import TokenStage
        from geny_executor.stages.s09_parse import ParseStage
        from geny_executor.stages.s21_yield import YieldStage

        pipeline.register_stage(InputStage())
        pipeline.register_stage(self._build_api_stage(config))
        pipeline.register_stage(TokenStage())
        pipeline.register_stage(ParseStage())
        pipeline.register_stage(YieldStage())

        # Context
        if "context" in self._stage_configs:
            from geny_executor.stages.s02_context import ContextStage

            pipeline.register_stage(ContextStage(**self._stage_configs["context"]))

        # System
        if "system" in self._stage_configs:
            from geny_executor.stages.s03_system import SystemStage

            pipeline.register_stage(
                SystemStage(
                    tool_registry=self._tool_registry,
                    **self._stage_configs["system"],
                )
            )

        # Guard
        if "guard" in self._stage_configs:
            from geny_executor.stages.s04_guard import GuardStage

            pipeline.register_stage(GuardStage(**self._stage_configs["guard"]))

        # Cache
        if "cache" in self._stage_configs:
            from geny_executor.stages.s05_cache import CacheStage

            cache_cfg = dict(self._stage_configs["cache"])  # Copy to avoid mutation
            strategy_name = cache_cfg.pop("strategy", "no_cache")
            strategy = self._resolve_cache_strategy(strategy_name)
            pipeline.register_stage(CacheStage(strategy=strategy))

        # Think
        if "think" in self._stage_configs:
            from geny_executor.stages.s08_think import ThinkStage

            pipeline.register_stage(ThinkStage(**self._stage_configs["think"]))

        # Tool
        if self._tool_registry:
            from geny_executor.stages.s10_tool import ToolStage

            tool_cfg = dict(self._stage_configs.get("tool", {}))
            pipeline.register_stage(
                ToolStage(
                    registry=self._tool_registry,
                    **tool_cfg,
                )
            )

        # Agent
        if "agent" in self._stage_configs:
            from geny_executor.stages.s12_agent import AgentStage

            pipeline.register_stage(AgentStage(**self._stage_configs["agent"]))

        # Evaluate
        if "evaluate" in self._stage_configs:
            from geny_executor.stages.s14_evaluate import EvaluateStage

            pipeline.register_stage(EvaluateStage(**self._stage_configs["evaluate"]))

        # Loop
        if "loop" in self._stage_configs:
            from geny_executor.stages.s16_loop import LoopStage

            loop_cfg = dict(self._stage_configs["loop"])
            pipeline.register_stage(LoopStage(**loop_cfg))

        # Emit
        if "emit" in self._stage_configs:
            from geny_executor.stages.s17_emit import EmitStage

            pipeline.register_stage(EmitStage(**self._stage_configs["emit"]))

        # Memory
        if "memory" in self._stage_configs:
            from geny_executor.stages.s18_memory import MemoryStage

            pipeline.register_stage(MemoryStage(**self._stage_configs["memory"]))

        # ── Sub-phase 9a scaffolds (S9a.5) ──
        # Each only registers when the host opts in via the matching
        # `with_*` method. Defaults are pass-through / bypass so a
        # registered scaffold is a no-op until 9b replaces it.

        if "tool_review" in self._stage_configs:
            from geny_executor.stages.s11_tool_review import ToolReviewStage

            pipeline.register_stage(ToolReviewStage(**self._stage_configs["tool_review"]))

        if "task_registry" in self._stage_configs:
            from geny_executor.stages.s13_task_registry import TaskRegistryStage

            pipeline.register_stage(TaskRegistryStage(**self._stage_configs["task_registry"]))

        if "hitl" in self._stage_configs:
            from geny_executor.stages.s15_hitl import HITLStage

            pipeline.register_stage(HITLStage(**self._stage_configs["hitl"]))

        if "summarize" in self._stage_configs:
            from geny_executor.stages.s19_summarize import SummarizeStage

            pipeline.register_stage(SummarizeStage(**self._stage_configs["summarize"]))

        if "persist" in self._stage_configs:
            from geny_executor.stages.s20_persist import PersistStage

            pipeline.register_stage(PersistStage(**self._stage_configs["persist"]))

        return pipeline

    def _build_api_stage(self, config: PipelineConfig) -> Any:
        """Build the API stage, selecting provider by artifact or model name."""
        # Check if user explicitly specified an artifact for s06_api
        artifact = config.artifacts.get("s06_api") or config.artifacts.get("api")

        # Auto-detect from model name if no explicit artifact
        if not artifact:
            artifact = self._infer_api_artifact()

        if artifact and artifact != "default":
            from geny_executor.core.artifact import create_stage

            return create_stage("s06_api", artifact=artifact, api_key=self._api_key)

        # Default: Anthropic
        from geny_executor.stages.s06_api import APIStage

        return APIStage(api_key=self._api_key)

    def _infer_api_artifact(self) -> Optional[str]:
        """Infer API provider artifact from model name prefix."""
        m = self._model.lower()
        if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            return "openai"
        if m.startswith(("gemini-",)):
            return "google"
        # claude-*, or unknown → default (Anthropic)
        return None

    def _resolve_cache_strategy(self, name: str) -> Any:
        from geny_executor.stages.s05_cache.strategies import (
            NoCacheStrategy,
            SystemCacheStrategy,
            AggressiveCacheStrategy,
        )

        strategies = {
            "no_cache": NoCacheStrategy,
            "system": SystemCacheStrategy,
            "aggressive": AggressiveCacheStrategy,
        }
        cls = strategies.get(name, NoCacheStrategy)
        return cls()
