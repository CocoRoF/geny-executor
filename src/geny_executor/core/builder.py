"""PipelineBuilder — declarative pipeline construction."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.config import ModelConfig, PipelineConfig
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.registry import ToolRegistry


class PipelineBuilder:
    """Declarative pipeline builder.

    Usage:
        pipeline = (
            PipelineBuilder("my-agent", api_key="sk-...")
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
        self._model = model or "claude-sonnet-4-20250514"
        self._config_kwargs: Dict[str, Any] = {}
        self._stage_configs: Dict[str, Dict[str, Any]] = {}
        self._tool_registry: Optional[ToolRegistry] = None

    def with_model(self, model: str, **kwargs: Any) -> PipelineBuilder:
        self._model = model
        self._config_kwargs.update(kwargs)
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

    def build(self) -> Pipeline:
        """Build the pipeline with all configured stages."""
        config = PipelineConfig(
            name=self._name,
            api_key=self._api_key,
            model=ModelConfig(model=self._model),
            **self._config_kwargs,
        )

        pipeline = Pipeline(config)

        # Always register: Input, API, Parse, Yield
        from geny_executor.stages.s01_input import InputStage
        from geny_executor.stages.s06_api import APIStage
        from geny_executor.stages.s07_token import TokenStage
        from geny_executor.stages.s09_parse import ParseStage
        from geny_executor.stages.s16_yield import YieldStage

        pipeline.register_stage(InputStage())
        pipeline.register_stage(APIStage(api_key=self._api_key))
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
            pipeline.register_stage(SystemStage(
                tool_registry=self._tool_registry,
                **self._stage_configs["system"],
            ))

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
            pipeline.register_stage(ToolStage(
                registry=self._tool_registry,
                **tool_cfg,
            ))

        # Agent
        if "agent" in self._stage_configs:
            from geny_executor.stages.s11_agent import AgentStage
            pipeline.register_stage(AgentStage(**self._stage_configs["agent"]))

        # Evaluate
        if "evaluate" in self._stage_configs:
            from geny_executor.stages.s12_evaluate import EvaluateStage
            pipeline.register_stage(EvaluateStage(**self._stage_configs["evaluate"]))

        # Loop
        if "loop" in self._stage_configs:
            from geny_executor.stages.s13_loop import LoopStage, StandardLoopController
            loop_cfg = dict(self._stage_configs["loop"])  # Copy to avoid mutation
            max_turns = loop_cfg.pop("max_turns", 50)
            pipeline.register_stage(LoopStage(
                StandardLoopController(max_turns=max_turns)
            ))

        # Emit
        if "emit" in self._stage_configs:
            from geny_executor.stages.s14_emit import EmitStage
            pipeline.register_stage(EmitStage(**self._stage_configs["emit"]))

        # Memory
        if "memory" in self._stage_configs:
            from geny_executor.stages.s15_memory import MemoryStage
            pipeline.register_stage(MemoryStage(**self._stage_configs["memory"]))

        return pipeline

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
