"""geny-executor: Harness-engineered agent pipeline library.

Usage:
    from geny_executor import Pipeline, PipelineConfig
    from geny_executor.stages.s01_input import InputStage
    from geny_executor.stages.s06_api import APIStage, MockProvider
    from geny_executor.stages.s09_parse import ParseStage
    from geny_executor.stages.s16_yield import YieldStage

    pipeline = Pipeline(PipelineConfig(name="my-agent"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(api_key="..."))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello!")
"""

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.config import PipelineConfig, ModelConfig
from geny_executor.core.state import PipelineState, TokenUsage, CacheMetrics
from geny_executor.core.result import PipelineResult
from geny_executor.core.stage import Stage, Strategy, StageDescription, StrategyInfo
from geny_executor.core.errors import (
    GenyExecutorError,
    PipelineError,
    StageError,
    GuardRejectError,
    APIError,
    ToolExecutionError,
    ErrorCategory,
)
from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.presets import PipelinePresets
from geny_executor.core.artifact import create_stage, list_artifacts, get_artifact_map
from geny_executor.events import EventBus, PipelineEvent
from geny_executor.memory import (
    GenyMemoryRetriever,
    GenyMemoryStrategy,
    GenyPersistence,
    GenyPresets,
)

__version__ = "0.6.0"

__all__ = [
    # Core
    "Pipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineResult",
    "ModelConfig",
    "TokenUsage",
    "CacheMetrics",
    # Abstractions
    "Stage",
    "Strategy",
    "StageDescription",
    "StrategyInfo",
    # Builder & Presets
    "PipelineBuilder",
    "PipelinePresets",
    # Artifact system
    "create_stage",
    "list_artifacts",
    "get_artifact_map",
    # Events
    "EventBus",
    "PipelineEvent",
    # Errors
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "APIError",
    "ToolExecutionError",
    "ErrorCategory",
    # Geny Memory Integration
    "GenyMemoryRetriever",
    "GenyMemoryStrategy",
    "GenyPersistence",
    "GenyPresets",
]
