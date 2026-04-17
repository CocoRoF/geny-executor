"""Core engine: Pipeline, Stage, Strategy, State, Config, Result, Errors, Mutation, Environment, Diff."""

from geny_executor.core.errors import (
    ErrorCategory,
    GenyExecutorError,
    GuardRejectError,
    MutationError,
    MutationLocked,
    PipelineError,
    StageError,
)
from geny_executor.core.stage import Stage, Strategy
from geny_executor.core.state import CacheMetrics, PipelineState, TokenUsage
from geny_executor.core.config import ModelConfig, PipelineConfig
from geny_executor.core.result import PipelineResult
from geny_executor.core.pipeline import Pipeline
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.mutation import MutationKind, MutationRecord, MutationResult, PipelineMutator
from geny_executor.core.diff import DiffEntry, EnvironmentDiff
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentManager,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    ToolsSnapshot,
)
from geny_executor.core.presets import PipelinePresets, PresetInfo, PresetManager

__all__ = [
    # Engine
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineState",
    "Stage",
    "Strategy",
    "ModelConfig",
    "TokenUsage",
    "CacheMetrics",
    # Schema
    "ConfigField",
    "ConfigSchema",
    # Slot
    "StrategySlot",
    # Snapshot
    "PipelineSnapshot",
    "StageSnapshot",
    # Mutation
    "PipelineMutator",
    "MutationKind",
    "MutationRecord",
    "MutationResult",
    # Diff
    "DiffEntry",
    "EnvironmentDiff",
    # Environment
    "EnvironmentManifest",
    "EnvironmentManager",
    "EnvironmentMetadata",
    "EnvironmentResolver",
    "EnvironmentSanitizer",
    "EnvironmentSummary",
    "ToolsSnapshot",
    # Presets
    "PipelinePresets",
    "PresetInfo",
    "PresetManager",
    # Errors
    "ErrorCategory",
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "MutationError",
    "MutationLocked",
]
