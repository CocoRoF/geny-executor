"""Default artifact for Stage 14: Emit."""

from geny_executor.stages.s14_emit.artifact.default.stage import EmitStage
from geny_executor.stages.s14_emit.artifact.default.emitters import (
    TextEmitter,
    CallbackEmitter,
    VTuberEmitter,
    TTSEmitter,
)

Stage = EmitStage

__all__ = [
    "Stage",
    "EmitStage",
    "TextEmitter",
    "CallbackEmitter",
    "VTuberEmitter",
    "TTSEmitter",
]
