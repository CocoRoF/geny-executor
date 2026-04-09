"""Emitters — backward-compatible re-exports."""

from geny_executor.stages.s14_emit.interface import Emitter
from geny_executor.stages.s14_emit.types import EmitResult, EmitterChain
from geny_executor.stages.s14_emit.artifact.default.emitters import (
    TextEmitter,
    CallbackEmitter,
    VTuberEmitter,
    TTSEmitter,
)

__all__ = [
    "Emitter",
    "EmitResult",
    "EmitterChain",
    "TextEmitter",
    "CallbackEmitter",
    "VTuberEmitter",
    "TTSEmitter",
]
