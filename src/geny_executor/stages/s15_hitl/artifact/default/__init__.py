"""Default artifact for Stage 15: HITL (S9b.3)."""

from geny_executor.stages.s15_hitl.artifact.default.requesters import (
    CallbackRequester,
    NullRequester,
)
from geny_executor.stages.s15_hitl.artifact.default.stage import HITLStage
from geny_executor.stages.s15_hitl.artifact.default.timeouts import (
    AutoApproveTimeout,
    AutoRejectTimeout,
    IndefiniteTimeout,
)

Stage = HITLStage

__all__ = [
    "AutoApproveTimeout",
    "AutoRejectTimeout",
    "CallbackRequester",
    "HITLStage",
    "IndefiniteTimeout",
    "NullRequester",
    "Stage",
]
