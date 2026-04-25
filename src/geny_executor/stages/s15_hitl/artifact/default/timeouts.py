"""Default HITL timeout policies for Stage 15 (S9b.3)."""

from __future__ import annotations

from typing import Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl.interface import TimeoutPolicy
from geny_executor.stages.s15_hitl.types import HITLDecision, HITLRequest


class IndefiniteTimeout(TimeoutPolicy):
    """Wait forever. Useful for fully manual approval flows."""

    @property
    def name(self) -> str:
        return "indefinite"

    @property
    def description(self) -> str:
        return "Wait indefinitely for the requester to return"

    @property
    def timeout_seconds(self) -> Optional[float]:
        return None

    def on_timeout(self, request: HITLRequest, state: PipelineState) -> HITLDecision:
        # Should never be called when timeout_seconds is None, but
        # provide a sane fallback so callers that bypass the contract
        # don't crash.
        return HITLDecision.CANCEL


class AutoApproveTimeout(TimeoutPolicy):
    """Approve once the timeout fires."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout = float(timeout_seconds)

    @property
    def name(self) -> str:
        return "auto_approve"

    @property
    def description(self) -> str:
        return "Approve automatically after timeout_seconds"

    @property
    def timeout_seconds(self) -> Optional[float]:
        return self._timeout

    def configure(self, config: dict) -> None:
        timeout = config.get("timeout_seconds")
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            self._timeout = float(timeout)

    def on_timeout(self, request: HITLRequest, state: PipelineState) -> HITLDecision:
        return HITLDecision.APPROVE


class AutoRejectTimeout(TimeoutPolicy):
    """Reject once the timeout fires."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout = float(timeout_seconds)

    @property
    def name(self) -> str:
        return "auto_reject"

    @property
    def description(self) -> str:
        return "Reject automatically after timeout_seconds"

    @property
    def timeout_seconds(self) -> Optional[float]:
        return self._timeout

    def configure(self, config: dict) -> None:
        timeout = config.get("timeout_seconds")
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            self._timeout = float(timeout)

    def on_timeout(self, request: HITLRequest, state: PipelineState) -> HITLDecision:
        return HITLDecision.REJECT


__all__ = [
    "AutoApproveTimeout",
    "AutoRejectTimeout",
    "IndefiniteTimeout",
]
