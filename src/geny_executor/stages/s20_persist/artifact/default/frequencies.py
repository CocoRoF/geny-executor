"""Default frequency policies for Stage 20 (S9b.5).

Three flavours covering common cadence shapes. Hosts pick whichever
matches their durability needs and plug into the stage's frequency
slot.
"""

from __future__ import annotations

from typing import List, Tuple

from geny_executor.core.state import PipelineState
from geny_executor.stages.s20_persist.interface import FrequencyPolicy


class EveryTurnFrequency(FrequencyPolicy):
    """Persist on every turn. Highest durability, highest IO cost."""

    @property
    def name(self) -> str:
        return "every_turn"

    @property
    def description(self) -> str:
        return "Write a checkpoint every turn"

    def should_persist(self, state: PipelineState) -> bool:
        return True


class EveryNTurnsFrequency(FrequencyPolicy):
    """Persist every N iterations.

    Turn 0, N, 2N, ... fire. Bounded by iteration count alone — no
    inspection of state contents.
    """

    def __init__(self, n: int = 5) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = int(n)

    @property
    def name(self) -> str:
        return "every_n_turns"

    @property
    def description(self) -> str:
        return "Write a checkpoint every N iterations"

    @property
    def n(self) -> int:
        return self._n

    def configure(self, config: dict) -> None:
        if "n" in config:
            n = int(config["n"])
            if n < 1:
                raise ValueError("n must be >= 1")
            self._n = n

    def should_persist(self, state: PipelineState) -> bool:
        return state.iteration % self._n == 0


# Default significance triggers.
_DEFAULT_SIGNIFICANT_EVENTS: Tuple[str, ...] = (
    "hitl.decision",
    "hitl.timeout",
    "tool_review.flag",
    "memory.insight_recorded",
    "summary.written",
    "task.failed",
)


class OnSignificantFrequency(FrequencyPolicy):
    """Persist only when a "significant" signal fires this turn.

    A turn is significant when:

    * Any event in ``significant_events`` was emitted this turn
      (matched against ``state.events`` and the event's
      ``iteration`` field).
    * OR ``state.shared['tool_review_flags']`` contains an
      ``error``-severity flag.
    * OR ``state.shared['turn_summary'].importance`` is HIGH/CRITICAL
      (when ``escalate_on_high_importance`` is True).
    * OR ``state.completion_signal`` is non-empty (terminal turn).
    """

    def __init__(
        self,
        *,
        significant_events: List[str] | None = None,
        escalate_on_high_importance: bool = True,
    ) -> None:
        self._events = frozenset(significant_events or _DEFAULT_SIGNIFICANT_EVENTS)
        self._escalate_importance = bool(escalate_on_high_importance)

    @property
    def name(self) -> str:
        return "on_significant"

    @property
    def description(self) -> str:
        return "Write a checkpoint when this turn produced a significant signal"

    def _has_event_this_turn(self, state: PipelineState) -> bool:
        iteration = state.iteration
        for evt in state.events:
            if evt.get("type") not in self._events:
                continue
            if int(evt.get("iteration", -1)) == iteration:
                return True
        return False

    def _has_review_error(self, state: PipelineState) -> bool:
        flags = state.shared.get("tool_review_flags") or []
        for flag in flags:
            severity = (
                flag.get("severity") if isinstance(flag, dict) else getattr(flag, "severity", "")
            )
            if severity == "error":
                return True
        return False

    def _has_high_importance_summary(self, state: PipelineState) -> bool:
        record = state.shared.get("turn_summary")
        if record is None:
            return False
        importance = getattr(record, "importance", None)
        if importance is None and isinstance(record, dict):
            importance_value = record.get("importance")
        else:
            importance_value = getattr(importance, "value", importance)
        return importance_value in ("high", "critical")

    def should_persist(self, state: PipelineState) -> bool:
        if state.completion_signal:
            return True
        if self._has_event_this_turn(state):
            return True
        if self._has_review_error(state):
            return True
        if self._escalate_importance and self._has_high_importance_summary(state):
            return True
        return False


__all__ = [
    "EveryNTurnsFrequency",
    "EveryTurnFrequency",
    "OnSignificantFrequency",
]
