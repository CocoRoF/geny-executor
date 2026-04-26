"""Auto-compaction frequency policies (PR-B.2.1).

A :class:`FrequencyPolicy` decides whether the configured Summarizer
should run on the current turn. Default summarisers run every turn;
adding a policy lets a host say "only summarise when context is
≥80% full" or "every 5 turns".

Wrap any existing Summarizer in :class:`FrequencyAwareSummarizerProxy`
to gate it. Hosts that prefer to drive the policy externally use
:meth:`FrequencyPolicy.should_fire` directly.

Three reference policies:

* :class:`NeverPolicy`         — never fire (effectively disables)
* :class:`EveryNTurnsPolicy`   — fire on iteration % n == 0
* :class:`OnContextFillPolicy` — fire when used_tokens / max_context
                                 ≥ ``threshold`` and ``min_turns_between``
                                 has elapsed since the last fire
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s19_summarize.interface import Summarizer
from geny_executor.stages.s19_summarize.types import SummaryRecord


@dataclass
class FrequencyContext:
    """What policies need to decide. Built from :class:`PipelineState`
    by :meth:`FrequencyPolicy.from_state`.
    """

    iteration: int
    last_fired_iteration: Optional[int]
    input_tokens: int
    output_tokens: int
    max_context_tokens: int


class FrequencyPolicy(ABC):
    """Strategy for "should the summariser run this turn?"."""

    @abstractmethod
    def should_fire(self, ctx: FrequencyContext) -> bool: ...

    @staticmethod
    def from_state(
        state: PipelineState,
        *,
        last_fired_iteration: Optional[int] = None,
    ) -> FrequencyContext:
        # Token + context numbers are best-effort; not every host
        # wires a token accountant.
        shared = getattr(state, "shared", {}) or {}
        return FrequencyContext(
            iteration=getattr(state, "iteration", 0),
            last_fired_iteration=last_fired_iteration,
            input_tokens=int(shared.get("input_tokens", 0)),
            output_tokens=int(shared.get("output_tokens", 0)),
            max_context_tokens=int(shared.get("max_context_tokens", 0)),
        )


class NeverPolicy(FrequencyPolicy):
    def should_fire(self, ctx: FrequencyContext) -> bool:
        return False


class EveryNTurnsPolicy(FrequencyPolicy):
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be > 0")
        self._n = n

    def should_fire(self, ctx: FrequencyContext) -> bool:
        if ctx.iteration <= 0:
            return False
        return ctx.iteration % self._n == 0


class OnContextFillPolicy(FrequencyPolicy):
    """Fire when used / max ≥ threshold AND ``min_turns_between`` elapsed.

    Without ``min_turns_between``, sitting just past the threshold for
    several turns would re-fire constantly (each summarisation barely
    drops the ratio). Default 5 keeps the cost predictable.
    """

    def __init__(self, threshold: float = 0.8, min_turns_between: int = 5) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if min_turns_between < 0:
            raise ValueError("min_turns_between must be >= 0")
        self._threshold = threshold
        self._min_between = min_turns_between

    def should_fire(self, ctx: FrequencyContext) -> bool:
        if ctx.max_context_tokens <= 0:
            return False
        if ctx.last_fired_iteration is not None:
            if ctx.iteration - ctx.last_fired_iteration < self._min_between:
                return False
        used = ctx.input_tokens + ctx.output_tokens
        ratio = used / ctx.max_context_tokens
        return ratio >= self._threshold


class FrequencyAwareSummarizerProxy(Summarizer):
    """Wrap any Summarizer with a FrequencyPolicy gate.

    On a "skip" turn the proxy returns ``None`` so the stage publishes
    nothing — same shape as :class:`NoSummarizer`. On a "fire" turn
    the inner summariser runs normally and the proxy stamps its own
    iteration counter so the next ``should_fire`` sees the right
    ``last_fired_iteration``.
    """

    def __init__(
        self,
        inner: Summarizer,
        policy: FrequencyPolicy,
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._last_fired_iteration: Optional[int] = None

    @property
    def name(self) -> str:
        return f"frequency_aware({self._inner.name})"

    @property
    def description(self) -> str:
        return (
            f"Summariser {self._inner.name} gated by "
            f"{type(self._policy).__name__}"
        )

    async def summarize(self, state: PipelineState) -> Optional[SummaryRecord]:
        ctx = FrequencyPolicy.from_state(
            state, last_fired_iteration=self._last_fired_iteration,
        )
        if not self._policy.should_fire(ctx):
            return None
        result = await self._inner.summarize(state)
        if result is not None:
            self._last_fired_iteration = ctx.iteration
        return result


__all__ = [
    "EveryNTurnsPolicy",
    "FrequencyAwareSummarizerProxy",
    "FrequencyContext",
    "FrequencyPolicy",
    "NeverPolicy",
    "OnContextFillPolicy",
]
