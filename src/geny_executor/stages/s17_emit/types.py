"""Emit stage data types."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from geny_executor.stages.s17_emit.interface import Emitter
    from geny_executor.core.state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class EmitResult:
    """Result of emission.

    ``emitter_name`` is added in S7.11 for tracking — :class:`OrderedEmitterChain`
    populates it on every result so callers can pair results with the
    emitter that produced them. The legacy :class:`EmitterChain` leaves
    it blank for unchanged behaviour.
    """

    emitted: bool = True
    channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    emitter_name: str = ""


class EmitterChain:
    """Chain of emitters — runs all in sequence."""

    def __init__(self, emitters: Optional[List[Emitter]] = None):
        self._emitters = emitters or []

    def add(self, emitter: Emitter) -> None:
        self._emitters.append(emitter)

    async def emit_all(self, state: PipelineState) -> List[EmitResult]:
        results = []
        for emitter in self._emitters:
            try:
                result = await emitter.emit(state)
                results.append(result)
            except Exception as e:
                logger.warning("Emitter %s failed: %s", emitter.name, e)
                results.append(
                    EmitResult(emitted=False, channels=[emitter.name], metadata={"error": str(e)})
                )
        return results

    @property
    def emitters(self) -> List[Emitter]:
        return list(self._emitters)


class OrderedEmitterChain:
    """Emitter chain with topological ordering and timeout-based backpressure (S7.11).

    Behaviour vs the legacy :class:`EmitterChain`:

    * **Ordering** — emitters are reordered by Kahn's topological sort
      using their :attr:`Emitter.requires` declarations. Cycles fall
      back to declared order and emit ``emit.cycle_detected``. Unknown
      ``requires`` names (a dep that isn't in the chain) are skipped
      with ``emit.unknown_dependency`` and the emitter is treated as
      having no deps so a typo cannot wedge the whole chain.
    * **Failure isolation** — same as the legacy chain: per-emitter
      try/except, exceptions become ``emitted=False`` results.
    * **Dependency-failure skip** — if any of an emitter's
      ``requires`` produced a non-``emitted`` result earlier in the
      same pass, the dependent is skipped with metadata
      ``{"skipped": "dep_failed", "deps": [...]}`` and an
      ``emit.skipped_dep_failed`` event.
    * **Backpressure** — per-emitter consecutive timeout count.
      Exceeding ``backpressure_threshold`` skips the emitter on this
      and subsequent passes until :meth:`reset_backpressure` is called
      or a successful emit on the same emitter resets the counter.
      Skipped emissions emit ``emit.skipped_backpressure``.
    * **Timeout** — if an emitter declares
      :attr:`Emitter.timeout_seconds`, the chain wraps the emit in
      :func:`asyncio.wait_for`. A timeout increments the backpressure
      counter and emits ``emit.timeout``.
    """

    DEFAULT_BACKPRESSURE_THRESHOLD = 3

    def __init__(
        self,
        emitters: Optional[List[Emitter]] = None,
        *,
        backpressure_threshold: int = DEFAULT_BACKPRESSURE_THRESHOLD,
    ) -> None:
        if backpressure_threshold < 1:
            raise ValueError("backpressure_threshold must be >= 1")
        self._emitters: List[Emitter] = list(emitters or [])
        self._threshold = int(backpressure_threshold)
        self._consecutive_timeouts: Dict[str, int] = {}

    def add(self, emitter: Emitter) -> None:
        self._emitters.append(emitter)

    @property
    def emitters(self) -> List[Emitter]:
        return list(self._emitters)

    @property
    def consecutive_timeouts(self) -> Dict[str, int]:
        return dict(self._consecutive_timeouts)

    def reset_backpressure(self, emitter_name: Optional[str] = None) -> None:
        """Clear the consecutive-timeout counter for one emitter (or all)."""
        if emitter_name is None:
            self._consecutive_timeouts.clear()
        else:
            self._consecutive_timeouts.pop(emitter_name, None)

    def _topological_order(self, state: "PipelineState") -> List[Emitter]:
        """Kahn's sort. Returns declared order on cycle, with an event."""
        names = {e.name for e in self._emitters}
        # Validate requires up front
        clean_requires: Dict[str, List[str]] = {}
        for em in self._emitters:
            valid: List[str] = []
            for req in em.requires:
                if req in names:
                    valid.append(req)
                else:
                    state.add_event(
                        "emit.unknown_dependency",
                        {"emitter": em.name, "dependency": req},
                    )
            clean_requires[em.name] = valid

        # Build adjacency
        in_degree: Dict[str, int] = {em.name: len(clean_requires[em.name]) for em in self._emitters}
        dependents: Dict[str, List[str]] = {em.name: [] for em in self._emitters}
        for em in self._emitters:
            for dep in clean_requires[em.name]:
                dependents[dep].append(em.name)

        # Process in declared order to keep stable output for ties.
        ready = [em for em in self._emitters if in_degree[em.name] == 0]
        ordered: List[Emitter] = []
        em_by_name = {em.name: em for em in self._emitters}
        while ready:
            em = ready.pop(0)
            ordered.append(em)
            for child in dependents[em.name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(em_by_name[child])

        if len(ordered) != len(self._emitters):
            # Cycle — fall back to declared order, log it.
            state.add_event(
                "emit.cycle_detected",
                {
                    "ordered_count": len(ordered),
                    "total": len(self._emitters),
                    "emitters": [em.name for em in self._emitters],
                },
            )
            return list(self._emitters)
        return ordered

    async def emit_all(self, state: "PipelineState") -> List[EmitResult]:
        ordered = self._topological_order(state)
        results: List[EmitResult] = []
        emitted_ok: Dict[str, bool] = {}

        for emitter in ordered:
            name = emitter.name

            # Backpressure: skip if over the consecutive-timeout threshold.
            if self._consecutive_timeouts.get(name, 0) >= self._threshold:
                results.append(
                    EmitResult(
                        emitted=False,
                        channels=[name],
                        metadata={
                            "skipped": "backpressure",
                            "consecutive_timeouts": self._consecutive_timeouts[name],
                        },
                        emitter_name=name,
                    )
                )
                state.add_event(
                    "emit.skipped_backpressure",
                    {"emitter": name, "consecutive_timeouts": self._consecutive_timeouts[name]},
                )
                emitted_ok[name] = False
                continue

            # Dep-failure skip: if any required emitter didn't emit, skip.
            failed_deps = [req for req in emitter.requires if not emitted_ok.get(req, False)]
            # Only count deps that actually ran in this pass.
            failed_deps = [d for d in failed_deps if d in emitted_ok]
            if failed_deps:
                results.append(
                    EmitResult(
                        emitted=False,
                        channels=[name],
                        metadata={"skipped": "dep_failed", "deps": failed_deps},
                        emitter_name=name,
                    )
                )
                state.add_event(
                    "emit.skipped_dep_failed",
                    {"emitter": name, "deps": failed_deps},
                )
                emitted_ok[name] = False
                continue

            try:
                if emitter.timeout_seconds is not None and emitter.timeout_seconds > 0:
                    result = await asyncio.wait_for(
                        emitter.emit(state), timeout=emitter.timeout_seconds
                    )
                else:
                    result = await emitter.emit(state)
            except asyncio.TimeoutError:
                self._consecutive_timeouts[name] = self._consecutive_timeouts.get(name, 0) + 1
                state.add_event(
                    "emit.timeout",
                    {
                        "emitter": name,
                        "timeout_seconds": emitter.timeout_seconds,
                        "consecutive_timeouts": self._consecutive_timeouts[name],
                    },
                )
                results.append(
                    EmitResult(
                        emitted=False,
                        channels=[name],
                        metadata={
                            "error": "timeout",
                            "consecutive_timeouts": self._consecutive_timeouts[name],
                        },
                        emitter_name=name,
                    )
                )
                emitted_ok[name] = False
                continue
            except Exception as exc:  # noqa: BLE001 — chain-wide failure isolation
                logger.warning("Emitter %s failed: %s", name, exc)
                # Non-timeout exceptions don't count toward backpressure
                # (backpressure is about latency, not correctness bugs).
                results.append(
                    EmitResult(
                        emitted=False,
                        channels=[name],
                        metadata={"error": str(exc)},
                        emitter_name=name,
                    )
                )
                emitted_ok[name] = False
                continue

            # Success — reset the per-emitter timeout counter.
            self._consecutive_timeouts.pop(name, None)
            if not result.emitter_name:
                result.emitter_name = name
            results.append(result)
            emitted_ok[name] = bool(result.emitted)

        return results
