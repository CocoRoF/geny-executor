"""Default artifact controllers for Stage 13: Loop."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens
from geny_executor.stages.s16_loop.interface import LoopController, LoopDecision

logger = logging.getLogger(__name__)


def _require_number(strategy: str, key: str, value: Any, *, minimum: float = 0.0) -> float:
    """Shared configure() validation: a real number >= minimum.

    ``bool`` is rejected explicitly — it subclasses ``int``, so a manifest
    typo like ``{"max_cost_usd": true}`` would otherwise silently become
    ``1.0`` and pass every later check.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{strategy}: {key!r} must be a number >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{strategy}: {key!r} must be >= {minimum}, got {value!r}")
    return float(value)


def _require_int(strategy: str, key: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{strategy}: {key!r} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{strategy}: {key!r} must be >= {minimum}, got {value!r}")
    return value


class StandardLoopController(LoopController):
    """Standard loop controller — tool_use continues, signals decide."""

    def __init__(self, max_turns: Optional[int] = None):
        self._max_turns = max_turns

    @property
    def name(self) -> str:
        return "standard"

    @property
    def description(self) -> str:
        return "Standard loop: tool_use continues, signals decide"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="standard",
            fields=[
                ConfigField(
                    name="max_turns",
                    type="integer",
                    label="Max turns",
                    description="Hard cap on loop iterations. 0 = defer to state.max_iterations.",
                    default=0,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        if "max_turns" in config:
            v = _require_int("standard", "max_turns", config["max_turns"], minimum=0)
            self._max_turns = v or None

    def get_config(self) -> Dict[str, Any]:
        return {"max_turns": self._max_turns or 0}

    def decide(self, state: PipelineState) -> str:
        if state.tool_results:
            return LoopDecision.CONTINUE

        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE
        if signal == "error":
            return LoopDecision.ERROR

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        max_t = self._max_turns or state.max_iterations
        if state.iteration >= max_t:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE


class SingleTurnController(LoopController):
    """Single turn — always complete after one pass."""

    @property
    def name(self) -> str:
        return "single_turn"

    @property
    def description(self) -> str:
        return "Always complete after one turn (no loop)"

    def decide(self, state: PipelineState) -> str:
        return LoopDecision.COMPLETE


class BudgetAwareLoopController(LoopController):
    """Budget-aware — stops if cost/token budget is low."""

    def __init__(self, cost_threshold_ratio: float = 0.9, token_threshold_ratio: float = 0.85):
        self._cost_ratio = cost_threshold_ratio
        self._token_ratio = token_threshold_ratio

    @property
    def name(self) -> str:
        return "budget_aware"

    @property
    def description(self) -> str:
        return "Stops when approaching budget limits"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="budget_aware",
            fields=[
                ConfigField(
                    name="cost_threshold_ratio",
                    type="number",
                    label="Cost threshold ratio",
                    description="Stop when total cost reaches this fraction of state.cost_budget_usd.",
                    default=0.9,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ConfigField(
                    name="token_threshold_ratio",
                    type="number",
                    label="Token threshold ratio",
                    description="Stop when token usage reaches this fraction of the context window budget.",
                    default=0.85,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        cost_ratio = self._cost_ratio
        token_ratio = self._token_ratio
        if "cost_threshold_ratio" in config:
            cost_ratio = _require_number(
                "budget_aware", "cost_threshold_ratio", config["cost_threshold_ratio"]
            )
            if cost_ratio > 1.0:
                raise ValueError(
                    f"budget_aware: 'cost_threshold_ratio' must be in [0.0, 1.0], got {cost_ratio!r}"
                )
        if "token_threshold_ratio" in config:
            token_ratio = _require_number(
                "budget_aware", "token_threshold_ratio", config["token_threshold_ratio"]
            )
            if token_ratio > 1.0:
                raise ValueError(
                    f"budget_aware: 'token_threshold_ratio' must be in [0.0, 1.0], "
                    f"got {token_ratio!r}"
                )
        self._cost_ratio = cost_ratio
        self._token_ratio = token_ratio

    def get_config(self) -> Dict[str, Any]:
        return {
            "cost_threshold_ratio": self._cost_ratio,
            "token_threshold_ratio": self._token_ratio,
        }

    def decide(self, state: PipelineState) -> str:
        if (
            state.cost_budget_usd
            and state.total_cost_usd >= state.cost_budget_usd * self._cost_ratio
        ):
            return LoopDecision.COMPLETE

        # audit R2: measure the ACTUAL next-request size (system +
        # messages + tools), NOT the session-cumulative token_usage —
        # which on a reused state crosses the window within a few turns
        # and then permanently forces COMPLETE, ending turns before the
        # model even sees its tool results.
        used = estimate_prompt_tokens(state)
        if used >= state.context_window_budget * self._token_ratio:
            return LoopDecision.COMPLETE

        if state.tool_results:
            return LoopDecision.CONTINUE

        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE


# ─────────────────────────────────────────────────────────────────
# Phase 7 Sprint S7.7 — Multi-dimensional budget
# ─────────────────────────────────────────────────────────────────


class BudgetDimension(ABC):
    """One dimension of a multi-dimensional loop budget.

    Subclasses inspect ``state`` and return ``True`` when their budget
    has been exhausted. ``MultiDimensionalBudgetController`` consults
    every registered dimension and stops the loop the moment any one
    of them returns True. ``name`` is surfaced in the
    ``loop.budget_exceeded`` event payload so admins can see *which*
    budget tripped.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable short identifier used in logs + events."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary; used in stage descriptions."""
        return self.name

    @abstractmethod
    def is_exceeded(self, state: PipelineState) -> bool:
        """Return True when this dimension has been exhausted."""
        ...


class IterationBudget(BudgetDimension):
    """Hard cap on loop iterations.

    ``max_iterations=None`` (new in 2.2.0) defers to the session-level
    ``state.max_iterations`` — the same fallback ``StandardLoopController``
    uses. This lets a manifest declare the *dimension* ("budget the loop on
    iterations") without duplicating the cap that already lives at the
    pipeline level (Geny prod declares exactly that shape).
    """

    def __init__(self, max_iterations: Optional[int] = None):
        self._max = max(1, int(max_iterations)) if max_iterations is not None else None

    @property
    def name(self) -> str:
        return "iteration"

    @property
    def description(self) -> str:
        if self._max is None:
            return "≤ state.max_iterations"
        return f"≤ {self._max} iterations"

    def is_exceeded(self, state: PipelineState) -> bool:
        cap = self._max if self._max is not None else state.max_iterations
        if not cap or cap <= 0:
            return False
        return state.iteration >= cap


class CostBudget(BudgetDimension):
    """Soft cap on cumulative USD cost.

    ``threshold_ratio`` lets hosts stop short of the absolute ceiling
    so the next turn doesn't blow past it. Default ``0.9`` matches
    the legacy ``BudgetAwareLoopController``.

    ``max_usd=None`` (new in 2.2.0) defers to the session-level
    ``state.cost_budget_usd`` — mirrors ``BudgetAwareLoopController``
    so a manifest can declare the dimension without re-stating the cap.
    """

    def __init__(self, max_usd: Optional[float] = None, *, threshold_ratio: float = 0.9):
        self._max_usd = float(max_usd) if max_usd is not None else None
        self._ratio = float(threshold_ratio)

    @property
    def name(self) -> str:
        return "cost"

    @property
    def description(self) -> str:
        if self._max_usd is None:
            return f"≤ state.cost_budget_usd (stop at {self._ratio:.0%})"
        return f"≤ ${self._max_usd:.2f} (stop at {self._ratio:.0%})"

    def is_exceeded(self, state: PipelineState) -> bool:
        cap = self._max_usd if self._max_usd is not None else (state.cost_budget_usd or 0.0)
        if cap <= 0:
            return False
        return state.total_cost_usd >= cap * self._ratio


class TokenBudget(BudgetDimension):
    """Soft cap on the context window.

    Compares the ACTUAL next-request size (``estimate_prompt_tokens`` —
    system + messages + tools) against ``state.context_window_budget``
    (the model's window) by default, OR an explicit ``max_tokens``
    override. (audit R2: reading the session-cumulative
    ``token_usage.total_tokens`` measured the wrong thing — a long
    session's cumulative usage permanently exceeds any per-request
    window, freezing the loop.)
    """

    def __init__(
        self,
        *,
        max_tokens: Optional[int] = None,
        threshold_ratio: float = 0.85,
    ):
        self._max_tokens = int(max_tokens) if max_tokens is not None else None
        self._ratio = float(threshold_ratio)

    @property
    def name(self) -> str:
        return "tokens"

    @property
    def description(self) -> str:
        if self._max_tokens is not None:
            return f"≤ {self._max_tokens} tokens (stop at {self._ratio:.0%})"
        return f"≤ context_window_budget (stop at {self._ratio:.0%})"

    def is_exceeded(self, state: PipelineState) -> bool:
        used = estimate_prompt_tokens(state)
        cap = self._max_tokens if self._max_tokens is not None else state.context_window_budget
        if cap <= 0:
            return False
        return used >= cap * self._ratio


class WallClockBudget(BudgetDimension):
    """Cap on real time elapsed since session start.

    Reads ``state.created_at`` (set at PipelineState construction)
    so the budget covers everything from session creation through
    the current loop check — including any time the host spent
    setting things up before run().
    """

    def __init__(
        self,
        max_seconds: float,
        *,
        clock: Optional[callable] = None,
    ):
        self._max_seconds = float(max_seconds)
        # ``clock`` is injectable for deterministic tests.
        self._clock = clock or time.monotonic
        # Capture a startup-relative origin in case the state's
        # ``created_at`` (datetime) drifts under clock skew. We
        # compare against monotonic time deltas at evaluation time.
        self._origin = self._clock()

    @property
    def name(self) -> str:
        return "wall_clock"

    @property
    def description(self) -> str:
        return f"≤ {self._max_seconds:.1f}s wall clock"

    def is_exceeded(self, state: PipelineState) -> bool:
        if self._max_seconds <= 0:
            return False
        elapsed = self._clock() - self._origin
        return elapsed >= self._max_seconds


class ToolCallBudget(BudgetDimension):
    """Hard cap on CUMULATIVE tool calls executed across the turn.

    Reads the running counter Stage 10 maintains at
    ``shared["executor.tool_calls_total"]``. (audit R2: the pre-2.51
    version counted ``len(state.tool_results)``, which Stage 10 REPLACES
    every round — so the cap only ever saw one round's calls and a
    runaway-agent guard of e.g. 20 never tripped. Falls back to the
    per-round count when the counter is absent, e.g. standalone use.)
    """

    def __init__(self, max_calls: int):
        self._max = max(1, int(max_calls))

    @property
    def name(self) -> str:
        return "tool_calls"

    @property
    def description(self) -> str:
        return f"≤ {self._max} tool calls"

    def is_exceeded(self, state: PipelineState) -> bool:
        total = state.shared.get("executor.tool_calls_total")
        if not isinstance(total, int):
            total = len(state.tool_results)
        return total >= self._max


#: Dimension names spellable in ``strategy_configs["controller"]["dimensions"]``.
#: Maps accepted spelling → canonical key. Geny prod manifests say
#: ``"iterations"`` / ``"cost_usd"`` / ``"walltime_seconds"`` (audit §2.1
#: file evidence: default_manifest.py loop entry) while the dimension
#: classes report singular ``"iteration"`` etc. — both spellings resolve
#: so the live manifests work without a host-side rewrite.
_DIMENSION_ALIASES: Dict[str, str] = {
    "iteration": "iteration",
    "iterations": "iteration",
    "cost": "cost",
    "cost_usd": "cost",
    "token": "tokens",
    "tokens": "tokens",
    "wall_clock": "wall_clock",
    "walltime_seconds": "wall_clock",
    "tool_calls": "tool_calls",
}


class MultiDimensionalBudgetController(LoopController):
    """Loop controller backed by a list of pluggable budget dimensions.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.7.

    The pre-S7.7 :class:`BudgetAwareLoopController` hard-coded two
    dimensions (cost + tokens) at fixed ratios. Hosts that needed a
    third (wall-clock for SLA, tool-call count for spam guards) had
    to subclass or fork.

    The multi-dimensional controller flips that around: build a list
    of :class:`BudgetDimension` instances, pass them in, and the
    controller stops the loop the moment ANY one of them reports
    exceeded. The active dimension's name lands in the
    ``loop.budget_exceeded`` event so admin UIs can render
    "stopped because: tokens" etc.

    When no dimension trips, the controller delegates to standard
    signal-driven loop logic (matches ``StandardLoopController``):
    pending tool results → continue, ``complete`` signal → complete,
    ``blocked`` → escalate, no pending tool calls → complete.

    An empty dimension list is allowed — the controller behaves like
    ``StandardLoopController`` in that case.
    """

    def __init__(self, dimensions: Optional[List[BudgetDimension]] = None):
        self._dimensions: List[BudgetDimension] = list(dimensions or [])
        self._last_exceeded: Optional[str] = None
        # Merged view of everything configure() has accepted so far.
        # Kept so (a) get_config() round-trips the manifest input and
        # (b) limits that arrive in a *later* configure() call (the
        # restore order is slot-swap-with-strategy_configs first, then
        # stage update_config forwarding max_turns) still apply to
        # dimensions built earlier.
        self._configured: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "multi_dim_budget"

    @property
    def description(self) -> str:
        if not self._dimensions:
            return "Multi-dim budget (no dimensions registered)"
        return "Multi-dim budget: " + ", ".join(d.description for d in self._dimensions)

    @property
    def dimensions(self) -> List[BudgetDimension]:
        """Defensive copy of the registered dimensions in declared order."""
        return list(self._dimensions)

    @property
    def last_exceeded_dimension(self) -> Optional[str]:
        """Name of the most recently exceeded dimension, or ``None``.

        Useful for downstream observability (events, audit logs)
        without re-walking the dimension list.
        """
        return self._last_exceeded

    def add(self, dimension: BudgetDimension) -> "MultiDimensionalBudgetController":
        """Append a dimension and return self for fluent composition."""
        self._dimensions.append(dimension)
        return self

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="multi_dim_budget",
            fields=[
                ConfigField(
                    name="dimensions",
                    type="array",
                    item_type="string",
                    label="Budget dimensions",
                    description=(
                        "Dimension names to enforce, first exceeded stops the loop. "
                        f"Accepted: {', '.join(sorted(_DIMENSION_ALIASES))}."
                    ),
                    default=[],
                    required=True,
                ),
                ConfigField(
                    name="max_turns",
                    type="integer",
                    label="Max turns",
                    description=(
                        "Cap for the iteration dimension. 0 = defer to state.max_iterations."
                    ),
                    default=0,
                    min_value=0,
                ),
                ConfigField(
                    name="max_cost_usd",
                    type="number",
                    label="Max cost (USD)",
                    description="Cap for the cost dimension. Blank = defer to state.cost_budget_usd.",
                    min_value=0,
                ),
                ConfigField(
                    name="cost_threshold_ratio",
                    type="number",
                    label="Cost threshold ratio",
                    description="Stop at this fraction of the cost cap.",
                    default=0.9,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ConfigField(
                    name="max_tokens",
                    type="integer",
                    label="Max tokens",
                    description="Cap for the tokens dimension. Blank = defer to context window budget.",
                    min_value=1,
                ),
                ConfigField(
                    name="token_threshold_ratio",
                    type="number",
                    label="Token threshold ratio",
                    description="Stop at this fraction of the token cap.",
                    default=0.85,
                    min_value=0.0,
                    max_value=1.0,
                ),
                ConfigField(
                    name="max_seconds",
                    type="number",
                    label="Max wall-clock seconds",
                    description="Required when the wall_clock dimension is declared.",
                    min_value=0,
                ),
                ConfigField(
                    name="max_tool_calls",
                    type="integer",
                    label="Max tool calls",
                    description="Required when the tool_calls dimension is declared.",
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        """Materialize budget dimensions from a manifest ``strategy_configs`` entry.

        2026-06-09 environment-philosophy audit §2.1: Geny prod declares
        ``{"dimensions": ["iterations"]}`` for this controller and the base
        no-op ``configure`` discarded it, leaving an empty dimension list.
        The manifest comment "adding dimensions is a strategy_configs edit —
        no code change" was false until this method existed.

        Limits may arrive separately from the dimension list: manifest
        restore swaps the slot with ``strategy_configs`` first, then the
        stage's ``update_config`` forwards the stage-level ``max_turns``
        through a second ``configure`` call. Both orders work because the
        merged config is kept and limit-only calls update live dimensions
        in place.
        """
        validated: Dict[str, Any] = {}

        if "dimensions" in config:
            names = config["dimensions"]
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError(
                    "multi_dim_budget: 'dimensions' must be a list of dimension "
                    f"names, got {names!r}"
                )
            unknown = [n for n in names if n not in _DIMENSION_ALIASES]
            if unknown:
                raise ValueError(
                    f"multi_dim_budget: unknown dimension(s) {unknown}. "
                    f"Accepted: {sorted(_DIMENSION_ALIASES)}"
                )
            validated["dimensions"] = [str(n) for n in names]

        if "max_turns" in config:
            validated["max_turns"] = _require_int(
                "multi_dim_budget", "max_turns", config["max_turns"], minimum=0
            )
        if "max_iterations" in config:
            validated["max_iterations"] = _require_int(
                "multi_dim_budget", "max_iterations", config["max_iterations"], minimum=1
            )
        if "max_cost_usd" in config:
            validated["max_cost_usd"] = _require_number(
                "multi_dim_budget", "max_cost_usd", config["max_cost_usd"]
            )
        if "max_tokens" in config:
            validated["max_tokens"] = _require_int(
                "multi_dim_budget", "max_tokens", config["max_tokens"], minimum=1
            )
        if "max_seconds" in config:
            validated["max_seconds"] = _require_number(
                "multi_dim_budget", "max_seconds", config["max_seconds"]
            )
        if "max_tool_calls" in config:
            validated["max_tool_calls"] = _require_int(
                "multi_dim_budget", "max_tool_calls", config["max_tool_calls"], minimum=1
            )
        for ratio_key in ("cost_threshold_ratio", "token_threshold_ratio"):
            if ratio_key in config:
                v = _require_number("multi_dim_budget", ratio_key, config[ratio_key])
                if v > 1.0:
                    raise ValueError(
                        f"multi_dim_budget: {ratio_key!r} must be in [0.0, 1.0], got {v!r}"
                    )
                validated[ratio_key] = v

        merged = {**self._configured, **validated}
        if "dimensions" in validated:
            self._dimensions = [self._build_dimension(n, merged) for n in validated["dimensions"]]
        elif validated:
            self._apply_limits(validated)
        self._configured = merged

    def get_config(self) -> Dict[str, Any]:
        cfg = dict(self._configured)
        if "dimensions" not in cfg and self._dimensions:
            # Programmatically-built controller: report live dimension
            # names so snapshots capture *something* restorable.
            cfg["dimensions"] = [d.name for d in self._dimensions]
        return cfg

    @staticmethod
    def _build_dimension(name: str, cfg: Dict[str, Any]) -> BudgetDimension:
        canonical = _DIMENSION_ALIASES[name]
        if canonical == "iteration":
            cap = cfg.get("max_turns") or cfg.get("max_iterations") or None
            return IterationBudget(cap)
        if canonical == "cost":
            return CostBudget(
                cfg.get("max_cost_usd"),
                threshold_ratio=cfg.get("cost_threshold_ratio", 0.9),
            )
        if canonical == "tokens":
            return TokenBudget(
                max_tokens=cfg.get("max_tokens"),
                threshold_ratio=cfg.get("token_threshold_ratio", 0.85),
            )
        if canonical == "wall_clock":
            if "max_seconds" not in cfg:
                raise ValueError("multi_dim_budget: dimension 'wall_clock' requires 'max_seconds'")
            return WallClockBudget(cfg["max_seconds"])
        # canonical == "tool_calls" — the alias table is exhaustive.
        if "max_tool_calls" not in cfg:
            raise ValueError("multi_dim_budget: dimension 'tool_calls' requires 'max_tool_calls'")
        return ToolCallBudget(cfg["max_tool_calls"])

    def _apply_limits(self, validated: Dict[str, Any]) -> None:
        """Push limit-only configure() calls into already-built dimensions.

        Same-module private access is deliberate: the dimension classes and
        this controller ship together, and adding public setters for what
        is an internal replay mechanism would widen the API for no caller.
        """
        for dim in self._dimensions:
            if isinstance(dim, IterationBudget):
                if "max_turns" in validated or "max_iterations" in validated:
                    cap = validated.get("max_turns") or validated.get("max_iterations") or None
                    dim._max = max(1, int(cap)) if cap else None
            elif isinstance(dim, CostBudget):
                if "max_cost_usd" in validated:
                    dim._max_usd = float(validated["max_cost_usd"])
                if "cost_threshold_ratio" in validated:
                    dim._ratio = float(validated["cost_threshold_ratio"])
            elif isinstance(dim, TokenBudget):
                if "max_tokens" in validated:
                    dim._max_tokens = int(validated["max_tokens"])
                if "token_threshold_ratio" in validated:
                    dim._ratio = float(validated["token_threshold_ratio"])
            elif isinstance(dim, WallClockBudget):
                if "max_seconds" in validated:
                    dim._max_seconds = float(validated["max_seconds"])
            elif isinstance(dim, ToolCallBudget):
                if "max_tool_calls" in validated:
                    dim._max = max(1, int(validated["max_tool_calls"]))

    def decide(self, state: PipelineState) -> str:
        # Walk dimensions in declared order; first exceeded wins.
        for dim in self._dimensions:
            try:
                if dim.is_exceeded(state):
                    self._last_exceeded = dim.name
                    logger.info(
                        "MultiDimensionalBudgetController: %s exhausted — stopping loop",
                        dim.name,
                    )
                    return LoopDecision.COMPLETE
            except Exception:
                # A broken dimension must not crash the loop —
                # log + skip; if all dimensions are broken the
                # controller falls through to the default signal
                # logic, which is the safe fallback.
                logger.warning(
                    "MultiDimensionalBudgetController: dim %r raised; skipping",
                    getattr(dim, "name", "?"),
                    exc_info=True,
                )
        self._last_exceeded = None

        if state.tool_results:
            return LoopDecision.CONTINUE

        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE
        if signal == "error":
            return LoopDecision.ERROR

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE
