"""Default artifact strategies for Stage 12: Evaluate."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Type

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_evaluate.interface import EvaluationStrategy, QualityScorer
from geny_executor.stages.s14_evaluate.types import EvaluationResult, QualityCriterion

logger = logging.getLogger(__name__)


class SignalBasedEvaluation(EvaluationStrategy):
    """Evaluate based on completion signals from Stage 9 (Parse)."""

    @property
    def name(self) -> str:
        return "signal_based"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        signal = state.completion_signal

        if signal is None or signal == "continue":
            if state.pending_tool_calls:
                return EvaluationResult(
                    passed=True, decision="continue", feedback="Tool calls pending."
                )
            return EvaluationResult(
                passed=True, decision="continue", feedback="No completion signal detected."
            )

        if signal == "complete":
            return EvaluationResult(
                passed=True,
                score=1.0,
                decision="complete",
                feedback=state.completion_detail or "Task completed.",
            )

        if signal == "blocked":
            return EvaluationResult(
                passed=False,
                score=0.0,
                decision="escalate",
                feedback=state.completion_detail or "Task blocked.",
            )

        if signal == "error":
            return EvaluationResult(
                passed=False,
                score=0.0,
                decision="error",
                feedback=state.completion_detail or "Error encountered.",
            )

        if signal == "delegate":
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback=f"Delegated to: {state.completion_detail or 'unknown'}",
            )

        return EvaluationResult(
            passed=True, decision="continue", feedback=f"Unknown signal: {signal}"
        )


class CriteriaBasedEvaluation(EvaluationStrategy):
    """Evaluate against predefined quality criteria."""

    def __init__(
        self,
        criteria: Optional[List[QualityCriterion]] = None,
        pass_threshold: float = 0.6,
    ):
        self._criteria = criteria or []
        self._pass_threshold = pass_threshold

    @property
    def name(self) -> str:
        return "criteria_based"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="criteria_based",
            fields=[
                ConfigField(
                    name="pass_threshold",
                    type="number",
                    label="Pass threshold",
                    description=(
                        "Weighted-average score required to pass (0.0–1.0). "
                        "Criteria themselves carry Python check callables and are "
                        "host-injected at construction time — they cannot be "
                        "expressed in a manifest."
                    ),
                    default=0.6,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply ``{pass_threshold}`` from a manifest ``strategy_configs`` entry.

        ``criteria`` is intentionally NOT configurable here: each
        :class:`QualityCriterion` carries a ``check`` callable, and a JSON
        manifest cannot round-trip Python callables. Hosts that need custom
        criteria construct the strategy directly.
        """
        if "pass_threshold" in config:
            v = config["pass_threshold"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"criteria_based: 'pass_threshold' must be a number in [0.0, 1.0], got {v!r}"
                )
            self._pass_threshold = float(v)

    def get_config(self) -> Dict[str, Any]:
        return {"pass_threshold": self._pass_threshold}

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        if not self._criteria:
            return EvaluationResult(passed=True, decision="complete")

        criteria_results: List[Dict[str, Any]] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion in self._criteria:
            score = 0.0
            if criterion.check is not None:
                try:
                    score = float(criterion.check(state))
                except Exception:
                    score = 0.0

            passed = score >= criterion.threshold
            criteria_results.append(
                {
                    "name": criterion.name,
                    "score": score,
                    "weight": criterion.weight,
                    "passed": passed,
                }
            )
            weighted_sum += score * criterion.weight
            total_weight += criterion.weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_passed = overall_score >= self._pass_threshold

        return EvaluationResult(
            passed=overall_passed,
            score=overall_score,
            decision="complete" if overall_passed else "retry",
            feedback=f"Score: {overall_score:.2f} (threshold: {self._pass_threshold})",
            criteria_results=criteria_results,
        )


class AgentEvaluation(EvaluationStrategy):
    """Use evaluator agent results from Stage 11."""

    @property
    def name(self) -> str:
        return "agent_evaluation"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        eval_input = state.metadata.get("evaluation_input")
        if not eval_input:
            return EvaluationResult(passed=True, decision="complete")

        if not eval_input.get("evaluator_success", False):
            return EvaluationResult(
                passed=True,
                decision="complete",
                feedback="Evaluator failed; accepting response as-is.",
                metadata=eval_input,
            )

        evaluator_text = eval_input.get("evaluator_response", "")
        score = self._extract_score(evaluator_text)

        return EvaluationResult(
            passed=score is None or score >= 0.6,
            score=score,
            decision="complete" if (score is None or score >= 0.6) else "retry",
            feedback=evaluator_text[:500],
            metadata=eval_input,
        )

    def _extract_score(self, text: str) -> Optional[float]:
        match = re.search(r"[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/\s*100", text)
        if match:
            return float(match.group(1)) / 100.0
        match = re.search(r"[Ss]core[:\s]+(\d+(?:\.\d+)?)", text)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
        return None


class NoScorer(QualityScorer):
    """No scoring — always returns 1.0."""

    @property
    def name(self) -> str:
        return "no_scorer"

    def score(self, state: PipelineState) -> float:
        return 1.0


class WeightedScorer(QualityScorer):
    """Weighted average scorer using configurable metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or {}

    @property
    def name(self) -> str:
        return "weighted"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="weighted",
            fields=[
                ConfigField(
                    name="weights",
                    type="object",
                    label="Metric weights",
                    description=(
                        "Mapping of state.metadata key → weight. Score is the "
                        "weighted average of the metadata values present."
                    ),
                    default={},
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, dict):
                raise ValueError(
                    f"weighted: 'weights' must be an object of metric → number, "
                    f"got {type(weights).__name__}"
                )
            parsed: Dict[str, float] = {}
            for key, value in weights.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"weighted: weight for {key!r} must be a number, got {value!r}"
                    )
                parsed[str(key)] = float(value)
            self._weights = parsed

    def get_config(self) -> Dict[str, Any]:
        return {"weights": dict(self._weights)}

    def score(self, state: PipelineState) -> float:
        if not self._weights:
            return 1.0
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self._weights.items():
            val = state.metadata.get(key)
            if val is not None:
                weighted_sum += float(val) * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 1.0


class EvaluationChain(EvaluationStrategy):
    """Run a sequence of evaluators, first definitive verdict wins.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.6 (Evaluator
    chain). The pre-S7.6 Stage 12 strategy slot held exactly one
    evaluator. Composing layered policy ("trust signals first; fall
    back to criteria; finally call an evaluator agent") meant writing
    a custom strategy that dispatched manually.

    The chain inverts that: hand it a list of evaluators. They run in
    order; the first one returning ``decision != "continue"`` wins.
    A trailing ``continue`` (everyone passed without a definitive
    verdict) returns the last evaluator's result so the chain
    behaves like "no one objected" — same semantics as
    ``GuardChain`` for Stage 4.

    Failure isolation: an evaluator that raises is logged at WARNING
    and skipped. The chain marches on rather than blowing up the
    whole evaluation pass — matches the fail-open semantics the rest
    of Stage 12 already follows.

    Use cases:
        * **Layered policy**: ``[SignalBasedEvaluation(),
          CriteriaBasedEvaluation(...), AgentEvaluation()]`` — cheap
          checks first, expensive evaluator-LLM only when nothing
          earlier said anything definitive.
        * **Multiple criteria packs** (``CriteriaBasedEvaluation``
          for safety, then for quality, then for length) without
          having to merge their criteria lists.
        * **A/B-style fallback** where a primary evaluator can
          decline ("continue") and a backup runs.

    Note: the chain itself is an :class:`EvaluationStrategy`, so
    nested chains are supported.
    """

    def __init__(self, evaluators: Optional[List[EvaluationStrategy]] = None):
        self._evaluators: List[EvaluationStrategy] = list(evaluators or [])
        # Turn-limit knobs accepted on behalf of wrapped evaluators
        # (binary_classify consumes them). Stored so get_config() can
        # round-trip exactly what configure() accepted.
        self._forwarded_config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "evaluation_chain"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="evaluation_chain",
            fields=[
                ConfigField(
                    name="evaluators",
                    type="array",
                    item_type="string",
                    label="Evaluators",
                    description=(
                        "Ordered list of registered evaluator names "
                        f"({', '.join(sorted(EVALUATOR_REGISTRY))}). Resolved to "
                        "instances at configure time; first definitive verdict wins."
                    ),
                    default=[],
                    required=True,
                ),
                ConfigField(
                    name="easy_max_turns",
                    type="integer",
                    label="Easy max turns",
                    description=(
                        "Forwarded to wrapped evaluators that understand it "
                        "(binary_classify): turn cap once a task is classified easy."
                    ),
                    min_value=1,
                ),
                ConfigField(
                    name="not_easy_max_turns",
                    type="integer",
                    label="Not-easy max turns",
                    description=(
                        "Forwarded to wrapped evaluators that understand it "
                        "(binary_classify): turn cap for multi-turn tasks."
                    ),
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        """Resolve ``{evaluators, easy_max_turns, not_easy_max_turns}`` into a live chain.

        2026-06-09 environment-philosophy audit §2.1: Geny prod declared
        ``strategy_configs={"strategy": {"evaluators": ["binary_classify",
        "signal_based"], ...}}`` and the base ``Strategy.configure`` no-op
        silently dropped it. The empty chain then returned
        ``decision="complete"`` every turn, killing the worker loop after a
        single iteration regardless of [CONTINUE] signals. This method is
        the fix: names are resolved to instances HERE, at configure time,
        and bad input fails loudly instead of degrading into an empty chain.

        Turn-limit keys are stored and forwarded to every resolved
        evaluator's ``configure`` — ``binary_classify`` consumes them,
        evaluators without the knob inherit the base no-op.
        """
        forwarded: Dict[str, Any] = {}
        for key in ("easy_max_turns", "not_easy_max_turns"):
            if key in config:
                v = config[key]
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise ValueError(
                        f"evaluation_chain: {key!r} must be an integer >= 1, got {v!r}"
                    )
                forwarded[key] = v

        if "evaluators" in config:
            names = config["evaluators"]
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError(
                    "evaluation_chain: 'evaluators' must be a list of registered "
                    f"evaluator names, got {names!r}"
                )
            unknown = [n for n in names if n not in EVALUATOR_REGISTRY]
            if unknown:
                raise ValueError(
                    f"evaluation_chain: unknown evaluator(s) {unknown}. "
                    f"Available: {sorted(EVALUATOR_REGISTRY)}"
                )
            self._evaluators = [EVALUATOR_REGISTRY[n]() for n in names]

        self._forwarded_config.update(forwarded)
        if self._forwarded_config:
            for evaluator in self._evaluators:
                evaluator.configure(dict(self._forwarded_config))

    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self._forwarded_config)
        if self._evaluators:
            cfg["evaluators"] = [getattr(ev, "name", "?") for ev in self._evaluators]
        return cfg

    @property
    def description(self) -> str:
        names = [getattr(ev, "name", "?") for ev in self._evaluators]
        return f"Sequential chain: {', '.join(names) if names else '(empty)'}"

    @property
    def evaluators(self) -> List[EvaluationStrategy]:
        """Read-only view of the wrapped evaluators in declared order."""
        return list(self._evaluators)

    def add(self, evaluator: EvaluationStrategy) -> "EvaluationChain":
        """Append an evaluator and return self for chaining."""
        self._evaluators.append(evaluator)
        return self

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        if not self._evaluators:
            # Empty chain → no verdict. Match the existing
            # CriteriaBasedEvaluation no-criteria default.
            return EvaluationResult(
                passed=True,
                decision="complete",
                feedback="empty evaluator chain",
            )

        last_result = EvaluationResult(
            passed=True,
            decision="continue",
            feedback="no evaluator produced a definitive decision",
        )
        for evaluator in self._evaluators:
            try:
                result = await evaluator.evaluate(state)
            except Exception as exc:
                logger.warning(
                    "EvaluationChain: evaluator %r raised; skipping (%s)",
                    getattr(evaluator, "name", "?"),
                    exc,
                    exc_info=True,
                )
                continue

            if result.decision != "continue":
                # First definitive verdict wins — short-circuit the rest.
                return result
            last_result = result
        return last_result


def _build_evaluator_registry() -> Dict[str, Type[EvaluationStrategy]]:
    """Single source for spellable evaluator names.

    Mirrors (and is consumed by) the default stage's ``strategy`` slot
    registry so that ``EvaluationChain.configure({"evaluators": [...]})``
    accepts exactly the names a manifest can already select as the slot
    strategy — no second naming scheme to drift. ``binary_classify`` is
    imported lazily-at-module-bottom because the adaptive artifact module
    only depends on interface/types (no cycle), matching what the default
    stage module already does.
    """
    from geny_executor.stages.s14_evaluate.artifact.adaptive.strategy import (
        BinaryClassifyEvaluation,
    )

    return {
        "signal_based": SignalBasedEvaluation,
        "criteria_based": CriteriaBasedEvaluation,
        "agent_evaluation": AgentEvaluation,
        "binary_classify": BinaryClassifyEvaluation,
        "evaluation_chain": EvaluationChain,
    }


#: name → class registry shared by the stage slot and EvaluationChain.configure.
EVALUATOR_REGISTRY: Dict[str, Type[EvaluationStrategy]] = _build_evaluator_registry()
