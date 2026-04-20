"""BinaryClassifyEvaluation — easy/not_easy adaptive evaluation.

Classifies tasks on the first turn based on the LLM's response pattern:
  - easy: No tool calls, completion signal present → 1-turn finish
  - not_easy: Tool calls or [CONTINUE] signal → multi-turn loop

This mirrors the philosophy of Geny's optimized-autonomous template
which used binary difficulty classification to minimize token usage.

After classification, subsequent turns use signal-based evaluation
(same as SignalBasedEvaluation) until the task is complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_evaluate.interface import EvaluationStrategy
from geny_executor.stages.s12_evaluate.types import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class BinaryClassifyConfig:
    """Configuration for binary task classification.

    Attributes:
        easy_max_turns: Max turns for easy tasks. Once classified as easy,
            state.max_iterations is reduced to this value.
        not_easy_max_turns: Max turns for not_easy tasks. Overrides
            state.max_iterations when classified as not_easy.
    """

    easy_max_turns: int = 1
    not_easy_max_turns: int = 30


class BinaryClassifyEvaluation(EvaluationStrategy):
    """Binary classify + signal-based evaluation.

    First turn:
      Inspects the LLM response to determine task class:
      - easy: no tool calls + (complete signal OR plain text) → finish
      - not_easy: tool calls OR continue signal → loop

    Subsequent turns (not_easy only):
      Uses completion signals from s09_parse:
      - [COMPLETE] → finish
      - [CONTINUE] / tool calls → continue
      - [BLOCKED] → escalate
      - [ERROR] → error

    This keeps easy tasks cheap (1 API call) while allowing complex
    tasks to use the full tool loop.
    """

    def __init__(self, config: Optional[BinaryClassifyConfig] = None):
        self._config = config or BinaryClassifyConfig()

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply ``{easy_max_turns, not_easy_max_turns}`` from a manifest.

        Manifest-restore calls this with the ``strategy_configs`` dict after
        the slot swaps to an instance built via ``cls()``. Unknown keys are
        ignored so the manifest can evolve without breaking older strategies.
        """
        if "easy_max_turns" in config:
            self._config.easy_max_turns = int(config["easy_max_turns"])
        if "not_easy_max_turns" in config:
            self._config.not_easy_max_turns = int(config["not_easy_max_turns"])

    @property
    def name(self) -> str:
        return "binary_classify"

    @property
    def description(self) -> str:
        return "Auto-classifies easy/not_easy on first turn, then signal-based"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        # ── First turn: classify ──
        if state.iteration <= 1 and "task_class" not in state.metadata:
            return self._classify_first_turn(state)

        # ── Subsequent turns: signal-based ──
        return self._evaluate_signal(state)

    def _classify_first_turn(self, state: PipelineState) -> EvaluationResult:
        """Classify on first turn based on response pattern."""
        has_tool_calls = bool(state.pending_tool_calls)
        signal = state.completion_signal

        if has_tool_calls:
            # Tools needed → not_easy
            state.metadata["task_class"] = "not_easy"
            state.max_iterations = self._config.not_easy_max_turns
            logger.info(
                "Binary classify: not_easy (tool calls detected, max_turns=%d)",
                self._config.not_easy_max_turns,
            )
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback="Classified as not_easy: tool calls pending.",
                metadata={"task_class": "not_easy"},
            )

        if signal == "continue":
            # Explicit continue → not_easy
            state.metadata["task_class"] = "not_easy"
            state.max_iterations = self._config.not_easy_max_turns
            logger.info(
                "Binary classify: not_easy (continue signal, max_turns=%d)",
                self._config.not_easy_max_turns,
            )
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback="Classified as not_easy: continue signal.",
                metadata={"task_class": "not_easy"},
            )

        # No tools, no continue → easy (complete immediately)
        state.metadata["task_class"] = "easy"
        state.max_iterations = self._config.easy_max_turns
        logger.info("Binary classify: easy (direct answer, 1 turn)")
        return EvaluationResult(
            passed=True,
            score=1.0,
            decision="complete",
            feedback="Classified as easy: direct answer.",
            metadata={"task_class": "easy"},
        )

    def _evaluate_signal(self, state: PipelineState) -> EvaluationResult:
        """Signal-based evaluation for subsequent turns."""
        signal = state.completion_signal

        # Tool calls always continue
        if state.pending_tool_calls:
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback="Tool calls pending.",
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
                feedback=f"Delegated: {state.completion_detail or 'unknown'}",
            )

        if signal == "continue" or signal is None:
            # No explicit signal but text present and no tools → might be done
            if state.final_text and not state.pending_tool_calls:
                return EvaluationResult(
                    passed=True,
                    score=0.8,
                    decision="complete",
                    feedback="No signal, treating text-only response as complete.",
                )
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback="Continuing...",
            )

        return EvaluationResult(
            passed=True,
            decision="continue",
            feedback=f"Unknown signal: {signal}",
        )
