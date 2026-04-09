"""Evaluate stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """Result of response evaluation."""

    passed: bool = True
    score: Optional[float] = None  # 0.0 - 1.0
    feedback: str = ""
    decision: str = "continue"  # continue | complete | retry | escalate
    criteria_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCriterion:
    """A single quality criterion for criteria-based evaluation."""

    name: str
    description: str
    weight: float = 1.0
    threshold: float = 0.5
    check: Optional[Any] = None  # Callable[[PipelineState], float]
