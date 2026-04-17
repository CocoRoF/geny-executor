"""Tool Scope — stage-level tool access control.

Provides conditional tool visibility per-stage based on iteration count,
cost budget, or custom signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from geny_executor.core.state import PipelineState


@dataclass
class ToolScopeRule:
    """Conditional tool rule evaluated at runtime.

    Attributes:
        tool_name: Name of the tool affected by this rule.
        action: ``"add"`` to include or ``"remove"`` to exclude.
        condition_type: One of ``"iteration"``, ``"cost"``, ``"stage"``, ``"always"``.
        condition_value: Comparison expression or threshold.
    """

    tool_name: str
    action: str  # "add" | "remove"
    condition_type: str  # "iteration" | "cost" | "stage" | "always"
    condition_value: Any = None

    def matches(self, state: PipelineState, stage_order: int = 0) -> bool:
        """Evaluate whether this rule applies to the current state."""
        if self.condition_type == "always":
            return True
        if self.condition_type == "iteration":
            return self._eval_comparison(state.iteration, self.condition_value)
        if self.condition_type == "cost":
            return self._eval_comparison(state.total_cost_usd, self.condition_value)
        if self.condition_type == "stage":
            return stage_order == self.condition_value
        return False

    @staticmethod
    def _eval_comparison(actual: Any, expr: Any) -> bool:
        """Evaluate a simple comparison expression safely.

        Supports: ``">= 3"``, ``"< 10"``, ``"== 5"``, or numeric threshold (>=).
        """
        if isinstance(expr, (int, float)):
            return actual >= expr

        if isinstance(expr, str):
            expr = expr.strip()
            for op in (">=", "<=", "!=", "==", ">", "<"):
                if expr.startswith(op):
                    try:
                        val = float(expr[len(op) :].strip())
                        if op == ">=":
                            return actual >= val
                        elif op == "<=":
                            return actual <= val
                        elif op == ">":
                            return actual > val
                        elif op == "<":
                            return actual < val
                        elif op == "==":
                            return actual == val
                        elif op == "!=":
                            return actual != val
                    except (ValueError, TypeError):
                        return False
        return False


@dataclass
class ToolScope:
    """Tool access scope definition."""

    include: Optional[Set[str]] = None  # None = all tools
    exclude: Optional[Set[str]] = None
    rules: List[ToolScopeRule] = field(default_factory=list)

    def resolve(
        self,
        all_tools: List[str],
        state: PipelineState,
        stage_order: int = 0,
    ) -> Set[str]:
        """Determine which tools are available given the current state."""
        available = set(all_tools)

        if self.include is not None:
            available &= self.include
        if self.exclude is not None:
            available -= self.exclude

        for rule in self.rules:
            if rule.matches(state, stage_order):
                if rule.action == "add":
                    available.add(rule.tool_name)
                elif rule.action == "remove":
                    available.discard(rule.tool_name)

        return available


class ToolScopeManager:
    """Stage-level tool scope management."""

    def __init__(self) -> None:
        self._global_scope: ToolScope = ToolScope()
        self._stage_scopes: Dict[int, ToolScope] = {}

    def set_global_scope(self, scope: ToolScope) -> None:
        self._global_scope = scope

    def get_global_scope(self) -> ToolScope:
        return self._global_scope

    def set_stage_scope(self, stage_order: int, scope: ToolScope) -> None:
        self._stage_scopes[stage_order] = scope

    def get_stage_scope(self, stage_order: int) -> Optional[ToolScope]:
        return self._stage_scopes.get(stage_order)

    def remove_stage_scope(self, stage_order: int) -> bool:
        return self._stage_scopes.pop(stage_order, None) is not None

    def resolve_for_stage(
        self,
        stage_order: int,
        all_tools: List[str],
        state: PipelineState,
    ) -> Set[str]:
        """Resolve available tools for a specific stage.

        Applies global scope first, then stage-specific scope.
        """
        global_tools = self._global_scope.resolve(all_tools, state, stage_order)

        stage_scope = self._stage_scopes.get(stage_order)
        if stage_scope:
            return stage_scope.resolve(list(global_tools), state, stage_order)

        return global_tools

    def describe(self) -> Dict[str, Any]:
        """Return scope configuration for UI."""
        return {
            "global": {
                "include": sorted(self._global_scope.include)
                if self._global_scope.include
                else None,
                "exclude": sorted(self._global_scope.exclude)
                if self._global_scope.exclude
                else None,
                "rules_count": len(self._global_scope.rules),
            },
            "stage_scopes": {
                order: {
                    "include": sorted(scope.include) if scope.include else None,
                    "exclude": sorted(scope.exclude) if scope.exclude else None,
                    "rules_count": len(scope.rules),
                }
                for order, scope in self._stage_scopes.items()
            },
        }
