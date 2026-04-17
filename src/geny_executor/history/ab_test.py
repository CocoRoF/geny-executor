"""A/B test runner — compare two environments side-by-side."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.history.models import ABSide, ABTestResult
from geny_executor.history.service import HistoryService


class ABTestRunner:
    """Run two environments with the same input and compare results."""

    def __init__(self, history: HistoryService) -> None:
        self._history = history

    def create_test(
        self,
        env_a_id: str,
        env_b_id: str,
        user_input: str,
        session_id: str = "ab_test",
        model: str = "claude-sonnet-4-20250514",
    ) -> ABTestResult:
        """Create a test structure (execution happens externally).

        This sets up both execution IDs. The actual pipeline execution
        is handled by the backend which has access to API keys.
        """
        exec_a = self._history.start_execution(session_id, model, user_input, env_a_id)
        exec_b = self._history.start_execution(session_id, model, user_input, env_b_id)

        return ABTestResult(
            env_a=ABSide(environment_id=env_a_id, execution_id=exec_a),
            env_b=ABSide(environment_id=env_b_id, execution_id=exec_b),
            user_input=user_input,
        )

    def complete_side(
        self,
        exec_id: str,
        result_text: str,
        usage: Dict[str, Any],
        duration_ms: int,
        iterations: int,
        tool_calls_count: int,
    ) -> None:
        """Record completion of one side of the A/B test."""
        self._history.finish_execution(
            exec_id,
            "completed",
            result_text=result_text,
            usage=usage,
        )

    def get_comparison(self, exec_a_id: str, exec_b_id: str) -> Optional[Dict[str, Any]]:
        """Compare two completed executions."""
        detail_a = self._history.get_execution_detail(exec_a_id)
        detail_b = self._history.get_execution_detail(exec_b_id)

        if not detail_a or not detail_b:
            return None

        return {
            "env_a": {
                "execution_id": exec_a_id,
                "model": detail_a["model"],
                "status": detail_a["status"],
                "result_text": detail_a.get("result_text", ""),
                "cost_usd": detail_a.get("cost_usd", 0),
                "duration_ms": detail_a.get("duration_ms", 0),
                "total_tokens": detail_a.get("total_tokens", 0),
                "iterations": detail_a.get("iterations", 0),
                "tool_calls": detail_a.get("tool_calls", 0),
            },
            "env_b": {
                "execution_id": exec_b_id,
                "model": detail_b["model"],
                "status": detail_b["status"],
                "result_text": detail_b.get("result_text", ""),
                "cost_usd": detail_b.get("cost_usd", 0),
                "duration_ms": detail_b.get("duration_ms", 0),
                "total_tokens": detail_b.get("total_tokens", 0),
                "iterations": detail_b.get("iterations", 0),
                "tool_calls": detail_b.get("tool_calls", 0),
            },
            "diff": {
                "cost_diff": (detail_a.get("cost_usd", 0) or 0)
                - (detail_b.get("cost_usd", 0) or 0),
                "duration_diff": (detail_a.get("duration_ms", 0) or 0)
                - (detail_b.get("duration_ms", 0) or 0),
                "token_diff": (detail_a.get("total_tokens", 0) or 0)
                - (detail_b.get("total_tokens", 0) or 0),
            },
        }
