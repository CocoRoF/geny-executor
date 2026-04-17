"""Cost analysis — per-session, per-model, and trend analysis."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

from geny_executor.history.models import CostSummary, CostTrendPoint, ModelCostBreakdown
from geny_executor.history.service import HistoryService


class CostAnalyzer:
    """Execution cost analysis."""

    # USD per 1M tokens
    PRICING: ClassVar[Dict[str, Dict[str, float]]] = {
        "claude-sonnet-4-20250514": {
            "input": 3.0,
            "output": 15.0,
            "cache_read": 0.3,
            "cache_write": 3.75,
        },
        "claude-opus-4-20250514": {
            "input": 15.0,
            "output": 75.0,
            "cache_read": 1.5,
            "cache_write": 18.75,
        },
        "claude-haiku-35-20250620": {
            "input": 0.80,
            "output": 4.0,
            "cache_read": 0.08,
            "cache_write": 1.0,
        },
    }

    def __init__(self, history: HistoryService) -> None:
        self._history = history

    def get_session_cost_summary(self, session_id: str) -> CostSummary:
        """Cost summary per model for a session."""
        rows = self._history._conn.execute(
            "SELECT"
            " model,"
            " COUNT(*) as executions,"
            " SUM(cost_usd) as total_cost,"
            " SUM(input_tokens) as total_input,"
            " SUM(output_tokens) as total_output,"
            " SUM(cache_read_tokens) as total_cache_read,"
            " SUM(cache_write_tokens) as total_cache_write,"
            " SUM(thinking_tokens) as total_thinking,"
            " SUM(tool_calls) as total_tools,"
            " AVG(cost_usd) as avg_cost"
            " FROM executions"
            " WHERE session_id = ?"
            " GROUP BY model",
            (session_id,),
        ).fetchall()

        return CostSummary(
            session_id=session_id,
            by_model=[
                ModelCostBreakdown(
                    model=r["model"],
                    executions=r["executions"],
                    total_cost=r["total_cost"] or 0.0,
                    total_input_tokens=r["total_input"] or 0,
                    total_output_tokens=r["total_output"] or 0,
                    total_cache_read=r["total_cache_read"] or 0,
                    total_cache_write=r["total_cache_write"] or 0,
                    total_thinking=r["total_thinking"] or 0,
                    total_tool_calls=r["total_tools"] or 0,
                    avg_cost_per_execution=r["avg_cost"] or 0.0,
                )
                for r in rows
            ],
            total_cost=sum(r["total_cost"] or 0 for r in rows),
            total_executions=sum(r["executions"] for r in rows),
        )

    def get_cost_trend(
        self,
        session_id: Optional[str] = None,
        granularity: str = "hour",
        limit: int = 168,
    ) -> List[CostTrendPoint]:
        """Cost trend over time."""
        fmt_map = {
            "hour": "%Y-%m-%dT%H:00:00",
            "day": "%Y-%m-%d",
            "week": "%Y-W%W",
        }
        fmt = fmt_map.get(granularity, "%Y-%m-%dT%H:00:00")

        where = "WHERE session_id = ?" if session_id else ""
        params: List[Any] = [session_id] if session_id else []

        rows = self._history._conn.execute(
            f"SELECT"  # noqa: S608
            f" strftime('{fmt}', started_at) as period,"
            f" COUNT(*) as executions,"
            f" SUM(cost_usd) as cost,"
            f" SUM(total_tokens) as tokens"
            f" FROM executions"
            f" {where}"
            f" GROUP BY period"
            f" ORDER BY period DESC"
            f" LIMIT ?",
            (*params, limit),
        ).fetchall()

        return [
            CostTrendPoint(
                period=r["period"],
                executions=r["executions"],
                cost=r["cost"] or 0.0,
                tokens=r["tokens"] or 0,
            )
            for r in reversed(list(rows))
        ]

    @classmethod
    def estimate_cost(
        cls,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Estimate cost for a given usage."""
        pricing = cls.PRICING.get(model)
        if not pricing:
            return 0.0
        cost = (
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
            + cache_read_tokens * pricing.get("cache_read", 0)
            + cache_write_tokens * pricing.get("cache_write", 0)
        ) / 1_000_000
        return round(cost, 6)
