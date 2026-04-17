"""Performance monitoring — waterfall charts and stage statistics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.history.models import (
    IterationWaterfall,
    StageStats,
    StageWaterfall,
    WaterfallData,
)
from geny_executor.history.service import HistoryService


class PerformanceMonitor:
    """Execution performance analysis."""

    def __init__(self, history: HistoryService) -> None:
        self._history = history

    def get_waterfall(self, exec_id: str) -> WaterfallData:
        """Generate waterfall chart data for an execution."""
        detail = self._history.get_execution_detail(exec_id)
        if not detail:
            raise ValueError(f"Execution not found: {exec_id}")

        timings = detail.get("stage_timings", [])

        # Group by iteration
        iterations: Dict[int, List[Dict[str, Any]]] = {}
        for t in timings:
            it = t["iteration"]
            if it not in iterations:
                iterations[it] = []
            iterations[it].append(t)

        return WaterfallData(
            execution_id=exec_id,
            total_duration_ms=detail.get("duration_ms", 0),
            iterations=[
                IterationWaterfall(
                    iteration=it,
                    stages=[
                        StageWaterfall(
                            order=s["stage_order"],
                            name=s["stage_name"],
                            duration_ms=s["duration_ms"],
                            was_cached=bool(s.get("was_cached")),
                            was_skipped=bool(s.get("was_skipped")),
                            tokens=(s.get("input_tokens", 0) or 0)
                            + (s.get("output_tokens", 0) or 0),
                        )
                        for s in stages
                    ],
                )
                for it, stages in sorted(iterations.items())
            ],
        )

    def get_stage_stats(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[int, StageStats]:
        """Get aggregate stage performance statistics."""
        where = "WHERE e.session_id = ?" if session_id else ""
        params: List[Any] = [session_id] if session_id else []

        rows = self._history._conn.execute(
            f"SELECT"  # noqa: S608
            f" st.stage_order,"
            f" st.stage_name,"
            f" COUNT(*) as count,"
            f" AVG(st.duration_ms) as avg_ms,"
            f" MIN(st.duration_ms) as min_ms,"
            f" MAX(st.duration_ms) as max_ms,"
            f" SUM(CASE WHEN st.was_cached THEN 1 ELSE 0 END) as cache_hits,"
            f" SUM(CASE WHEN st.was_skipped THEN 1 ELSE 0 END) as skips,"
            f" AVG(st.input_tokens) as avg_input_tokens,"
            f" AVG(st.output_tokens) as avg_output_tokens"
            f" FROM stage_timings st"
            f" JOIN executions e ON e.id = st.execution_id"
            f" {where}"
            f" GROUP BY st.stage_order, st.stage_name"
            f" ORDER BY st.stage_order",
            params,
        ).fetchall()

        return {
            row["stage_order"]: StageStats(
                order=row["stage_order"],
                name=row["stage_name"],
                count=row["count"],
                avg_ms=row["avg_ms"] or 0.0,
                min_ms=row["min_ms"] or 0.0,
                max_ms=row["max_ms"] or 0.0,
                cache_hit_rate=(
                    row["cache_hits"] / row["count"] if row["count"] > 0 else 0.0
                ),
                skip_rate=(
                    row["skips"] / row["count"] if row["count"] > 0 else 0.0
                ),
                avg_input_tokens=row["avg_input_tokens"] or 0.0,
                avg_output_tokens=row["avg_output_tokens"] or 0.0,
            )
            for row in rows
        }

    def get_bottlenecks(
        self, exec_id: str, threshold_pct: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Identify bottleneck stages (> threshold_pct of total duration)."""
        waterfall = self.get_waterfall(exec_id)
        total = waterfall.total_duration_ms
        if total <= 0:
            return []

        bottlenecks: List[Dict[str, Any]] = []
        for it in waterfall.iterations:
            for stage in it.stages:
                pct = stage.duration_ms / total
                if pct >= threshold_pct:
                    bottlenecks.append(
                        {
                            "iteration": it.iteration,
                            "stage_order": stage.order,
                            "stage_name": stage.name,
                            "duration_ms": stage.duration_ms,
                            "percentage": round(pct * 100, 1),
                        }
                    )
        return bottlenecks
