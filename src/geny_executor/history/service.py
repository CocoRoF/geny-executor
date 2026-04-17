"""HistoryService — SQLite-based execution history persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from geny_executor.history.models import StageTimingRecord, ToolCallRecord

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    environment_id  TEXT,
    model           TEXT NOT NULL,
    user_input      TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    result_text     TEXT,
    total_tokens    INTEGER DEFAULT 0,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cache_read_tokens  INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0.0,
    iterations      INTEGER DEFAULT 0,
    tool_calls      INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,
    error_type      TEXT,
    error_message   TEXT,
    error_stage     INTEGER,
    duration_ms     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stage_timings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id    TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    iteration       INTEGER NOT NULL,
    stage_order     INTEGER NOT NULL,
    stage_name      TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT NOT NULL,
    duration_ms     INTEGER NOT NULL,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    was_cached      BOOLEAN DEFAULT FALSE,
    was_skipped     BOOLEAN DEFAULT FALSE,
    tool_name       TEXT,
    tool_success    BOOLEAN,
    tool_duration_ms INTEGER
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id    TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    iteration       INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    input_json      TEXT,
    output_text     TEXT,
    is_error        BOOLEAN DEFAULT FALSE,
    duration_ms     INTEGER DEFAULT 0,
    called_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_tags (
    execution_id    TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    tag             TEXT NOT NULL,
    PRIMARY KEY (execution_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_exec_session ON executions(session_id);
CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_exec_model ON executions(model);
CREATE INDEX IF NOT EXISTS idx_exec_started ON executions(started_at);
CREATE INDEX IF NOT EXISTS idx_timing_exec ON stage_timings(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_exec ON tool_calls(execution_id);
"""

_ALLOWED_ORDERS = frozenset(
    {
        "started_at DESC",
        "started_at ASC",
        "cost_usd DESC",
        "duration_ms DESC",
        "total_tokens DESC",
    }
)


class HistoryService:
    """SQLite-backed execution history persistence."""

    def __init__(
        self,
        db_path: str = "environments/history.db",
        blob_path: str = "environments/blobs",
    ) -> None:
        self._db_path = Path(db_path)
        self._blob_path = Path(blob_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._blob_path.mkdir(parents=True, exist_ok=True)
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Execution lifecycle ──────────────────────────────

    def start_execution(
        self,
        session_id: str,
        model: str,
        user_input: str,
        environment_id: Optional[str] = None,
    ) -> str:
        """Record execution start. Returns execution id."""
        exec_id = str(uuid4())
        self._conn.execute(
            "INSERT INTO executions (id, session_id, environment_id, model, user_input, started_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                exec_id,
                session_id,
                environment_id,
                model,
                user_input,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return exec_id

    def finish_execution(
        self,
        exec_id: str,
        status: str,
        result_text: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record execution completion."""
        now = datetime.now(timezone.utc).isoformat()
        updates: Dict[str, Any] = {"finished_at": now, "status": status}

        if result_text:
            updates["result_text"] = result_text[:2000]
        if usage:
            updates.update(
                {
                    "total_tokens": usage.get("total_tokens", 0),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_tokens", 0),
                    "cache_write_tokens": usage.get("cache_write_tokens", 0),
                    "cost_usd": usage.get("cost_usd", 0.0),
                    "iterations": usage.get("iterations", 0),
                    "tool_calls": usage.get("tool_calls", 0),
                    "thinking_tokens": usage.get("thinking_tokens", 0),
                }
            )
        if error:
            updates.update(
                {
                    "error_type": error.get("type", ""),
                    "error_message": error.get("message", ""),
                    "error_stage": error.get("stage"),
                }
            )

        # duration_ms
        row = self._conn.execute(
            "SELECT started_at FROM executions WHERE id = ?", (exec_id,)
        ).fetchone()
        if row:
            started = datetime.fromisoformat(row["started_at"])
            finished = datetime.fromisoformat(now)
            updates["duration_ms"] = int(
                (finished - started).total_seconds() * 1000
            )

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE executions SET {set_clause} WHERE id = ?",  # noqa: S608
            (*updates.values(), exec_id),
        )
        self._conn.commit()

    # ── Stage & Tool recording ───────────────────────────

    def record_stage_timing(self, exec_id: str, timing: StageTimingRecord) -> None:
        """Record a stage timing entry."""
        self._conn.execute(
            "INSERT INTO stage_timings"
            " (execution_id, iteration, stage_order, stage_name,"
            "  started_at, finished_at, duration_ms,"
            "  input_tokens, output_tokens, was_cached, was_skipped,"
            "  tool_name, tool_success, tool_duration_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exec_id,
                timing.iteration,
                timing.stage_order,
                timing.stage_name,
                timing.started_at,
                timing.finished_at,
                timing.duration_ms,
                timing.input_tokens,
                timing.output_tokens,
                timing.was_cached,
                timing.was_skipped,
                timing.tool_name,
                timing.tool_success,
                timing.tool_duration_ms,
            ),
        )
        self._conn.commit()

    def record_tool_call(self, exec_id: str, tool_call: ToolCallRecord) -> None:
        """Record a tool call entry."""
        self._conn.execute(
            "INSERT INTO tool_calls"
            " (execution_id, iteration, tool_name, input_json, output_text,"
            "  is_error, duration_ms, called_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exec_id,
                tool_call.iteration,
                tool_call.tool_name,
                tool_call.input_json[:10000] if tool_call.input_json else None,
                tool_call.output_text[:5000] if tool_call.output_text else None,
                tool_call.is_error,
                tool_call.duration_ms,
                tool_call.called_at,
            ),
        )
        self._conn.commit()

    def add_tags(self, exec_id: str, tags: List[str]) -> None:
        """Add tags to an execution."""
        for tag in tags:
            self._conn.execute(
                "INSERT OR IGNORE INTO execution_tags (execution_id, tag) VALUES (?, ?)",
                (exec_id, tag),
            )
        self._conn.commit()

    # ── Event stream (JSONL blobs) ───────────────────────

    def save_event_stream(self, exec_id: str, events: List[Dict[str, Any]]) -> None:
        """Save event stream as JSONL file."""
        blob_dir = self._blob_path / exec_id
        blob_dir.mkdir(parents=True, exist_ok=True)
        with open(blob_dir / "events.jsonl", "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def load_event_stream(self, exec_id: str) -> List[Dict[str, Any]]:
        """Load event stream from JSONL file."""
        path = self._blob_path / exec_id / "events.jsonl"
        if not path.exists():
            return []
        events: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    # ── Queries ──────────────────────────────────────────

    def list_executions(
        self,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "started_at DESC",
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List executions with optional filters. Returns (rows, total)."""
        where_clauses: List[str] = []
        params: List[Any] = []

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)
        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if status:
            where_clauses.append("status = ?")
            params.append(status)

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM executions {where}", params  # noqa: S608
        ).fetchone()[0]

        if order_by not in _ALLOWED_ORDERS:
            order_by = "started_at DESC"

        rows = self._conn.execute(
            f"SELECT * FROM executions {where} ORDER BY {order_by} LIMIT ? OFFSET ?",  # noqa: S608
            (*params, limit, offset),
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            # Attach tags
            tag_rows = self._conn.execute(
                "SELECT tag FROM execution_tags WHERE execution_id = ?", (d["id"],)
            ).fetchall()
            d["tags"] = [tr["tag"] for tr in tag_rows]
            result.append(d)

        return result, total

    def get_execution_detail(self, exec_id: str) -> Optional[Dict[str, Any]]:
        """Get full execution detail including timings and tool calls."""
        row = self._conn.execute(
            "SELECT * FROM executions WHERE id = ?", (exec_id,)
        ).fetchone()
        if not row:
            return None

        detail = dict(row)

        # Tags
        tag_rows = self._conn.execute(
            "SELECT tag FROM execution_tags WHERE execution_id = ?", (exec_id,)
        ).fetchall()
        detail["tags"] = [tr["tag"] for tr in tag_rows]

        # Stage timings
        detail["stage_timings"] = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM stage_timings WHERE execution_id = ?"
                " ORDER BY iteration, stage_order",
                (exec_id,),
            ).fetchall()
        ]

        # Tool calls
        detail["tool_call_records"] = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM tool_calls WHERE execution_id = ? ORDER BY called_at",
                (exec_id,),
            ).fetchall()
        ]

        return detail

    def delete_execution(self, exec_id: str) -> bool:
        """Delete an execution and associated data."""
        row = self._conn.execute(
            "SELECT id FROM executions WHERE id = ?", (exec_id,)
        ).fetchone()
        if not row:
            return False
        # CASCADE deletes stage_timings, tool_calls, execution_tags
        self._conn.execute("DELETE FROM executions WHERE id = ?", (exec_id,))
        self._conn.commit()

        # Clean up blob
        blob_dir = self._blob_path / exec_id
        if blob_dir.exists():
            import shutil
            shutil.rmtree(blob_dir, ignore_errors=True)

        return True

    def get_stats(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics."""
        where = "WHERE session_id = ?" if session_id else ""
        params = [session_id] if session_id else []

        row = self._conn.execute(
            f"SELECT"  # noqa: S608
            f" COUNT(*) as total,"
            f" SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed,"
            f" SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors,"
            f" SUM(cost_usd) as total_cost,"
            f" SUM(total_tokens) as total_tokens,"
            f" AVG(duration_ms) as avg_duration_ms"
            f" FROM executions {where}",
            params,
        ).fetchone()

        return {
            "total": row["total"] or 0,
            "completed": row["completed"] or 0,
            "errors": row["errors"] or 0,
            "total_cost": row["total_cost"] or 0.0,
            "total_tokens": row["total_tokens"] or 0,
            "avg_duration_ms": row["avg_duration_ms"] or 0.0,
        }
