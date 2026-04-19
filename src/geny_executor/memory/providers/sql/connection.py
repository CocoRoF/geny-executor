"""Async-friendly connection wrapper around stdlib `sqlite3`.

`sqlite3` is synchronous. The wrapper serialises every call through a
single asyncio lock and runs the actual statement on the calling
thread; `sqlite3` is fast enough that thread-pool offloading would
add more overhead than it saves for the per-session workloads the
memory subsystem targets.

The dialect surface is intentionally narrow (`execute`, `executemany`,
`fetchone`, `fetchall`, `commit`, `close`). A Postgres + pgvector
adapter can satisfy the same shape via `psycopg` / `asyncpg` in a
follow-up sub-PR without changing the per-store SQL builders — the
SQL is constrained to the SQLite-and-Postgres common subset.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from geny_executor.memory.providers.sql.schema import SQLITE_DDL


DSN = Union[str, Path]


class _SQLiteConnection:
    """Single-writer SQLite handle.

    Owns the underlying `sqlite3.Connection` and the asyncio lock that
    serialises access. All public methods are coroutines so callers
    can `await` without dropping into a sync section.
    """

    def __init__(self, dsn: DSN) -> None:
        self._dsn = str(dsn)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    # ── lifecycle ───────────────────────────────────────────────────

    async def open(self) -> None:
        if self._conn is not None:
            return
        # `check_same_thread=False` so the lock — not the GIL thread
        # ID — is the source of single-writer truth.
        self._conn = sqlite3.connect(self._dsn, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        for stmt in SQLITE_DDL:
            self._conn.execute(stmt)
        self._conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None

    # ── execution ───────────────────────────────────────────────────

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        async with self._lock:
            conn = self._require()
            conn.execute(sql, tuple(params))
            conn.commit()

    async def executemany(self, sql: str, seq_params: Iterable[Sequence[Any]]) -> None:
        async with self._lock:
            conn = self._require()
            conn.executemany(sql, [tuple(p) for p in seq_params])
            conn.commit()

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        async with self._lock:
            conn = self._require()
            cur = conn.execute(sql, tuple(params))
            row = cur.fetchone()
            cur.close()
            return row

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        async with self._lock:
            conn = self._require()
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
            cur.close()
            return list(rows)

    async def execute_returning(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Tuple[Optional[int], int]:
        """Execute `sql`; return (lastrowid, rowcount). Useful for
        INSERT/UPDATE/DELETE that need to know what they touched.
        """
        async with self._lock:
            conn = self._require()
            cur = conn.execute(sql, tuple(params))
            last = cur.lastrowid
            count = cur.rowcount
            cur.close()
            conn.commit()
            return last, count

    async def transaction(self, statements: Iterable[Tuple[str, Sequence[Any]]]) -> None:
        """Run a batch of (sql, params) pairs in one transaction."""
        async with self._lock:
            conn = self._require()
            try:
                for sql, params in statements:
                    conn.execute(sql, tuple(params))
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    # ── snapshot helpers ────────────────────────────────────────────

    async def iterdump_table(self, table: str) -> List[sqlite3.Row]:
        return await self.fetchall(f"SELECT * FROM {table}")

    async def truncate_all(self, tables: Iterable[str]) -> None:
        async with self._lock:
            conn = self._require()
            for t in tables:
                conn.execute(f"DELETE FROM {t}")
            conn.commit()

    # ── internals ───────────────────────────────────────────────────

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("connection is not open; call `await open()` first")
        return self._conn

    @property
    def dsn(self) -> str:
        return self._dsn


__all__ = ["_SQLiteConnection", "DSN"]
