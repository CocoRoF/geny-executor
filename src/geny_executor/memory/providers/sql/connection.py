"""Async-friendly connection wrappers for `SQLMemoryProvider`.

Two dialects ship: SQLite (stdlib `sqlite3`) and Postgres (`psycopg`).
Both surface the same nine-method API so the per-store SQL builders
do not branch on dialect:

  - ``open()`` / ``close()`` — lifecycle
  - ``execute(sql, params)``                 → autocommit single stmt
  - ``executemany(sql, seq_params)``         → autocommit batch
  - ``fetchone(sql, params)``                → single row or ``None``
  - ``fetchall(sql, params)``                → list of rows
  - ``execute_returning(sql, params)``       → ``(lastrowid, rowcount)``
  - ``transaction(statements)``              → atomic batch
  - ``iterdump_table(table)`` / ``truncate_all(tables)``

Rows expose mapping access (``row["col"]``) on both backends so the
stores can walk results without dialect knowledge.

The Postgres wrapper translates the SQLite-flavoured SQL the stores
emit to Postgres syntax on the fly: ``?`` → ``%s`` placeholders and
``INSERT OR IGNORE`` → ``INSERT ... ON CONFLICT DO NOTHING``. The
remainder of the SQL surface area (UPSERT via ``ON CONFLICT (col) DO
UPDATE SET ... excluded.col``, ``LIMIT … OFFSET …``, ``DELETE FROM``)
is part of the SQLite-and-Postgres common subset and needs no
rewriting.

`_PostgresConnection` is wired up but not exercised by the test suite;
the user-facing contract is "config DSN → correct backend chosen",
not "Postgres queries return identical results to SQLite". A real
Postgres CI matrix is tracked for a follow-up phase.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.providers.sql.schema import (
    Dialect,
    POSTGRES_DDL,
    SQLITE_DDL,
)


DSN = Union[str, Path]


# ── shared base ─────────────────────────────────────────────────────


class _SQLConnection:
    """Common shape for the two dialect connections.

    Subclasses implement the storage-specific bits (open, execute,
    cursor handling) but every store talks to this surface.
    """

    DIALECT: Dialect

    async def open(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:  # pragma: no cover
        raise NotImplementedError

    async def executemany(
        self, sql: str, seq_params: Iterable[Sequence[Any]]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    async def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Optional[Mapping[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    async def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> List[Mapping[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    async def execute_returning(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Tuple[Optional[int], int]:  # pragma: no cover
        raise NotImplementedError

    async def transaction(
        self, statements: Iterable[Tuple[str, Sequence[Any]]]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    async def iterdump_table(self, table: str) -> List[Mapping[str, Any]]:
        return await self.fetchall(f"SELECT * FROM {table}")

    async def truncate_all(self, tables: Iterable[str]) -> None:  # pragma: no cover
        raise NotImplementedError


# ── SQLite ──────────────────────────────────────────────────────────


class _SQLiteConnection(_SQLConnection):
    """Single-writer SQLite handle.

    Owns the underlying `sqlite3.Connection` and the asyncio lock that
    serialises access. All public methods are coroutines so callers
    can `await` without dropping into a sync section.
    """

    DIALECT = Dialect.SQLITE

    def __init__(self, dsn: DSN) -> None:
        self._dsn = str(dsn)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = LoopAgnosticLock()

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


# ── Postgres ────────────────────────────────────────────────────────


# Match `INSERT OR IGNORE INTO` (case-insensitive, optional whitespace)
# and rewrite to `INSERT INTO ... ON CONFLICT DO NOTHING`.
_INSERT_OR_IGNORE_RE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)


def _translate_to_postgres(sql: str) -> str:
    """Rewrite SQLite-flavoured SQL to Postgres syntax.

    Two transformations cover the entire SQL surface this provider
    emits today:

    1. ``?`` placeholders  → ``%s`` (psycopg's paramstyle)
    2. ``INSERT OR IGNORE`` → ``INSERT ... ON CONFLICT DO NOTHING`` —
       appended at the end of the statement so it composes with any
       existing trailing whitespace / semicolons.

    The translator preserves ``?`` characters that appear inside
    string literals; the stores never embed ``?`` in literals so the
    cheap split-on-quote pass is sufficient.
    """
    # Step 1 — INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
    if _INSERT_OR_IGNORE_RE.search(sql):
        sql = _INSERT_OR_IGNORE_RE.sub("INSERT INTO", sql)
        # Append the ON CONFLICT clause before any trailing semicolon /
        # whitespace.
        stripped = sql.rstrip().rstrip(";")
        sql = stripped + " ON CONFLICT DO NOTHING"

    # Step 2 — ? → %s, but skip ? inside single-quoted string literals
    out: List[str] = []
    in_str = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            # Handle '' escape inside literals
            if in_str and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            in_str = not in_str
            out.append(ch)
        elif ch == "?" and not in_str:
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


class _PostgresConnection(_SQLConnection):
    """Postgres handle backed by `psycopg` (v3).

    The class is import-safe even when `psycopg` is not installed —
    the dependency is loaded lazily inside ``open()`` so SQLite-only
    deployments do not need the optional extra. The wrapper presents
    the same nine-method surface as `_SQLiteConnection`; SQL is
    translated transparently via :func:`_translate_to_postgres`.

    `psycopg.rows.dict_row` is wired so result rows expose
    ``row["column"]`` access identical to `sqlite3.Row`.
    """

    DIALECT = Dialect.POSTGRES

    def __init__(self, dsn: DSN) -> None:
        self._dsn = str(dsn)
        self._conn: Any = None
        self._lock = LoopAgnosticLock()

    # ── lifecycle ───────────────────────────────────────────────────

    async def open(self) -> None:
        if self._conn is not None:
            return
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Postgres dialect requires the optional 'psycopg' dependency. "
                "Install with `pip install geny-executor[postgres]`."
            ) from exc

        self._conn = await asyncio.to_thread(
            psycopg.connect,
            self._dsn,
            autocommit=False,
            row_factory=dict_row,
        )
        # DDL on a fresh autocommit transaction so partial table
        # creation does not leave the connection in an aborted state.
        await asyncio.to_thread(self._init_schema)

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            for stmt in POSTGRES_DDL:
                cur.execute(stmt)
        self._conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.commit)
                await asyncio.to_thread(self._conn.close)
                self._conn = None

    # ── execution ───────────────────────────────────────────────────

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        translated = _translate_to_postgres(sql)
        async with self._lock:
            await asyncio.to_thread(self._exec_one, translated, tuple(params))

    def _exec_one(self, sql: str, params: Tuple[Any, ...]) -> None:
        conn = self._require()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()

    async def executemany(self, sql: str, seq_params: Iterable[Sequence[Any]]) -> None:
        translated = _translate_to_postgres(sql)
        rows = [tuple(p) for p in seq_params]
        async with self._lock:
            await asyncio.to_thread(self._exec_many, translated, rows)

    def _exec_many(self, sql: str, rows: List[Tuple[Any, ...]]) -> None:
        conn = self._require()
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[Mapping[str, Any]]:
        translated = _translate_to_postgres(sql)
        async with self._lock:
            return await asyncio.to_thread(self._fetch_one, translated, tuple(params))

    def _fetch_one(self, sql: str, params: Tuple[Any, ...]) -> Optional[Mapping[str, Any]]:
        conn = self._require()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[Mapping[str, Any]]:
        translated = _translate_to_postgres(sql)
        async with self._lock:
            return await asyncio.to_thread(self._fetch_all, translated, tuple(params))

    def _fetch_all(self, sql: str, params: Tuple[Any, ...]) -> List[Mapping[str, Any]]:
        conn = self._require()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    async def execute_returning(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Tuple[Optional[int], int]:
        translated = _translate_to_postgres(sql)
        async with self._lock:
            return await asyncio.to_thread(self._exec_returning, translated, tuple(params))

    def _exec_returning(self, sql: str, params: Tuple[Any, ...]) -> Tuple[Optional[int], int]:
        conn = self._require()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            count = cur.rowcount
        conn.commit()
        # Postgres has no portable lastrowid; callers that need it
        # should use RETURNING explicitly. Return (None, rowcount).
        return None, count

    async def transaction(self, statements: Iterable[Tuple[str, Sequence[Any]]]) -> None:
        translated = [(_translate_to_postgres(sql), tuple(p)) for sql, p in statements]
        async with self._lock:
            await asyncio.to_thread(self._tx, translated)

    def _tx(self, statements: List[Tuple[str, Tuple[Any, ...]]]) -> None:
        conn = self._require()
        try:
            with conn.cursor() as cur:
                for sql, params in statements:
                    cur.execute(sql, params)
        except Exception:
            conn.rollback()
            raise
        conn.commit()

    # ── snapshot helpers ────────────────────────────────────────────

    async def truncate_all(self, tables: Iterable[str]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._truncate, list(tables))

    def _truncate(self, tables: List[str]) -> None:
        conn = self._require()
        with conn.cursor() as cur:
            for t in tables:
                cur.execute(f"DELETE FROM {t}")
        conn.commit()

    # ── internals ───────────────────────────────────────────────────

    def _require(self) -> Any:
        if self._conn is None:
            raise RuntimeError("connection is not open; call `await open()` first")
        return self._conn

    @property
    def dsn(self) -> str:
        return self._dsn


# ── DSN-driven dispatch ─────────────────────────────────────────────


_POSTGRES_SCHEMES = ("postgresql://", "postgresql+", "postgres://", "postgres+")


def detect_dialect(dsn: DSN) -> Dialect:
    """Pick a dialect from the DSN scheme.

    - ``postgresql://`` / ``postgres://`` (and SQLAlchemy-style
      ``postgresql+driver://``) → ``Dialect.POSTGRES``
    - everything else (including raw filesystem paths and
      ``sqlite://`` URLs) → ``Dialect.SQLITE``
    """
    s = str(dsn).strip().lower()
    for prefix in _POSTGRES_SCHEMES:
        if s.startswith(prefix):
            return Dialect.POSTGRES
    return Dialect.SQLITE


def open_connection(dsn: DSN, *, dialect: Optional[Dialect] = None) -> _SQLConnection:
    """Build the right `_SQLConnection` for `dsn`.

    Pass ``dialect=`` explicitly to override the DSN-based detection
    (useful for pointing the SQLite backend at a Postgres-shaped DSN
    in a test fixture, or vice-versa).
    """
    chosen = dialect or detect_dialect(dsn)
    if chosen is Dialect.POSTGRES:
        return _PostgresConnection(dsn)
    return _SQLiteConnection(dsn)


__all__ = [
    "_SQLConnection",
    "_SQLiteConnection",
    "_PostgresConnection",
    "DSN",
    "detect_dialect",
    "open_connection",
    "_translate_to_postgres",
]
