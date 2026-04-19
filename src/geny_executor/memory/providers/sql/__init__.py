"""SQL-backed `MemoryProvider`.

Default dialect is SQLite via the stdlib `sqlite3` module. The
schema is dialect-agnostic enough that a future Postgres+pgvector
backend can plug in via the same `SQLMemoryProvider` shell — that
work lives in sub-PR 2c.1 / Phase 2d.
"""

from geny_executor.memory.providers.sql.provider import SQLMemoryProvider

__all__ = ["SQLMemoryProvider"]
