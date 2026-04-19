"""SQL-backed `MemoryProvider`.

Two dialects ship today: SQLite (default, stdlib `sqlite3`) and
Postgres (optional `[postgres]` extra → `psycopg`). Pick by DSN
scheme — ``postgresql://`` / ``postgres://`` routes to the Postgres
backend, anything else (filesystem path or ``sqlite://`` URL) routes
to SQLite. Override via the ``dialect=`` constructor kwarg if needed.
"""

from geny_executor.memory.providers.sql.provider import SQLMemoryProvider
from geny_executor.memory.providers.sql.schema import Dialect

__all__ = ["SQLMemoryProvider", "Dialect"]
