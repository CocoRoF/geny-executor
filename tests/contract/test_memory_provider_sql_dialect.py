"""Dialect-routing tests for `SQLMemoryProvider`.

The Postgres dialect ships in v0.20.0 as a wired-but-untested
backend — the user-facing contract is "config DSN routes to the
correct backend", not "Postgres queries return identical results to
SQLite". A real Postgres CI matrix is tracked for a follow-up phase,
so these tests pin down only the routing logic and the SQL
translator that lets the dialect-agnostic stores talk to psycopg.

What is tested here:
  - `detect_dialect` picks Postgres for `postgresql://` /
    `postgres://` / SQLAlchemy-style `postgresql+driver://` and
    SQLite for everything else (filesystem path, `sqlite://`).
  - `_translate_to_postgres` rewrites `?` → `%s` and
    `INSERT OR IGNORE` → `INSERT ... ON CONFLICT DO NOTHING`,
    leaving SQLite-and-Postgres common-subset SQL untouched.
  - `_PostgresConnection` is constructible without `psycopg`
    installed and only fails at `open()` time with a clear
    install-extra hint.
  - `MemoryProviderFactory` honours an explicit `dialect` override
    in the config, and the SQL provider exposes a `dialect`
    property that downstream consumers (web, CLI) can introspect.
"""

from __future__ import annotations

import pytest

from geny_executor.memory.factory import MemoryProviderFactory
from geny_executor.memory.providers import SQLMemoryProvider
from geny_executor.memory.providers.sql.connection import (
    _PostgresConnection,
    _SQLiteConnection,
    _translate_to_postgres,
    detect_dialect,
    open_connection,
)
from geny_executor.memory.providers.sql.schema import (
    Dialect,
    POSTGRES_DDL,
    SQLITE_DDL,
    TABLES,
)


# ── DSN scheme detection ────────────────────────────────────────────


@pytest.mark.parametrize(
    "dsn,expected",
    [
        ("postgresql://user:pw@host:5432/db", Dialect.POSTGRES),
        ("postgres://user@host/db", Dialect.POSTGRES),
        ("postgresql+psycopg://user@host/db", Dialect.POSTGRES),
        ("POSTGRESQL://upper@host/db", Dialect.POSTGRES),
        ("/var/lib/session.db", Dialect.SQLITE),
        ("sqlite:///tmp/session.db", Dialect.SQLITE),
        ("file:memory?mode=memory", Dialect.SQLITE),
        ("./session.db", Dialect.SQLITE),
    ],
)
def test_detect_dialect_picks_from_scheme(dsn: str, expected: Dialect):
    assert detect_dialect(dsn) is expected


def test_open_connection_returns_sqlite_for_filesystem_path(tmp_path):
    conn = open_connection(tmp_path / "x.db")
    assert isinstance(conn, _SQLiteConnection)
    assert conn.DIALECT is Dialect.SQLITE


def test_open_connection_returns_postgres_for_postgres_dsn():
    # Construction must not require psycopg — only `.open()` does.
    conn = open_connection("postgresql://localhost/db")
    assert isinstance(conn, _PostgresConnection)
    assert conn.DIALECT is Dialect.POSTGRES


def test_open_connection_explicit_dialect_overrides_scheme(tmp_path):
    # SQLite-shaped DSN forced into Postgres branch — useful for
    # integration tests that point at a Postgres instance via a
    # bare DSN string.
    conn = open_connection(tmp_path / "x.db", dialect=Dialect.POSTGRES)
    assert isinstance(conn, _PostgresConnection)


# ── SQL translation: ? → %s, INSERT OR IGNORE → ON CONFLICT ─────────


def test_translate_placeholders():
    sql = "SELECT * FROM notes WHERE filename = ? AND scope = ?"
    assert _translate_to_postgres(sql) == ("SELECT * FROM notes WHERE filename = %s AND scope = %s")


def test_translate_insert_or_ignore():
    sql = "INSERT OR IGNORE INTO note_tags (filename, tag) VALUES (?, ?)"
    out = _translate_to_postgres(sql)
    assert out.startswith("INSERT INTO note_tags")
    assert out.endswith("ON CONFLICT DO NOTHING")
    assert "%s" in out and "?" not in out


def test_translate_insert_or_ignore_case_insensitive():
    sql = "insert or ignore into note_links (source, target, origin) values (?, ?, 'wikilink')"
    out = _translate_to_postgres(sql)
    assert out.lower().startswith("insert into note_links")
    assert out.endswith("ON CONFLICT DO NOTHING")
    # A `?` inside a single-quoted literal would be preserved, but
    # 'wikilink' has none. Still, the question-mark substitution
    # should hit only the two real placeholders.
    assert out.count("%s") == 2


def test_translate_preserves_question_mark_inside_string_literal():
    sql = "SELECT * FROM t WHERE col = 'who?' AND other = ?"
    out = _translate_to_postgres(sql)
    # Literal '?' stays, the real placeholder becomes %s.
    assert "'who?'" in out
    assert out.count("%s") == 1


def test_translate_leaves_common_subset_untouched():
    # ON CONFLICT(col) DO UPDATE SET ... excluded.col is portable
    # between SQLite (>=3.24) and Postgres, and must not be rewritten.
    sql = (
        "INSERT INTO vector_rows (filename) VALUES (?) "
        "ON CONFLICT(filename) DO UPDATE SET filename = excluded.filename"
    )
    out = _translate_to_postgres(sql)
    assert "ON CONFLICT(filename)" in out
    assert "excluded.filename" in out
    assert "%s" in out


# ── Postgres connection requires psycopg only at open() ─────────────


def test_postgres_connection_construction_does_not_require_psycopg():
    # Should not raise even if psycopg is not importable. The error
    # surfaces only at open() time with a clear install-extra hint.
    conn = _PostgresConnection("postgresql://localhost/x")
    assert conn.dsn == "postgresql://localhost/x"


# ── DDL parity: same table set, Postgres-flavoured types ────────────


def test_postgres_ddl_uses_postgres_types():
    joined = "\n".join(POSTGRES_DDL)
    assert "SERIAL PRIMARY KEY" in joined
    assert "BYTEA NOT NULL" in joined
    # Should not carry SQLite-only syntax.
    assert "AUTOINCREMENT" not in joined
    assert "BLOB " not in joined and "BLOB\n" not in joined


def test_sqlite_ddl_uses_sqlite_types():
    joined = "\n".join(SQLITE_DDL)
    assert "INTEGER PRIMARY KEY AUTOINCREMENT" in joined
    assert "BLOB NOT NULL" in joined
    assert "SERIAL" not in joined
    assert "BYTEA" not in joined


def test_table_set_is_dialect_agnostic():
    assert "stm_turns" in TABLES
    assert "ltm_documents" in TABLES
    assert "notes" in TABLES
    assert "note_tags" in TABLES
    assert "note_links" in TABLES
    assert "vector_rows" in TABLES
    assert "provider_meta" in TABLES


# ── Factory wiring ──────────────────────────────────────────────────


def test_factory_routes_postgres_dsn_to_postgres_dialect():
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "sql",
            "dsn": "postgresql://user@localhost/db",
        }
    )
    assert isinstance(provider, SQLMemoryProvider)
    assert provider.dialect is Dialect.POSTGRES
    # Backend label in the descriptor reflects the chosen dialect.
    backends = {b.layer: b for b in provider.descriptor.backends}
    assert all(b.backend == "postgres" for b in backends.values())
    assert provider.descriptor.metadata["dialect"] == "postgres"


def test_factory_routes_filesystem_dsn_to_sqlite_dialect(tmp_path):
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "sql",
            "dsn": str(tmp_path / "session.db"),
        }
    )
    assert isinstance(provider, SQLMemoryProvider)
    assert provider.dialect is Dialect.SQLITE
    assert provider.descriptor.metadata["dialect"] == "sqlite"


def test_factory_explicit_dialect_overrides_scheme(tmp_path):
    factory = MemoryProviderFactory()
    provider = factory.build(
        {
            "provider": "sql",
            "dsn": str(tmp_path / "session.db"),
            "dialect": "postgres",
        }
    )
    assert provider.dialect is Dialect.POSTGRES


def test_factory_rejects_bogus_dialect_string(tmp_path):
    factory = MemoryProviderFactory()
    with pytest.raises(ValueError):
        factory.build(
            {
                "provider": "sql",
                "dsn": str(tmp_path / "x.db"),
                "dialect": "mysql",
            }
        )
