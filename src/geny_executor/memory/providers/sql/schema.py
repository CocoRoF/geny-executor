"""DDL for `SQLMemoryProvider` — dialect-aware.

Two dialects ship today: SQLite (default) and Postgres. The table
shape is identical across both — only the column types, the
auto-increment syntax, and the BLOB type change. A single source of
truth (`_TABLE_SPECS`) drives both DDL tuples so they cannot drift.

Each statement is idempotent (`IF NOT EXISTS`) so `initialize()` is
safe to call repeatedly.

Table layout mirrors the file provider's planes:

  - `stm_turns`        ↔ `transcripts/session.jsonl`
  - `ltm_documents`    ↔ `memory/MEMORY.md` + dated/topic files
  - `notes`            ↔ `memory/{category}/{filename}.md`
  - `note_tags`        — normalised tags for `notes`
  - `note_links`       — wikilink + explicit edges
  - `vector_rows`      ↔ `vectordb/index.bin` + `metadata.json`
  - `provider_meta`    — single-row provider state (schema version, …)
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

SCHEMA_VERSION = "1"


class Dialect(str, Enum):
    """SQL dialects supported by `SQLMemoryProvider`.

    The string values are stable identifiers safe to use in
    descriptors, config files, and log messages.
    """

    SQLITE = "sqlite"
    POSTGRES = "postgres"


# ── per-dialect type maps ──────────────────────────────────────────


_TYPE_MAP = {
    Dialect.SQLITE: {
        "auto_pk": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "blob": "BLOB",
    },
    Dialect.POSTGRES: {
        # SERIAL keeps parity with stdlib sqlite3 — a synthetic
        # auto-incrementing integer PK that callers do not supply.
        "auto_pk": "SERIAL PRIMARY KEY",
        "blob": "BYTEA",
    },
}


def _ddl(dialect: Dialect) -> Tuple[str, ...]:
    auto_pk = _TYPE_MAP[dialect]["auto_pk"]
    blob = _TYPE_MAP[dialect]["blob"]
    return (
        f"""
        CREATE TABLE IF NOT EXISTS stm_turns (
            id            {auto_pk},
            type          TEXT    NOT NULL DEFAULT 'message',
            role          TEXT    NOT NULL,
            content_kind  TEXT    NOT NULL,
            content       TEXT    NOT NULL,
            ts            TEXT    NOT NULL,
            metadata_json TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_stm_turns_id ON stm_turns(id)",
        f"""
        CREATE TABLE IF NOT EXISTS ltm_documents (
            id         {auto_pk},
            kind       TEXT    NOT NULL,
            ref_name   TEXT    NOT NULL,
            body       TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            updated_at TEXT    NOT NULL,
            UNIQUE(kind, ref_name)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ltm_kind ON ltm_documents(kind)",
        """
        CREATE TABLE IF NOT EXISTS notes (
            filename         TEXT PRIMARY KEY,
            title            TEXT NOT NULL,
            body             TEXT NOT NULL,
            importance       TEXT NOT NULL,
            category         TEXT,
            scope            TEXT NOT NULL,
            backend          TEXT,
            frontmatter_json TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)",
        "CREATE INDEX IF NOT EXISTS idx_notes_importance ON notes(importance)",
        """
        CREATE TABLE IF NOT EXISTS note_tags (
            filename TEXT NOT NULL,
            tag      TEXT NOT NULL,
            PRIMARY KEY(filename, tag),
            FOREIGN KEY(filename) REFERENCES notes(filename) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag)",
        """
        CREATE TABLE IF NOT EXISTS note_links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            origin TEXT NOT NULL,
            PRIMARY KEY(source, target, origin),
            FOREIGN KEY(source) REFERENCES notes(filename) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target)",
        f"""
        CREATE TABLE IF NOT EXISTS vector_rows (
            filename     TEXT PRIMARY KEY,
            scope        TEXT NOT NULL,
            category     TEXT,
            backend      TEXT,
            preview      TEXT,
            dimension    INTEGER NOT NULL,
            vector_blob  {blob} NOT NULL,
            created_at   TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS provider_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        # Singleton row holding the session summary (D1: stage 19
        # writes once at session close). `id` is locked at 1 so the
        # UPSERT in `_SQLSTMStore.write_summary` always touches the
        # same row.
        """
        CREATE TABLE IF NOT EXISTS stm_summary (
            id   INTEGER PRIMARY KEY,
            body TEXT NOT NULL
        )
        """,
    )


def build_ddl(dialect: Dialect) -> Tuple[str, ...]:
    """Return the DDL tuple for `dialect`. Each statement is
    idempotent and safe to run on a freshly-opened connection.
    """
    return _ddl(dialect)


# ── public, pre-rendered tuples ────────────────────────────────────


SQLITE_DDL: Tuple[str, ...] = build_ddl(Dialect.SQLITE)
POSTGRES_DDL: Tuple[str, ...] = build_ddl(Dialect.POSTGRES)


# All tables we create — used by snapshot/restore to dump and reload.
# Identical across dialects.
TABLES: Tuple[str, ...] = (
    "stm_turns",
    "stm_summary",
    "ltm_documents",
    "notes",
    "note_tags",
    "note_links",
    "vector_rows",
    "provider_meta",
)


# Backwards-compat alias kept for existing call sites.
SQLITE_TABLES: Tuple[str, ...] = TABLES


__all__ = [
    "SCHEMA_VERSION",
    "Dialect",
    "build_ddl",
    "SQLITE_DDL",
    "POSTGRES_DDL",
    "TABLES",
    "SQLITE_TABLES",
]
