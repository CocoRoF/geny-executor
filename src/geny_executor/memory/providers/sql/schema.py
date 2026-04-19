"""DDL for `SQLMemoryProvider`.

One module owns every table the provider creates so the schema is
trivial to audit and the test suite can lock the column list. Each
statement is idempotent (`IF NOT EXISTS`) so `initialize()` is safe
to call repeatedly.

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

from typing import Tuple

SCHEMA_VERSION = "1"


# Each statement is independent and idempotent — execute in order on
# a fresh connection during `initialize()`.
SQLITE_DDL: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS stm_turns (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        type          TEXT    NOT NULL DEFAULT 'message',
        role          TEXT    NOT NULL,
        content_kind  TEXT    NOT NULL,    -- 'string' | 'json'
        content       TEXT    NOT NULL,    -- raw string or JSON-encoded
        ts            TEXT    NOT NULL,    -- ISO-8601
        metadata_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_stm_turns_id ON stm_turns(id)",
    """
    CREATE TABLE IF NOT EXISTS ltm_documents (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT    NOT NULL,        -- 'main' | 'dated' | 'topic'
        ref_name   TEXT    NOT NULL,        -- 'MEMORY.md' | 'YYYY-MM-DD.md' | 'topics/<slug>.md'
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
        origin TEXT NOT NULL,                -- 'wikilink' | 'explicit'
        PRIMARY KEY(source, target, origin),
        FOREIGN KEY(source) REFERENCES notes(filename) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target)",
    """
    CREATE TABLE IF NOT EXISTS vector_rows (
        filename     TEXT PRIMARY KEY,
        scope        TEXT NOT NULL,
        category     TEXT,
        backend      TEXT,
        preview      TEXT,
        dimension    INTEGER NOT NULL,
        vector_blob  BLOB NOT NULL,           -- packed little-endian float32
        created_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)


# All tables we create — used by snapshot/restore to dump and reload.
SQLITE_TABLES: Tuple[str, ...] = (
    "stm_turns",
    "ltm_documents",
    "notes",
    "note_tags",
    "note_links",
    "vector_rows",
    "provider_meta",
)


__all__ = ["SCHEMA_VERSION", "SQLITE_DDL", "SQLITE_TABLES"]
