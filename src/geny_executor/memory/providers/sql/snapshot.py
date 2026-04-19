"""Snapshot helpers for SQLMemoryProvider.

The snapshot payload is a JSON-encoded dump of every owned table,
plus a SHA-256 checksum over the canonical bytes. This is portable
across SQLite filenames and forms the basis of the C6 backup test.

Restore is destructive on the *target* connection: the existing
tables are truncated and replaced row-for-row.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, Iterable, List, Tuple

from geny_executor.memory.providers.sql.connection import _SQLConnection
from geny_executor.memory.providers.sql.schema import SCHEMA_VERSION, TABLES


SNAPSHOT_FORMAT_VERSION = "1"


async def build_snapshot(conn: _SQLConnection) -> Tuple[bytes, str]:
    """Dump every owned table into a JSON document and return
    `(payload_bytes, sha256_hex)`. Binary BLOB columns are base64-encoded.
    """
    payload: Dict[str, Any] = {
        "format": SNAPSHOT_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "tables": {},
    }
    for table in TABLES:
        rows = await conn.fetchall(f"SELECT * FROM {table}")
        encoded_rows: List[Dict[str, Any]] = []
        for row in rows:
            encoded_rows.append({k: _encode_value(row[k]) for k in row.keys()})
        payload["tables"][table] = encoded_rows
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    checksum = hashlib.sha256(raw).hexdigest()
    return raw, checksum


async def restore_snapshot(
    conn: _SQLConnection,
    payload: bytes,
    checksum: str,
) -> None:
    """Verify checksum, then replace every owned table's contents."""
    actual = hashlib.sha256(payload).hexdigest()
    if checksum and actual != checksum:
        raise ValueError(f"snapshot checksum mismatch: expected {checksum!r}, got {actual!r}")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("snapshot payload is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("snapshot payload must be a JSON object")
    tables = document.get("tables") or {}
    if not isinstance(tables, dict):
        raise ValueError("snapshot payload missing `tables` mapping")

    # Wipe everything first so partial table sets don't leave stale rows.
    await conn.truncate_all(TABLES)

    for table in TABLES:
        rows = tables.get(table) or []
        if not rows:
            continue
        await _restore_table(conn, table, rows)


async def _restore_table(
    conn: _SQLConnection,
    table: str,
    rows: Iterable[Dict[str, Any]],
) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    sql = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"
    seq_params = [tuple(_decode_value(row.get(col)) for col in columns) for row in rows]
    await conn.executemany(sql, seq_params)


# ── value codecs ─────────────────────────────────────────────────────


def _encode_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(bytes(value)).decode("ascii")}
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and "__b64__" in value:
        return base64.b64decode(value["__b64__"])
    return value


__all__ = [
    "SNAPSHOT_FORMAT_VERSION",
    "build_snapshot",
    "restore_snapshot",
]
