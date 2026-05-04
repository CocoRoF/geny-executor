"""STM plane backed by SQLite.

`stm_turns` is append-only at the API surface: every `append()` adds a
row, `recent(n)` reads the last N by `id`, `truncate(keep_last=N)`
deletes everything older than the N most recent ids.

`Turn.content` may be a string or a structured Anthropic content
block. We store both — a `content_kind` discriminator records which
form the column holds so reads can re-hydrate without ambiguity.
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from typing import Any, List, Optional

from geny_executor.memory.provider import Turn
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.memory.providers.sql.connection import _SQLConnection


class _SQLSTMStore:
    """`STMHandle`-conformant store on SQL (SQLite or Postgres)."""

    def __init__(self, conn: _SQLConnection, *, tz: tzinfo) -> None:
        self._conn = conn
        self._tz = tz

    # ── STMHandle contract ──────────────────────────────────────────

    async def append(self, turn: Turn) -> None:
        kind, payload = _encode_content(turn.content)
        ts = _normalise_ts(turn.timestamp, self._tz)
        meta = json.dumps(turn.metadata, ensure_ascii=False) if turn.metadata else None
        await self._conn.execute(
            """
            INSERT INTO stm_turns (type, role, content_kind, content, ts, metadata_json)
            VALUES ('message', ?, ?, ?, ?, ?)
            """,
            (turn.role, kind, payload, ts, meta),
        )

    async def append_event(
        self,
        name: str,
        data: Optional[dict] = None,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        """Append an event row alongside message rows.

        Stored with ``type='event'`` and the event payload encoded
        via ``_encode_content`` (kind=text or json) so existing
        readers see a uniform row shape. `recent` / `search` filter
        to ``type='message'`` so events don't leak into message
        views.
        """
        ts = now_in(self._tz)
        ts_str = ts.isoformat()
        kind, payload = _encode_content(dict(data) if data else {})
        meta = json.dumps(metadata, ensure_ascii=False) if metadata else None
        await self._conn.execute(
            """
            INSERT INTO stm_turns (type, role, content_kind, content, ts, metadata_json)
            VALUES ('event', ?, ?, ?, ?, ?)
            """,
            (str(name), kind, payload, ts_str, meta),
        )

    async def recent(self, n: int = 20) -> List[Turn]:
        if n <= 0:
            return []
        rows = await self._conn.fetchall(
            "SELECT * FROM stm_turns WHERE type = 'message' ORDER BY id DESC LIMIT ?",
            (n,),
        )
        # Reverse so callers see chronological order
        return [_row_to_turn(r) for r in reversed(rows)]

    async def search(self, text: str, *, limit: int = 10) -> List[Turn]:
        needle = text.strip()
        if not needle or limit <= 0:
            return []
        # Case-insensitive substring on the raw `content` column. For
        # JSON-encoded structured content this still finds substring
        # matches because the JSON form is searchable. Events are
        # filtered out — protocol scopes search to messages.
        rows = await self._conn.fetchall(
            """
            SELECT * FROM stm_turns
            WHERE type = 'message' AND LOWER(content) LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (f"%{needle.lower()}%", limit),
        )
        return [_row_to_turn(r) for r in rows]

    async def truncate(self, *, keep_last: int) -> int:
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        # Total rows
        row = await self._conn.fetchone("SELECT COUNT(*) AS n FROM stm_turns")
        total = int(row["n"]) if row else 0
        if total <= keep_last:
            return 0
        if keep_last == 0:
            await self._conn.execute("DELETE FROM stm_turns")
            return total
        cutoff = await self._conn.fetchone(
            "SELECT id FROM stm_turns ORDER BY id DESC LIMIT 1 OFFSET ?",
            (keep_last - 1,),
        )
        if cutoff is None:
            return 0
        await self._conn.execute(
            "DELETE FROM stm_turns WHERE id < ?",
            (cutoff["id"],),
        )
        return total - keep_last

    # ── snapshot helpers ────────────────────────────────────────────

    async def all_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM stm_turns ORDER BY id ASC")
        return [dict(r) for r in rows]


# ── helpers ──────────────────────────────────────────────────────────


def _encode_content(content: Any) -> tuple[str, str]:
    if isinstance(content, str):
        return "string", content
    return "json", json.dumps(content, ensure_ascii=False)


def _decode_content(kind: str, raw: str) -> Any:
    if kind == "json":
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw
    return raw


def _normalise_ts(stamp: datetime, tz: tzinfo) -> str:
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=tz)
    return stamp.astimezone(tz).isoformat()


def _row_to_turn(row: Any) -> Turn:
    metadata: dict = {}
    raw_meta = row["metadata_json"]
    if raw_meta:
        try:
            decoded = json.loads(raw_meta)
            if isinstance(decoded, dict):
                metadata = decoded
        except (TypeError, ValueError):
            metadata = {}
    stamp = _parse_ts(row["ts"]) or now_in(_utc())
    return Turn(
        role=str(row["role"]),
        content=_decode_content(str(row["content_kind"]), str(row["content"])),
        timestamp=stamp,
        metadata=metadata,
    )


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _utc() -> tzinfo:
    from datetime import timezone

    return timezone.utc


__all__ = ["_SQLSTMStore"]
