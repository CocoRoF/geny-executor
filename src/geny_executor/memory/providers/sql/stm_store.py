"""STM plane backed by SQLite.

`stm_turns` is append-only at the API surface: every `append()` adds a
row, `recent(n)` reads the last N by `id`, `truncate(keep_last=N)`
deletes everything older than the N most recent ids.

2.53.0 — session-scoped stores: a store constructed with a non-empty
``session_id`` stamps it on every appended row and FILTERS every read/
trim to that session, so one database (one schema) can host many
sessions' turns side by side. The per-session summary lives in
``stm_summaries`` keyed by session id; a store without a session id
keeps the legacy whole-table view + singleton ``stm_summary`` row, so
existing single-session-per-database deployments are unchanged.

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

    def __init__(
        self, conn: _SQLConnection, *, tz: tzinfo, session_id: str = ""
    ) -> None:
        self._conn = conn
        self._tz = tz
        self._session_id = str(session_id or "")

    def _scoped(self, base_where: str) -> tuple[str, tuple]:
        """세션 스코프 WHERE 절 — session_id 미설정이면 레거시(전체) 뷰."""
        if not self._session_id:
            return base_where, ()
        return f"{base_where} AND session_id = ?", (self._session_id,)

    # ── STMHandle contract ──────────────────────────────────────────

    async def append(self, turn: Turn) -> None:
        content_kind, payload = _encode_content(turn.content)
        ts = _normalise_ts(turn.timestamp, self._tz)
        meta = json.dumps(turn.metadata, ensure_ascii=False) if turn.metadata else None
        await self._conn.execute(
            """
            INSERT INTO stm_turns (
                type, role, content_kind, content, ts, metadata_json,
                event_id, linked_event_id, kind, direction,
                counterpart_id, counterpart_role, session_id
            )
            VALUES ('message', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.role,
                content_kind,
                payload,
                ts,
                meta,
                turn.event_id,
                turn.linked_event_id,
                turn.kind,
                turn.direction,
                turn.counterpart_id,
                turn.counterpart_role,
                turn.session_id or self._session_id or None,
            ),
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
            INSERT INTO stm_turns (type, role, content_kind, content, ts, metadata_json, session_id)
            VALUES ('event', ?, ?, ?, ?, ?, ?)
            """,
            (str(name), kind, payload, ts_str, meta, self._session_id or None),
        )

    async def recent(self, n: int = 20) -> List[Turn]:
        if n <= 0:
            return []
        where, scope_params = self._scoped("type = 'message'")
        rows = await self._conn.fetchall(
            f"SELECT * FROM stm_turns WHERE {where} ORDER BY id DESC LIMIT ?",
            (*scope_params, n),
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
        where, scope_params = self._scoped("type = 'message' AND LOWER(content) LIKE ?")
        rows = await self._conn.fetchall(
            f"""
            SELECT * FROM stm_turns
            WHERE {where}
            ORDER BY id DESC LIMIT ?
            """,
            (f"%{needle.lower()}%", *scope_params, limit),
        )
        return [_row_to_turn(r) for r in rows]

    async def truncate(self, *, keep_last: int) -> int:
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        where, scope_params = self._scoped("1 = 1")
        row = await self._conn.fetchone(
            f"SELECT COUNT(*) AS n FROM stm_turns WHERE {where}", scope_params
        )
        total = int(row["n"]) if row else 0
        if total <= keep_last:
            return 0
        if keep_last == 0:
            await self._conn.execute(
                f"DELETE FROM stm_turns WHERE {where}", scope_params
            )
            return total
        cutoff = await self._conn.fetchone(
            f"SELECT id FROM stm_turns WHERE {where} ORDER BY id DESC LIMIT 1 OFFSET ?",
            (*scope_params, keep_last - 1),
        )
        if cutoff is None:
            return 0
        await self._conn.execute(
            f"DELETE FROM stm_turns WHERE id < ? AND {where}",
            (cutoff["id"], *scope_params),
        )
        return total - keep_last

    # ── Session summary ─────────────────────────────────────────────

    async def read_summary(self) -> Optional[str]:
        if self._session_id:
            row = await self._conn.fetchone(
                "SELECT body FROM stm_summaries WHERE session_id = ?",
                (self._session_id,),
            )
        else:
            row = await self._conn.fetchone("SELECT body FROM stm_summary WHERE id = 1")
        if row is None:
            return None
        body = row.get("body") if isinstance(row, dict) else row["body"]
        return body or None

    async def write_summary(self, body: str) -> None:
        # UPSERT pattern (works across sqlite >= 3.24 + Postgres).
        if self._session_id:
            await self._conn.execute(
                (
                    "INSERT INTO stm_summaries (session_id, body, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT (session_id) DO UPDATE SET "
                    "body = excluded.body, updated_at = excluded.updated_at"
                ),
                (self._session_id, body, now_in(self._tz).isoformat()),
            )
            return
        await self._conn.execute(
            (
                "INSERT INTO stm_summary (id, body) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET body = excluded.body"
            ),
            (body,),
        )

    # ── snapshot helpers ────────────────────────────────────────────

    async def all_rows(self) -> List[dict]:
        where, scope_params = self._scoped("1 = 1")
        rows = await self._conn.fetchall(
            f"SELECT * FROM stm_turns WHERE {where} ORDER BY id ASC", scope_params
        )
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

    def _interaction(name: str) -> Optional[str]:
        try:
            value = row[name]
        except (KeyError, IndexError):
            return None
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str or None

    return Turn(
        role=str(row["role"]),
        content=_decode_content(str(row["content_kind"]), str(row["content"])),
        timestamp=stamp,
        metadata=metadata,
        event_id=_interaction("event_id"),
        linked_event_id=_interaction("linked_event_id"),
        kind=_interaction("kind"),
        direction=_interaction("direction"),
        counterpart_id=_interaction("counterpart_id"),
        counterpart_role=_interaction("counterpart_role"),
        session_id=_interaction("session_id"),
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
