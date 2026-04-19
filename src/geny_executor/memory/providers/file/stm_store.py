"""STM plane backed by `transcripts/session.jsonl`.

Geny format — one JSON object per line, with the following shape:

    {"type": "message", "role": "...", "content": "...",
     "ts": "<ISO-8601>", "metadata": {...}}

Truncation: the file is capped at 2000 lines. When `truncate(keep_last=N)`
is called, the file is rewritten to hold the final N turns. This
matches Geny's truncation semantics.

All writes are line-append and fsync-free. Callers that require
durability should invoke `flush()` (a thin wrapper around file system
flush) explicitly — the cross-provider contract doesn't require
fsync-on-write and the ephemeral provider doesn't offer it either.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.memory.provider import Turn
from geny_executor.memory.providers.file.timezone import now_in


MAX_STM_LINES = 2000


class _JSONLSTMStore:
    """Append-only JSONL file backed STM.

    Concurrency: all reads/writes are serialised through an asyncio
    lock. Multiple coroutines in one process are safe; cross-process
    access is not a goal for Phase 2a (Phase 2c SQL provider will
    cover multi-writer scenarios).
    """

    def __init__(self, path: Path, *, tz: tzinfo) -> None:
        self._path = path
        self._tz = tz
        self._lock = asyncio.Lock()

    # ── NotesHandle contract ────────────────────────────────────────

    async def append(self, turn: Turn) -> None:
        rec = _turn_to_record(turn, self._tz)
        line = json.dumps(rec, ensure_ascii=False)
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    async def recent(self, n: int = 20) -> List[Turn]:
        if n <= 0:
            return []
        lines = await self._read_lines()
        return [t for t in (_record_to_turn(line) for line in lines[-n:]) if t is not None]

    async def search(self, text: str, *, limit: int = 10) -> List[Turn]:
        needle = text.lower().strip()
        if not needle or limit <= 0:
            return []
        lines = await self._read_lines()
        out: List[Turn] = []
        for line in reversed(lines):
            turn = _record_to_turn(line)
            if turn is None:
                continue
            haystack = _turn_haystack(turn).lower()
            if needle in haystack:
                out.append(turn)
                if len(out) >= limit:
                    break
        return out

    async def truncate(self, *, keep_last: int) -> int:
        """Rewrite the file to keep only the last `keep_last` lines.
        Returns the number of dropped lines.
        """
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        async with self._lock:
            lines = self._read_lines_sync()
            total = len(lines)
            if total <= keep_last:
                return 0
            tail = lines[-keep_last:] if keep_last else []
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for line in tail:
                    fh.write(line + "\n")
            tmp.replace(self._path)
            return total - keep_last

    # ── Housekeeping ────────────────────────────────────────────────

    async def enforce_line_cap(self) -> int:
        """Bound the file to `MAX_STM_LINES`. Returns dropped count.
        Call after every append if strict Geny-compatibility is needed.
        """
        lines = await self._read_lines()
        if len(lines) <= MAX_STM_LINES:
            return 0
        return await self.truncate(keep_last=MAX_STM_LINES)

    async def all_turns(self) -> List[Turn]:
        return [
            t for t in (_record_to_turn(line) for line in await self._read_lines()) if t is not None
        ]

    # ── internal ────────────────────────────────────────────────────

    async def _read_lines(self) -> List[str]:
        async with self._lock:
            return self._read_lines_sync()

    def _read_lines_sync(self) -> List[str]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]


# ── record <-> Turn converters ───────────────────────────────────────


def _turn_to_record(turn: Turn, tz: tzinfo) -> Dict[str, Any]:
    """Serialize a `Turn` into a JSONL record matching Geny's schema.

    `timestamp` is normalised into the provider's configured timezone
    and emitted as ISO-8601. Geny's reader expects `ts`, not
    `timestamp`, so we write `ts`.
    """
    stamp = turn.timestamp
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=tz)
    else:
        stamp = stamp.astimezone(tz)
    rec: Dict[str, Any] = {
        "type": "message",
        "role": turn.role,
        "content": turn.content,
        "ts": stamp.isoformat(),
    }
    if turn.metadata:
        rec["metadata"] = dict(turn.metadata)
    return rec


def _record_to_turn(raw: str) -> Optional[Turn]:
    try:
        rec = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("type") not in (None, "message"):
        # Skip non-message events (tool_call, state_change, ...).
        # Phase 2a exposes only turns through STMHandle; the raw events
        # remain on disk for the web mirror to render directly.
        return None
    ts_raw = rec.get("ts") or rec.get("timestamp")
    stamp = _parse_ts(ts_raw) or now_in(_utc())
    return Turn(
        role=str(rec.get("role", "user")),
        content=rec.get("content", ""),
        timestamp=stamp,
        metadata=dict(rec.get("metadata", {}) or {}),
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


def _turn_haystack(turn: Turn) -> str:
    if isinstance(turn.content, str):
        return turn.content
    try:
        return json.dumps(turn.content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(turn.content)
