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

import json
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.provider import MemoryHooks, RecordReceipt, Turn
from geny_executor.memory.providers.file.timezone import now_in


MAX_STM_LINES = 2000


class _JSONLSTMStore:
    """Append-only JSONL file backed STM.

    Concurrency: all reads/writes are serialised through a
    loop-agnostic lock so hosts that drive the store from a sync
    bridge (multiple short-lived event loops) don't trigger
    cross-loop ``Future attached to a different loop`` errors.
    Cross-process access is not a goal here — SQL provider covers
    multi-writer scenarios.
    """

    def __init__(
        self,
        path: Path,
        *,
        tz: tzinfo,
        hooks: Optional[MemoryHooks] = None,
    ) -> None:
        self._path = path
        self._tz = tz
        self._lock = LoopAgnosticLock()
        self._hooks = hooks or MemoryHooks()

    # ── NotesHandle contract ────────────────────────────────────────

    async def append(self, turn: Turn) -> None:
        rec = _turn_to_record(turn, self._tz)
        line = json.dumps(rec, ensure_ascii=False)
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        # Fire after_record_turn hook outside the write lock so a
        # slow business callback can't stall the next append. Default
        # `RecordReceipt()` because STM-only writes don't have notes
        # / vector counts to report.
        await _fire_hook(
            self._hooks.after_record_turn,
            "after_record_turn",
            turn,
            RecordReceipt(),
        )

    async def append_event(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a non-message event line to the STM jsonl.

        Used by hosts that record tool calls / state transitions /
        background-trigger fires inline with the conversation
        transcript. The line shape mirrors Geny's legacy event
        record (``type=event``) so downstream readers (web mirror,
        operator dashboards) keep working unchanged.

        ``recent`` / ``search`` skip event lines — those are
        message-only views per the protocol.
        """
        ts = now_in(self._tz).isoformat()
        rec: Dict[str, Any] = {"type": "event", "event": str(name), "ts": ts}
        if data:
            rec["data"] = dict(data)
        if metadata:
            rec["metadata"] = dict(metadata)
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


async def _fire_hook(callback, name: str, *args) -> None:
    """Run a `MemoryHooks.after_*` callback safely.

    Failures are logged at debug level and swallowed — hooks are
    business logic, never the source of memory-write failure. Hosts
    that need a hook to be load-bearing should raise to a higher
    layer themselves.
    """
    if callback is None:
        return
    try:
        await callback(*args)
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).debug(
            "memory hook %s raised; skipping",
            name,
            exc_info=True,
        )
