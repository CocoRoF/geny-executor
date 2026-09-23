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
import logging
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.provider import MemoryHooks, RecordReceipt, Turn
from geny_executor.memory.providers.file.timezone import now_in


logger = logging.getLogger(__name__)

MAX_STM_LINES = 2000
#: Byte budget for the whole jsonl. The line cap alone let a session grow to
#: 270 MB in production — 2,000 lines is no bound when single event lines
#: carry hundreds of KB (inlined observation frames, giant tool results).
#: recent()/search() and the transcripts UI re-read the WHOLE file, so the
#: byte budget is what actually keeps those O(file) paths sane.
MAX_STM_BYTES = 16 * 1024 * 1024
#: A single record longer than this is truncated at append time (its
#: ``content`` tail replaced with a marker). Applies to turns AND events —
#: nothing conversational needs half a megabyte of inline payload.
MAX_RECORD_BYTES = 64 * 1024
#: Enforce the line cap every N appends (not every append — the cap
#: re-reads the whole jsonl). Bounds the file to MAX_STM_LINES + this.
_CAP_CHECK_EVERY = 200


def _image_marker(block: Dict[str, Any]) -> Dict[str, Any]:
    raw = block.get("source")
    source: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    media = str(source.get("media_type") or "image")
    size_kb = len(str(source.get("data") or "")) * 3 // 4 // 1024
    return {"type": "text", "text": f"[image: {media}, ~{size_kb} KB, not kept in the transcript]"}


def _drop_images(blocks: List[Any]) -> List[Any]:
    """Images (top level and inside tool results) → a marker naming them."""
    out: List[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        if block.get("type") == "image":
            out.append(_image_marker(block))
            continue
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            block = {**block, "content": _drop_images(block["content"])}
        out.append(block)
    return out


def _clip_block_texts(blocks: List[Any], keep: int) -> List[Any]:
    """Every text / tool-result string over *keep* characters keeps its head."""

    def clip(text: str) -> str:
        if len(text) <= keep:
            return text
        return text[:keep] + f"\n… [{len(text) - keep} chars truncated at record cap]"

    out: List[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        block = dict(block)
        if isinstance(block.get("text"), str):
            block["text"] = clip(block["text"])
        content = block.get("content")
        if isinstance(content, str):
            block["content"] = clip(content)
        elif isinstance(content, list):
            block["content"] = _clip_block_texts(content, keep)
        out.append(block)
    return out


def _fits(rec: Dict[str, Any]) -> Optional[str]:
    line = json.dumps(rec, ensure_ascii=False)
    return line if len(line.encode("utf-8", "ignore")) <= MAX_RECORD_BYTES else None


def _bound_record_line(line: str) -> str:
    """Bring an oversized serialized record under MAX_RECORD_BYTES — and keep
    it a record.

    The result must stay valid JSON. The previous fallback cut the raw line
    in half, which is not a smaller record but an unreadable one: ``recent``
    skips it, so the whole row — the user's words that came with a
    screenshot included — vanished from the conversation. Production had 588
    such rows in one session, every one of them an image row, because only
    string content had a cut and block-list content fell through to the
    hard cut.

    Order: images go first (they are the bytes, and the words beside them are
    the meaning); then the longest texts keep their heads; and only if the
    record is still over — a pathological metadata payload — does it become
    an envelope that says what was dropped.
    """
    if len(line.encode("utf-8", "ignore")) <= MAX_RECORD_BYTES:
        return line
    try:
        rec = json.loads(line)
    except Exception:  # noqa: BLE001 — not JSON to begin with
        return json.dumps(
            {"type": "event", "event": "stm.unreadable_record", "data": {"bytes": len(line)}}
        )
    if not isinstance(rec, dict):
        return json.dumps({"type": "event", "event": "stm.unreadable_record"})

    content = rec.get("content")
    if isinstance(content, str) and len(content) > 4096:
        overshoot = len(line.encode("utf-8", "ignore")) - MAX_RECORD_BYTES
        keep = max(4096, len(content) - overshoot - 256)
        dropped = len(content) - keep
        rec["content"] = content[:keep] + f"\n… [{dropped} chars truncated at record cap]"
        fitted = _fits(rec)
        if fitted is not None:
            return fitted
    elif isinstance(content, list):
        rec["content"] = _drop_images(content)
        fitted = _fits(rec)
        if fitted is not None:
            return fitted
        base = rec["content"]
        keep = MAX_RECORD_BYTES // 4
        while keep >= 256:
            rec["content"] = _clip_block_texts(base, keep)
            fitted = _fits(rec)
            if fitted is not None:
                return fitted
            keep //= 2

    rec.pop("metadata", None)
    fitted = _fits(rec)
    if fitted is not None:
        return fitted
    envelope = {
        k: rec[k]
        for k in ("type", "role", "ts", "event_id", "kind", "direction", "counterpart_id")
        if k in rec
    }
    envelope["content"] = f"[record over {MAX_RECORD_BYTES} bytes, not kept]"
    return json.dumps(envelope, ensure_ascii=False)


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
        #: (stat-signature, parsed lines) — see _read_lines_sync.
        self._lines_cache: Optional[tuple] = None
        # Appends since the last line-cap enforcement (audit M2): the cap
        # runs periodically rather than every append (which would re-read
        # the whole jsonl each time), bounding the file to
        # MAX_STM_LINES + _CAP_CHECK_EVERY without per-append O(n) cost.
        self._appends_since_cap = 0

    # ── NotesHandle contract ────────────────────────────────────────

    async def append(self, turn: Turn) -> None:
        rec = _turn_to_record(turn, self._tz)
        line = _bound_record_line(json.dumps(rec, ensure_ascii=False))
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
        # Periodically bound the file so recent()/search() (which read the
        # whole jsonl) can't grow unboundedly over a long session (M2).
        self._appends_since_cap += 1
        if self._appends_since_cap >= _CAP_CHECK_EVERY:
            self._appends_since_cap = 0
            try:
                await self.enforce_line_cap()
                await self.enforce_byte_cap()
            except Exception:  # noqa: BLE001 — capping is best-effort
                logger.debug("stm: enforce_line_cap failed", exc_info=True)

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
        if len(line.encode("utf-8", "ignore")) > MAX_RECORD_BYTES:
            # Event payloads (observation frames, giant tool results) were the
            # production 270 MB transcript's fat lines. Keep the envelope,
            # drop the payload with an explicit marker.
            rec["data"] = {
                "truncated": True,
                "reason": f"event payload over {MAX_RECORD_BYTES} bytes",
            }
            rec.pop("metadata", None)
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

    async def enforce_byte_cap(self) -> int:
        """Drop OLDEST lines until the file fits MAX_STM_BYTES. Returns
        dropped-line count. Line cap alone is no bound when individual lines
        are hundreds of KB — this is the bound that keeps the whole-file
        readers (recent/search/transcripts UI) usable."""
        async with self._lock:
            try:
                if not self._path.exists() or self._path.stat().st_size <= MAX_STM_BYTES:
                    return 0
            except OSError:
                return 0
            lines = self._read_lines_sync()
            sizes = [len(ln.encode("utf-8", "ignore")) + 1 for ln in lines]
            total = sum(sizes)
            drop = 0
            while drop < len(lines) and total > MAX_STM_BYTES:
                total -= sizes[drop]
                drop += 1
            if drop == 0:
                return 0
            tail = lines[drop:]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for line in tail:
                    fh.write(line + "\n")
            tmp.replace(self._path)
            return drop

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

    # ── Session summary ─────────────────────────────────────────────

    async def read_summary(self) -> Optional[str]:
        """Read ``transcripts/summary.md`` if it exists. Returns
        ``None`` when no summary has been written yet (e.g. brand-new
        session, or session-close hasn't fired).
        """
        summary_path = self._path.parent / "summary.md"
        async with self._lock:
            if not summary_path.exists():
                return None
            try:
                text = summary_path.read_text(encoding="utf-8")
            except OSError:
                return None
        return text or None

    async def write_summary(self, body: str) -> None:
        """Atomically write ``transcripts/summary.md``.

        Called by Stage 19 Summarizer at session close (D1) — a single
        once-per-session write, NOT a per-turn append. Safe to call
        repeatedly; the file is overwritten in full each time.
        """
        summary_path = self._path.parent / "summary.md"
        async with self._lock:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(body)
            tmp.replace(summary_path)

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
        """Whole-file line read with an (mtime_ns, size)-keyed cache.

        recent()/search() — and host UIs layered on them — re-read the entire
        jsonl on every call; on a byte-capped 16 MB transcript that is 16 MB
        of IO+strip per page view. The file only changes through our own
        append/truncate (plus rare external edits, which move mtime/size and
        invalidate naturally), so caching the parsed line list against the
        stat signature makes repeat reads O(1) without any staleness window."""
        try:
            st = self._path.stat()
        except OSError:
            self._lines_cache = None
            return []
        key = (st.st_mtime_ns, st.st_size)
        cached = self._lines_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        with self._path.open("r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh if line.strip()]
        self._lines_cache = (key, lines)
        return lines


# ── record <-> Turn converters ───────────────────────────────────────


_TURN_INTERACTION_FIELDS: tuple = (
    "event_id",
    "linked_event_id",
    "kind",
    "direction",
    "counterpart_id",
    "counterpart_role",
    "session_id",
)


def _turn_to_record(turn: Turn, tz: tzinfo) -> Dict[str, Any]:
    """Serialize a `Turn` into a JSONL record matching Geny's schema.

    `timestamp` is normalised into the provider's configured timezone
    and emitted as ISO-8601. Geny's reader expects `ts`, not
    `timestamp`, so we write `ts`.

    Interaction fields (``event_id`` / ``linked_event_id`` / ``kind`` /
    ``direction`` / ``counterpart_id`` / ``counterpart_role`` /
    ``session_id``) — when present — land at the row's top level so
    downstream readers (web mirror, Geny CLI, dashboards) can index
    them without parsing ``metadata``.
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
    for fname in _TURN_INTERACTION_FIELDS:
        value = getattr(turn, fname, None)
        if value is None:
            continue
        rec[fname] = str(value)
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

    def _interaction(name: str) -> Optional[str]:
        value = rec.get(name)
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str or None

    return Turn(
        role=str(rec.get("role", "user")),
        content=rec.get("content", ""),
        timestamp=stamp,
        metadata=dict(rec.get("metadata", {}) or {}),
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
