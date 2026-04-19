"""LTM plane backed by SQLite.

Three logical surfaces (`main`, `dated`, `topic`) are merged onto the
single `ltm_documents` table with a `kind` discriminator and a
`(kind, ref_name)` UNIQUE index. Append-style writes use
`INSERT … ON CONFLICT(kind, ref_name) DO UPDATE` to fold new content
into the existing body, matching the file provider's semantics:

  - `append(body, heading=…)` → `kind='main'`, `ref_name='MEMORY.md'`
  - `write_dated(body, day=…)` → `kind='dated'`, `ref_name='YYYY-MM-DD.md'`
  - `write_topic(slug, body)` → `kind='topic'`, `ref_name='topics/<slug>.md'`

Each kind keeps the same render formatting as the file provider so
search excerpts read identically across backends.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, tzinfo
from typing import List, Optional, Tuple

from geny_executor.memory.provider import NoteRef, Scope
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.memory.providers.sql.connection import _SQLiteConnection
from geny_executor.stages.s02_context.types import MemoryChunk


_RECENCY_HALFLIFE_DAYS = 30.0
_SLUG_RE = re.compile(r"[^a-zA-Z0-9가-힣\-_]+")


class _SQLLTMStore:
    """`LTMHandle`-conformant store on SQLite."""

    def __init__(
        self,
        conn: _SQLiteConnection,
        *,
        tz: tzinfo,
        scope: Scope = Scope.SESSION,
    ) -> None:
        self._conn = conn
        self._tz = tz
        self._scope = scope

    # ── LTMHandle contract ──────────────────────────────────────────

    async def append(self, body: str, *, heading: Optional[str] = None) -> NoteRef:
        stamp = now_in(self._tz)
        block = _render_evergreen_block(body, stamp=stamp, heading=heading)
        await self._upsert("main", "MEMORY.md", block, stamp)
        return NoteRef(
            filename="MEMORY.md",
            scope=self._scope,
            category="ltm-main",
            backend="sqlite",
        )

    async def write_dated(self, body: str, *, day: Optional[datetime] = None) -> NoteRef:
        when = (day or now_in(self._tz)).astimezone(self._tz)
        date_key = when.date().isoformat()
        block = _render_dated_block(body, stamp=when)
        ref_name = f"{date_key}.md"
        await self._upsert("dated", ref_name, block, when)
        return NoteRef(
            filename=ref_name,
            scope=self._scope,
            category="ltm-dated",
            backend="sqlite",
        )

    async def write_topic(self, slug: str, body: str) -> NoteRef:
        slug = _slugify(slug)
        stamp = now_in(self._tz)
        block = _render_dated_block(body, stamp=stamp)
        ref_name = f"topics/{slug}.md"
        await self._upsert("topic", ref_name, block, stamp)
        return NoteRef(
            filename=ref_name,
            scope=self._scope,
            category="ltm-topic",
            backend="sqlite",
        )

    async def read_main(self) -> str:
        row = await self._conn.fetchone(
            "SELECT body FROM ltm_documents WHERE kind = 'main' AND ref_name = 'MEMORY.md'"
        )
        return str(row["body"]) if row else ""

    async def search(self, text: str, *, limit: int = 5) -> List[MemoryChunk]:
        needle = text.lower().strip()
        if not needle or limit <= 0:
            return []
        keywords = [t for t in re.split(r"\s+", needle) if t]
        if not keywords:
            return []
        rows = await self._conn.fetchall("SELECT kind, ref_name, body FROM ltm_documents")
        scored: List[Tuple[float, MemoryChunk]] = []
        for row in rows:
            kind = str(row["kind"])
            ref_name = str(row["ref_name"])
            content = str(row["body"])
            density = _keyword_density(content, keywords)
            if density <= 0:
                continue
            recency = 1.0 if kind == "main" else _recency_bonus(ref_name, kind=kind, tz=self._tz)
            score = density * 0.7 + recency * 0.3
            source = {
                "main": "long_term",
                "dated": "long_term_dated",
                "topic": "long_term_topic",
            }.get(kind, "long_term")
            chunk = MemoryChunk(
                key=ref_name,
                content=_excerpt(content, keywords, max_chars=1200),
                source=source,
                relevance_score=score,
                metadata={"ltm_kind": source, "filename": ref_name},
            )
            scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])
        return [c for _, c in scored[:limit]]

    # ── snapshot helpers ────────────────────────────────────────────

    async def all_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM ltm_documents ORDER BY id ASC")
        return [dict(r) for r in rows]

    # ── internals ───────────────────────────────────────────────────

    async def _upsert(self, kind: str, ref_name: str, block: str, stamp: datetime) -> None:
        ts_iso = stamp.astimezone(self._tz).isoformat()
        existing = await self._conn.fetchone(
            "SELECT body FROM ltm_documents WHERE kind = ? AND ref_name = ?",
            (kind, ref_name),
        )
        if existing is None:
            await self._conn.execute(
                """
                INSERT INTO ltm_documents (kind, ref_name, body, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (kind, ref_name, block, ts_iso, ts_iso),
            )
            return
        prior = str(existing["body"])
        joined = _join(prior, block)
        await self._conn.execute(
            """
            UPDATE ltm_documents
               SET body = ?, updated_at = ?
             WHERE kind = ? AND ref_name = ?
            """,
            (joined, ts_iso, kind, ref_name),
        )


# ── helpers ──────────────────────────────────────────────────────────


def _join(prior: str, block: str) -> str:
    """Match the file provider's `_append`: blank-line separator unless
    the existing body already ends with one.
    """
    if not prior:
        return block.rstrip() + "\n"
    sep = "\n\n" if not prior.endswith("\n\n") else ""
    return prior + sep + block.rstrip() + "\n"


def _render_evergreen_block(body: str, *, stamp: datetime, heading: Optional[str] = None) -> str:
    lines: List[str] = [f"<!-- {stamp.strftime('%Y-%m-%d %H:%M %Z')} -->"]
    if heading:
        lines.append(f"## {heading}")
        lines.append("")
    lines.append(body.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def _render_dated_block(body: str, *, stamp: datetime) -> str:
    lines = [
        "---",
        f"_({stamp.strftime('%H:%M')} {stamp.strftime('%Z')})_",
        "",
        body.rstrip(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _slugify(raw: str) -> str:
    raw = raw.strip().lower()
    slug = _SLUG_RE.sub("_", raw).strip("_")
    return (slug or "topic")[:64]


def _keyword_density(content: str, keywords: List[str]) -> float:
    body = content.lower()
    total_words = max(1, len(body.split()))
    hits = sum(body.count(k) for k in keywords)
    return hits / total_words if hits > 0 else 0.0


def _recency_bonus(ref_name: str, *, kind: str, tz: tzinfo) -> float:
    if kind == "topic":
        return 0.6  # mid-decay default for topic files
    stem = ref_name.rsplit("/", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]
    try:
        when = datetime.fromisoformat(stem).replace(tzinfo=tz)
    except ValueError:
        return 0.5
    today = now_in(tz)
    age = today - when
    if age < timedelta(0):
        age = timedelta(0)
    age_days = age.total_seconds() / 86400.0
    return float(math.pow(2.0, -age_days / _RECENCY_HALFLIFE_DAYS))


def _excerpt(content: str, keywords: List[str], *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    low = content.lower()
    for kw in keywords:
        pos = low.find(kw)
        if pos < 0:
            continue
        start = max(0, pos - max_chars // 4)
        end = min(len(content), start + max_chars)
        snippet = content[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(content):
            snippet = snippet + "…"
        return snippet
    return content[:max_chars] + "…"


__all__ = ["_SQLLTMStore"]
