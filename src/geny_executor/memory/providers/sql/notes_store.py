"""Notes plane backed by SQLite.

`notes` carries the canonical row, `note_tags` normalises the
many-to-many tag relation, and `note_links` records both wikilink
edges (`origin='wikilink'`) parsed from the body and explicit edges
created via `link()` (`origin='explicit'`). Backlinks are derived on
read by scanning `note_links` with `target = ?`.

Behaviour mirrors the file provider's `_FilesystemNotesStore` so the
shared contract suite passes verbatim.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, tzinfo
from typing import Any, Dict, Iterable, List, Optional, Tuple

from geny_executor.memory.provider import (
    Importance,
    Note,
    NoteDraft,
    NoteGraph,
    NoteMeta,
    NotePatch,
    NoteRef,
    NotesHandle,
    Scope,
)
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.memory.providers.sql.connection import _SQLiteConnection
from geny_executor.stages.s02_context.types import MemoryChunk


_WIKILINK = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9가-힣\-]+")


class _SQLNotesStore(NotesHandle):
    """SQL-backed structured notes store."""

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

    # ── NotesHandle contract ────────────────────────────────────────

    async def list(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        importance: Optional[Importance] = None,
    ) -> List[NoteMeta]:
        clauses: List[str] = []
        params: List[Any] = []
        if category is not None:
            clauses.append("(notes.category IS ? OR notes.category = ?)")
            params.extend([None if not category else category, category or ""])
        if importance is not None:
            clauses.append("notes.importance = ?")
            params.append(importance.value)
        join = ""
        if tag is not None:
            join = "JOIN note_tags nt ON nt.filename = notes.filename"
            clauses.append("nt.tag = ?")
            params.append(tag)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT notes.* FROM notes {join} {where} ORDER BY notes.filename ASC"
        rows = await self._conn.fetchall(sql, params)
        out: List[NoteMeta] = []
        for row in rows:
            note = await self._row_to_note(row)
            out.append(note.as_meta())
        return out

    async def read(self, filename: str) -> Optional[Note]:
        row = await self._conn.fetchone(
            "SELECT * FROM notes WHERE filename = ?",
            (filename,),
        )
        if row is None:
            return None
        return await self._row_to_note(row)

    async def write(self, draft: NoteDraft) -> NoteMeta:
        filename = await self._resolve_filename(draft)
        now = now_in(self._tz)
        ts = now.isoformat()
        front = json.dumps(draft.frontmatter or {}, ensure_ascii=False)
        await self._conn.execute(
            """
            INSERT INTO notes (
                filename, title, body, importance, category, scope, backend,
                frontmatter_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                draft.title,
                draft.body,
                draft.importance.value,
                draft.category,
                self._scope.value,
                "sqlite",
                front,
                ts,
                ts,
            ),
        )
        await self._replace_tags(filename, draft.tags)
        await self._replace_wikilinks(filename, draft.body)
        return NoteMeta(
            ref=NoteRef(
                filename=filename,
                scope=self._scope,
                category=draft.category,
                backend="sqlite",
            ),
            title=draft.title,
            importance=draft.importance,
            tags=list(draft.tags),
            category=draft.category,
            created_at=now,
            updated_at=now,
            size_bytes=len(draft.body.encode("utf-8")),
            backlinks=await self._count_backlinks(filename),
        )

    async def update(self, filename: str, patch: NotePatch) -> NoteMeta:
        current = await self.read(filename)
        if current is None:
            raise KeyError(f"note {filename!r} not found")
        if patch.title is not None:
            current.title = patch.title
        if patch.append_body is not None:
            current.body = (current.body + "\n\n" + patch.append_body).strip()
        elif patch.body is not None:
            current.body = patch.body
        if patch.importance is not None:
            current.importance = patch.importance
        if patch.tags is not None:
            current.tags = list(patch.tags)
        if patch.category is not None:
            current.category = patch.category
        if patch.frontmatter is not None:
            current.frontmatter = dict(patch.frontmatter)
        current.updated_at = now_in(self._tz)
        await self._conn.execute(
            """
            UPDATE notes
               SET title = ?, body = ?, importance = ?, category = ?,
                   frontmatter_json = ?, updated_at = ?
             WHERE filename = ?
            """,
            (
                current.title,
                current.body,
                current.importance.value,
                current.category,
                json.dumps(current.frontmatter or {}, ensure_ascii=False),
                current.updated_at.isoformat(),
                filename,
            ),
        )
        await self._replace_tags(filename, current.tags)
        await self._replace_wikilinks(filename, current.body)
        meta = current.as_meta()
        meta.backlinks = await self._count_backlinks(filename)
        return meta

    async def delete(self, filename: str) -> bool:
        # Cascade is on note_tags / note_links FKs; provider_meta etc.
        # are independent.
        _, count = await self._conn.execute_returning(
            "DELETE FROM notes WHERE filename = ?",
            (filename,),
        )
        # Also drop any explicit/wikilink rows that referenced the
        # deleted file as a target — those don't cascade since the
        # target is plain text, not a FK.
        await self._conn.execute(
            "DELETE FROM note_links WHERE target = ?",
            (filename,),
        )
        return count > 0

    async def link(self, source: str, target: str) -> bool:
        exists = await self._conn.fetchone(
            "SELECT 1 FROM notes WHERE filename = ?",
            (source,),
        )
        if exists is None:
            raise KeyError(f"source note {source!r} not found")
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO note_links (source, target, origin)
            VALUES (?, ?, 'explicit')
            """,
            (source, target),
        )
        return True

    async def graph(self) -> NoteGraph:
        note_rows = await self._conn.fetchall("SELECT * FROM notes ORDER BY filename ASC")
        nodes: List[NoteMeta] = []
        for row in note_rows:
            note = await self._row_to_note(row)
            nodes.append(note.as_meta())
        edges_rows = await self._conn.fetchall(
            "SELECT source, target FROM note_links ORDER BY source, target"
        )
        seen: List[Tuple[str, str]] = []
        seen_set: set[Tuple[str, str]] = set()
        for row in edges_rows:
            edge = (str(row["source"]), str(row["target"]))
            if edge in seen_set:
                continue
            seen_set.add(edge)
            seen.append(edge)
        return NoteGraph(nodes=nodes, edges=seen)

    async def search(
        self,
        text: str,
        *,
        limit: int = 5,
        importance_floor: Importance = Importance.LOW,
    ) -> List[MemoryChunk]:
        needle = text.lower().strip()
        if not needle or limit <= 0:
            return []
        floor = importance_floor.boost
        keywords = [t for t in re.split(r"\s+", needle) if t]
        if not keywords:
            return []
        rows = await self._conn.fetchall("SELECT * FROM notes")
        keyword_set = set(keywords)
        scored: List[Tuple[float, MemoryChunk]] = []
        for row in rows:
            note = await self._row_to_note(row)
            if note.importance.boost < floor:
                continue
            haystack = (note.title + "\n" + note.body).lower()
            keyword_hits = sum(haystack.count(k) for k in keywords)
            tag_lower = {t.lower() for t in (note.tags or [])}
            tag_overlap = len(keyword_set & tag_lower)
            if keyword_hits == 0 and tag_overlap == 0:
                continue
            score = (1.0 + keyword_hits) * note.importance.boost + 0.3 * tag_overlap
            scored.append(
                (
                    score,
                    MemoryChunk(
                        key=note.ref.filename,
                        content=note.body[:1200],
                        source="note",
                        relevance_score=score,
                        metadata={
                            "title": note.title,
                            "importance": note.importance.value,
                            "tags": list(note.tags),
                            "category": note.category,
                        },
                    ),
                )
            )
        scored.sort(key=lambda pair: -pair[0])
        return [chunk for _, chunk in scored[:limit]]

    # ── snapshot helpers ────────────────────────────────────────────

    async def all(self) -> List[Note]:
        rows = await self._conn.fetchall("SELECT * FROM notes ORDER BY filename ASC")
        return [await self._row_to_note(r) for r in rows]

    async def all_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM notes ORDER BY filename ASC")
        return [dict(r) for r in rows]

    async def all_tag_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM note_tags ORDER BY filename, tag")
        return [dict(r) for r in rows]

    async def all_link_rows(self) -> List[dict]:
        rows = await self._conn.fetchall("SELECT * FROM note_links ORDER BY source, target, origin")
        return [dict(r) for r in rows]

    # ── internals ───────────────────────────────────────────────────

    async def _resolve_filename(self, draft: NoteDraft) -> str:
        if draft.filename:
            return draft.filename
        base = _slug(draft.title) + ".md"
        existing = await self._conn.fetchone(
            "SELECT filename FROM notes WHERE filename = ?", (base,)
        )
        if existing is None:
            return base
        return f"{_slug(draft.title)}-{uuid.uuid4().hex[:6]}.md"

    async def _replace_tags(self, filename: str, tags: Iterable[str]) -> None:
        await self._conn.execute("DELETE FROM note_tags WHERE filename = ?", (filename,))
        unique = list(dict.fromkeys(tags))
        if not unique:
            return
        await self._conn.executemany(
            "INSERT OR IGNORE INTO note_tags (filename, tag) VALUES (?, ?)",
            [(filename, str(t)) for t in unique],
        )

    async def _replace_wikilinks(self, filename: str, body: str) -> None:
        await self._conn.execute(
            "DELETE FROM note_links WHERE source = ? AND origin = 'wikilink'",
            (filename,),
        )
        targets = _extract_links(body)
        if not targets:
            return
        await self._conn.executemany(
            """
            INSERT OR IGNORE INTO note_links (source, target, origin)
            VALUES (?, ?, 'wikilink')
            """,
            [(filename, t) for t in targets],
        )

    async def _count_backlinks(self, filename: str) -> int:
        row = await self._conn.fetchone(
            "SELECT COUNT(DISTINCT source) AS n FROM note_links WHERE target = ?",
            (filename,),
        )
        return int(row["n"]) if row else 0

    async def _row_to_note(self, row: Any) -> Note:
        filename = str(row["filename"])
        tag_rows = await self._conn.fetchall(
            "SELECT tag FROM note_tags WHERE filename = ? ORDER BY tag",
            (filename,),
        )
        tags = [str(r["tag"]) for r in tag_rows]
        link_rows = await self._conn.fetchall(
            "SELECT target FROM note_links WHERE source = ? ORDER BY target",
            (filename,),
        )
        links_out = []
        seen: set[str] = set()
        for r in link_rows:
            t = str(r["target"])
            if t in seen:
                continue
            seen.add(t)
            links_out.append(t)
        backlink_rows = await self._conn.fetchall(
            "SELECT DISTINCT source FROM note_links WHERE target = ? ORDER BY source",
            (filename,),
        )
        links_in = [str(r["source"]) for r in backlink_rows]
        importance = _parse_importance(row["importance"])
        scope = _parse_scope(row["scope"])
        front = _parse_json_dict(row["frontmatter_json"])
        return Note(
            ref=NoteRef(
                filename=filename,
                scope=scope,
                category=_optional_str(row["category"]),
                backend=str(row["backend"] or "sqlite"),
            ),
            title=str(row["title"]),
            body=str(row["body"]),
            importance=importance,
            tags=tags,
            category=_optional_str(row["category"]),
            frontmatter=front,
            links_out=links_out,
            links_in=links_in,
            created_at=_parse_ts(row["created_at"]),
            updated_at=_parse_ts(row["updated_at"]),
        )


# ── helpers ──────────────────────────────────────────────────────────


def _slug(title: str) -> str:
    slug = _SLUG_RE.sub("-", title).strip("-").lower() or "note"
    return slug[:80]


def _extract_links(body: str) -> List[str]:
    seen: List[str] = []
    for match in _WIKILINK.finditer(body):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def _parse_importance(raw: Any) -> Importance:
    if not raw:
        return Importance.MEDIUM
    try:
        return Importance(str(raw).lower())
    except ValueError:
        return Importance.MEDIUM


def _parse_scope(raw: Any) -> Scope:
    if not raw:
        return Scope.SESSION
    try:
        return Scope(str(raw).lower())
    except ValueError:
        return Scope.SESSION


def _parse_ts(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _parse_json_dict(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except (TypeError, ValueError):
        return {}


def _optional_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw)
    return s if s else None


__all__ = ["_SQLNotesStore"]
