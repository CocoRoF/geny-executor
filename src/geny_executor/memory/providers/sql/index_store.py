"""Derived index plane for SQLMemoryProvider.

The index is a *derived view* over `notes`, `note_tags`, and
`note_links`. Unlike the file provider, there is no on-disk JSON cache
to materialise — the database itself is the canonical source. We
still expose `snapshot()` returning the same payload shape so callers
that consume the index (e.g. the web mirror) see one schema across
backends.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import tzinfo
from typing import Any, Dict, List, Optional

from geny_executor.memory.provider import NoteGraph
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.memory.providers.sql.connection import _SQLConnection
from geny_executor.memory.providers.sql.notes_store import _SQLNotesStore


class _SQLIndexStore:
    """Read-mostly view on the SQL notes store."""

    def __init__(
        self,
        notes: _SQLNotesStore,
        *,
        conn: _SQLConnection,
        tz: tzinfo,
    ) -> None:
        self._notes = notes
        self._conn = conn
        self._tz = tz
        self._lock = asyncio.Lock()

    # ── IndexHandle contract ────────────────────────────────────────

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return await self._compute()

    async def tag_counts(self) -> Dict[str, int]:
        rows = await self._conn.fetchall("SELECT tag FROM note_tags")
        counter: Counter[str] = Counter()
        for row in rows:
            counter[str(row["tag"])] += 1
        return dict(counter)

    async def graph(self) -> NoteGraph:
        return await self._notes.graph()

    async def rebuild(self) -> None:
        # No materialised cache. The hook is preserved so callers can
        # treat the SQL provider exactly like the file provider.
        return None

    async def build_vault_map(
        self,
        *,
        recent_limit: int = 5,
        top_tags: int = 10,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        descriptions = dict(category_descriptions or {})
        payload = await self._compute()
        files = payload.get("files") or {}
        categories: Dict[str, Dict[str, Any]] = {}
        for entry in files.values():
            cat = entry.get("category") or "root"
            slot = categories.setdefault(
                cat,
                {
                    "files": 0,
                    "last_modified": "",
                    "description": descriptions.get(cat, ""),
                },
            )
            slot["files"] += 1
            modified = entry.get("modified") or ""
            if modified > slot["last_modified"]:
                slot["last_modified"] = modified
        tag_map = payload.get("tag_map") or {}
        tag_pairs = sorted(
            ((t, len(names)) for t, names in tag_map.items()),
            key=lambda x: -x[1],
        )[:top_tags]
        recent = sorted(
            files.values(),
            key=lambda f: f.get("modified") or "",
            reverse=True,
        )[:recent_limit]
        recent_view = [
            {
                "filename": f.get("filename"),
                "title": f.get("title") or f.get("filename"),
                "category": f.get("category") or "root",
                "modified": f.get("modified", ""),
            }
            for f in recent
        ]
        return {
            "categories": categories,
            "top_tags": tag_pairs,
            "recently_modified": recent_view,
            "memory_md_preview": "",
            "total_files": payload.get("total_files", len(files)),
            "generated_at": now_in(self._tz).isoformat(),
        }

    async def render_vault_map(
        self,
        *,
        recent_limit: int = 5,
        top_tags: int = 10,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        vmap = await self.build_vault_map(
            recent_limit=recent_limit,
            top_tags=top_tags,
            category_descriptions=category_descriptions,
        )
        lines: List[str] = ["## Vault Map"]
        cats = vmap.get("categories") or {}
        if cats:
            lines.append("- Categories:")
            for cat, slot in sorted(cats.items()):
                count = int(slot.get("files") or 0)
                desc = (slot.get("description") or "").strip()
                lines.append(
                    f"  - `{cat}` ({count}) — {desc}" if desc else f"  - `{cat}` ({count})"
                )
        tags = vmap.get("top_tags") or []
        if tags:
            tag_summary = ", ".join(f"{t}({n})" for t, n in tags[:5])
            lines.append(f"- Top tags: {tag_summary}")
        recent = vmap.get("recently_modified") or []
        if recent:
            lines.append("- Recently modified:")
            for r in recent:
                lines.append(f"  - `{r['filename']}` — {r.get('title') or ''}")
        return "\n".join(lines)

    # ── internals ───────────────────────────────────────────────────

    async def _compute(self) -> Dict[str, Any]:
        notes = await self._notes.all()
        files: Dict[str, Dict[str, Any]] = {}
        tag_map: Dict[str, List[str]] = {}
        link_graph: Dict[str, List[str]] = {}
        for note in notes:
            fname = note.ref.filename
            files[fname] = {
                "filename": fname,
                "title": note.title,
                "category": note.category or "root",
                "tags": list(note.tags or []),
                "importance": note.importance.value,
                "created": note.created_at.isoformat() if note.created_at else "",
                "modified": note.updated_at.isoformat() if note.updated_at else "",
                "char_count": len(note.body or ""),
                "links_to": list(note.links_out or []),
                "linked_from": list(note.links_in or []),
                "summary": _summary(note.body or ""),
            }
            for tag in note.tags or []:
                tag_map.setdefault(tag, []).append(fname)
            if note.links_out:
                link_graph[fname] = list(note.links_out)

        return {
            "files": files,
            "tag_map": {tag: sorted(names) for tag, names in tag_map.items()},
            "link_graph": link_graph,
            "last_rebuilt": now_in(self._tz).isoformat(),
            "total_files": len(files),
            "total_chars": sum(e["char_count"] for e in files.values()),
        }


def _summary(body: str, *, limit: int = 200) -> str:
    for para in body.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped[:limit]
    return body.strip()[:limit]


__all__ = ["_SQLIndexStore"]
