"""Derived index plane backed by `memory/_index.json`.

The index is a *cache* of derivable facts (tag counts, wikilink graph,
per-file summary). It can always be rebuilt by rescanning the notes
store. Writes update the cache in place; `rebuild()` discards the
cache and regenerates it from disk.

This module intentionally does not talk to disk *except* via the
notes store it wraps — duplicating scan logic here would drift from
the format that `_FilesystemNotesStore` writes.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import tzinfo
from typing import Any, Dict, List, Tuple

from geny_executor.memory.provider import NoteGraph, NoteMeta
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore
from geny_executor.memory.providers.file.timezone import now_in


class _FileIndexStore:
    """Read-mostly view on the notes store with on-disk cache."""

    def __init__(
        self,
        notes: _FilesystemNotesStore,
        *,
        layout: DirectoryLayout,
        tz: tzinfo,
    ) -> None:
        self._notes = notes
        self._layout = layout
        self._tz = tz
        self._lock = asyncio.Lock()

    # ── IndexHandle contract ────────────────────────────────────────

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            payload = await self._compute()
        self._write_cache(payload)
        return payload

    async def tag_counts(self) -> Dict[str, int]:
        payload = await self._cached_or_compute()
        counter: Counter[str] = Counter()
        for entry in payload.get("files", {}).values():
            for tag in entry.get("tags", []):
                counter[str(tag)] += 1
        return dict(counter)

    async def graph(self) -> NoteGraph:
        payload = await self._cached_or_compute()
        nodes: List[NoteMeta] = []
        edges: List[Tuple[str, str]] = []
        notes = await self._notes.all()
        by_name = {n.ref.filename: n for n in notes}
        for fname, entry in payload.get("files", {}).items():
            note = by_name.get(fname)
            if note is not None:
                nodes.append(note.as_meta())
        for src, targets in payload.get("link_graph", {}).items():
            for tgt in targets or []:
                edges.append((src, tgt))
        return NoteGraph(nodes=nodes, edges=edges)

    async def rebuild(self) -> None:
        async with self._lock:
            await self._notes.clear_cache()
            payload = await self._compute()
            self._write_cache(payload)

    # ── internal ────────────────────────────────────────────────────

    async def _cached_or_compute(self) -> Dict[str, Any]:
        cached = self._read_cache()
        if cached is not None:
            return cached
        async with self._lock:
            payload = await self._compute()
        self._write_cache(payload)
        return payload

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

    def _read_cache(self) -> Dict[str, Any] | None:
        path = self._layout.index_json
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _write_cache(self, payload: Dict[str, Any]) -> None:
        self._layout.memory.mkdir(parents=True, exist_ok=True)
        self._layout.index_json.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )


# ── helpers ──────────────────────────────────────────────────────────


def _summary(body: str, *, limit: int = 200) -> str:
    """Take the first non-heading paragraph of `body` and trim to
    `limit` chars. Matches Geny's `_summary` for diff-friendliness.
    """
    for para in body.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped[:limit]
    return body.strip()[:limit]
