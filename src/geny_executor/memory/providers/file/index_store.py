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
from typing import Any, Dict, List, Optional, Tuple

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

    async def build_vault_map(
        self,
        *,
        recent_limit: int = 5,
        top_tags: int = 10,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Snapshot suitable for prompt-injection rendering.

        The shape mirrors the legacy host vault-map: per-category
        aggregates (file count + last_modified + optional host
        description), top tags, recently modified notes, optional
        MEMORY.md preview, plus totals.

        ``category_descriptions`` is the host-supplied label map —
        executor never has business meaning for a category, so the
        host (Geny) injects "critical = always-pinned facts", etc.
        """
        descriptions = dict(category_descriptions or {})
        payload = await self._cached_or_compute()
        files = payload.get("files") or {}
        tag_map = payload.get("tag_map") or {}

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

        tag_pairs = sorted(
            ((tag, len(names)) for tag, names in tag_map.items()),
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

        # MEMORY.md preview — best-effort. Executor doesn't try to
        # parse frontmatter; just strips a leading `---` block.
        preview = ""
        ltm_path = self._layout.main_ltm
        if ltm_path.exists():
            try:
                text = ltm_path.read_text(encoding="utf-8")
                if text.startswith("---"):
                    end = text.find("\n---", 3)
                    if end > 0:
                        text = text[end + 4 :]
                preview = text.strip()[:200]
            except OSError:
                preview = ""

        return {
            "categories": categories,
            "top_tags": tag_pairs,
            "recently_modified": recent_view,
            "memory_md_preview": preview,
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
        """Render the vault map as a markdown block ready for the
        Static Layer of the system prompt.

        Hosts that want a different shape can call ``build_vault_map``
        and render their own markdown — this is the executor's
        opinionated default (≤ 500 chars in typical use).
        """
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
                if desc:
                    lines.append(f"  - `{cat}` ({count}) — {desc}")
                else:
                    lines.append(f"  - `{cat}` ({count})")
            lines.append(
                "  Use `memory_list(category=…)` to browse a folder, "
                "`memory_read(filename=…)` for full content."
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
        preview = vmap.get("memory_md_preview") or ""
        if preview:
            single = preview.replace("\n", " ").strip()[:200]
            lines.append(f"- MEMORY.md preview: {single}")
        return "\n".join(lines)

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
