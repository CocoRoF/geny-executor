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

import json
import os
import tempfile
from collections import Counter
from datetime import tzinfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.provider import NoteGraph, NoteMeta, NoteOutline, NoteSummary
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore
from geny_executor.memory.providers.file.timezone import now_in


class _FileIndexStore:
    """Read-mostly view on the notes store with on-disk cache.

    The store owns the on-disk hierarchical sidecars:

    - ``<root>/_index.json`` — flat canonical inventory (all files +
      tag map + link graph). Updated on every write.
    - ``<cat>/_index.json`` — per-category shard listing only that
      category's notes + tag counts + canonical description. Updated
      incrementally per affected category.
    - ``<root>/_summary.json`` — folder-tree overview (every category
      + file_count + description). Updated on every write because the
      total + per-category counts shift.

    Hosts inject ``category_descriptions`` at provider construction so
    the canonical labels (Geny's "critical = always-pinned facts" etc.)
    show up uniformly in every sidecar.
    """

    SUBINDEX_FILENAME = "_index.json"
    SUMMARY_FILENAME = "_summary.json"

    def __init__(
        self,
        notes: _FilesystemNotesStore,
        *,
        layout: DirectoryLayout,
        tz: tzinfo,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        self._notes = notes
        self._layout = layout
        self._tz = tz
        self._lock = LoopAgnosticLock()
        self._category_descriptions: Dict[str, str] = dict(category_descriptions or {})

    def set_category_descriptions(self, descriptions: Dict[str, str]) -> None:
        """Late-set or replace the canonical description map. Called
        when hosts switch hooks; the next sidecar refresh picks up
        the new labels.
        """
        self._category_descriptions = dict(descriptions or {})

    # ── IndexHandle contract ────────────────────────────────────────

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            payload = await self._compute()
        self._write_cache(payload)
        # snapshot() is a read-side operation that also refreshes the
        # disk cache; piggy-back on it to refresh hierarchical sidecars
        # so a fresh read sees consistent shards.
        self._write_hierarchical_sidecars(payload, category=None)
        return payload

    async def refresh_for_category(self, category: Optional[str]) -> None:
        """Incrementally refresh the sidecars affected by a single
        category change. Always rewrites:

        - ``<root>/_index.json`` (flat — totals shift)
        - ``<root>/_summary.json`` (folder tree — counts shift)
        - ``<cat>/_index.json`` for the changed category

        Other category shards are left untouched. Called after every
        ``NotesStore.write`` / ``update`` / ``delete``.
        """
        async with self._lock:
            payload = await self._compute()
        self._write_cache(payload)
        self._write_hierarchical_sidecars(payload, category=category)

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

    async def list_notes(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List["NoteSummary"]:
        from geny_executor.memory._progressive import make_summary

        notes = await self._notes.all()
        tag_lower = tag.lower() if isinstance(tag, str) else None
        filtered = []
        for n in notes:
            cat = n.category or "root"
            if category is not None and cat != category:
                continue
            if tag_lower is not None:
                tags_lower = {str(t).lower() for t in (n.tags or [])}
                if tag_lower not in tags_lower:
                    continue
            modified = n.updated_at.isoformat() if n.updated_at else ""
            filtered.append((modified, n))
        filtered.sort(key=lambda pair: (pair[0], pair[1].ref.filename), reverse=True)
        sliced = filtered[offset : offset + max(0, int(limit))]
        return [
            make_summary(
                filename=n.ref.filename,
                title=n.title,
                category=n.category or "root",
                tags=list(n.tags or []),
                importance=n.importance.value
                if hasattr(n.importance, "value")
                else str(n.importance),
                body=n.body or "",
                modified=modified,
            )
            for modified, n in sliced
        ]

    async def read_outline(self, filename: str) -> Optional["NoteOutline"]:
        from geny_executor.memory._progressive import parse_outline

        note = await self._notes.read(filename)
        if note is None:
            return None
        return parse_outline(filename=filename, title=note.title, body=note.body or "")

    async def read_section(self, filename: str, heading: str) -> Optional[str]:
        from geny_executor.memory._progressive import extract_section

        note = await self._notes.read(filename)
        if note is None:
            return None
        return extract_section(note.body or "", heading)

    async def list_categories(self) -> List[Dict[str, Any]]:
        """Every direct subdirectory of `memory/` (canonical + host-defined),
        with file_count from the snapshot. Empty folders are included
        with `file_count=0` so hosts can render a category sidebar
        before any note has been written.
        """
        snap = await self._cached_or_compute()
        files_by_cat: Dict[str, int] = {}
        for entry in (snap.get("files") or {}).values():
            cat = entry.get("category") or "root"
            files_by_cat[cat] = files_by_cat.get(cat, 0) + 1

        result: List[Dict[str, Any]] = []
        seen: set = set()
        for cat_dir in self._layout.category_dirs():
            cat_name = "root" if cat_dir == self._layout.memory else cat_dir.name
            if cat_name in seen:
                continue
            seen.add(cat_name)
            try:
                rel_path = str(cat_dir.relative_to(self._layout.root))
            except ValueError:
                rel_path = str(cat_dir)
            result.append(
                {
                    "name": cat_name,
                    "file_count": files_by_cat.get(cat_name, 0),
                    "path": rel_path,
                    "exists": cat_dir.exists(),
                }
            )
        return result

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
        _atomic_write_json(self._layout.index_json, payload)

    # ── hierarchical sidecars (EXEC-5) ──────────────────────────────

    def _write_hierarchical_sidecars(
        self,
        payload: Dict[str, Any],
        *,
        category: Optional[str],
    ) -> None:
        """Write per-category ``<cat>/_index.json`` shards + the root
        ``_summary.json`` overview.

        ``category=None`` rewrites every category shard (used by full
        ``snapshot()``); a specific value rewrites only that one shard
        (used by ``refresh_for_category`` per write/update/delete).
        The root summary is always rewritten because totals shift on
        any change.
        """
        self._layout.memory.mkdir(parents=True, exist_ok=True)

        files = (payload.get("files") or {}).values()
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for entry in files:
            cat = entry.get("category") or "root"
            by_cat.setdefault(cat, []).append(entry)

        # Discover canonical + on-disk categories so empty folders
        # still receive an empty shard ("category exists" signal).
        all_cats: Set[str] = set(by_cat.keys()) | {"root"}
        for cat_dir in self._layout.category_dirs():
            cat_name = "root" if cat_dir == self._layout.memory else cat_dir.name
            all_cats.add(cat_name)

        targets: Iterable[str]
        if category is None:
            targets = sorted(all_cats)
        else:
            # Always include ``root`` alongside the named category so
            # the root sidecars stay consistent.
            targets = sorted({category, "root"})

        for cat in targets:
            # root → the canonical flat ``_index.json`` already lives at
            # ``<memory>/_index.json`` (written by ``_write_cache``).
            # Per-category shards have the same filename inside their
            # own folder; if we wrote a "root shard" we'd overwrite the
            # flat index. Skip — root coverage stays in the flat index
            # plus the root summary.
            if cat == "root":
                continue
            cat_files = by_cat.get(cat, [])
            cat_dir = self._layout.memory / cat
            try:
                cat_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            tag_counts: Dict[str, int] = {}
            for f in cat_files:
                for t in f.get("tags") or []:
                    tag_counts[str(t)] = tag_counts.get(str(t), 0) + 1
            shard_payload = {
                "version": "2",
                "category": cat,
                "description": self._category_descriptions.get(cat, ""),
                "file_count": len(cat_files),
                "files": {f["filename"]: f for f in cat_files if f.get("filename")},
                "tag_counts": tag_counts,
                "last_rebuilt": now_in(self._tz).isoformat(),
            }
            shard_path = cat_dir / self.SUBINDEX_FILENAME
            try:
                _atomic_write_json(shard_path, shard_payload)
            except OSError:
                continue

        # Always rewrite the root folder-tree summary.
        summary = {
            "version": "2",
            "categories": [
                {
                    "name": cat,
                    "file_count": len(by_cat.get(cat, [])),
                    "path": ("memory" if cat == "root" else f"memory/{cat}"),
                    "description": self._category_descriptions.get(cat, ""),
                    "exists": True,
                }
                for cat in sorted(all_cats)
            ],
            "category_descriptions": dict(self._category_descriptions),
            "generated_at": now_in(self._tz).isoformat(),
        }
        try:
            _atomic_write_json(self._layout.memory / self.SUMMARY_FILENAME, summary)
        except OSError:
            pass


# ── helpers ──────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Tempfile + os.replace atomic JSON write — never leaves a
    half-written sidecar visible to readers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".json.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
