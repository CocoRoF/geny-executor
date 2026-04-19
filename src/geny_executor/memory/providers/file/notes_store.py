"""Structured notes plane backed by markdown + YAML frontmatter.

Notes live under ``memory/{category}/*.md`` (or directly under
``memory/`` for category ``root``). Each file has a ``---``-delimited
frontmatter block (see `frontmatter.py`) followed by a markdown body.

Matches Geny's on-disk format so a legacy Geny reader can consume the
output (or vice versa). No Geny code is imported.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
from geny_executor.memory.providers.file import frontmatter
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.stages.s02_context.types import MemoryChunk


_WIKILINK = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9가-힣\-]+")


class _FilesystemNotesStore(NotesHandle):
    """File-backed notes store.

    The in-memory state is a lazy cache of `{filename -> Note}` rebuilt
    on first access and after any write. A full scan over `memory/`
    rebuilds the cache; individual ops keep it consistent by mutating
    the cache in place.
    """

    def __init__(
        self,
        layout: DirectoryLayout,
        *,
        tz: tzinfo,
        scope: Scope = Scope.SESSION,
    ) -> None:
        self._layout = layout
        self._tz = tz
        self._scope = scope
        self._lock = asyncio.Lock()
        self._cache: Dict[str, Note] = {}
        self._loaded = False
        self._explicit_links: Dict[str, Set[str]] = {}

    # ── NotesHandle contract ────────────────────────────────────────

    async def list(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        importance: Optional[Importance] = None,
    ) -> List[NoteMeta]:
        async with self._lock:
            await self._ensure_loaded()
            out: List[NoteMeta] = []
            for note in self._cache.values():
                if category is not None and (note.category or "") != category:
                    continue
                if tag is not None and tag not in (note.tags or []):
                    continue
                if importance is not None and note.importance != importance:
                    continue
                out.append(note.as_meta())
        return out

    async def read(self, filename: str) -> Optional[Note]:
        async with self._lock:
            await self._ensure_loaded()
            return _clone(self._cache.get(filename))

    async def write(self, draft: NoteDraft) -> NoteMeta:
        async with self._lock:
            await self._ensure_loaded()
            filename = self._resolve_filename(draft)
            now = now_in(self._tz)
            note = Note(
                ref=NoteRef(
                    filename=filename,
                    scope=self._scope,
                    category=draft.category,
                    backend="filesystem",
                ),
                title=draft.title,
                body=draft.body,
                importance=draft.importance,
                tags=list(draft.tags),
                category=draft.category,
                frontmatter=dict(draft.frontmatter or {}),
                links_out=_extract_links(draft.body),
                created_at=now,
                updated_at=now,
            )
            self._write_to_disk(note)
            self._cache[filename] = note
            self._refresh_backlinks()
            return note.as_meta()

    async def update(self, filename: str, patch: NotePatch) -> NoteMeta:
        async with self._lock:
            await self._ensure_loaded()
            current = self._cache.get(filename)
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
            current.links_out = _extract_links(current.body)
            current.updated_at = now_in(self._tz)

            self._write_to_disk(current)
            self._refresh_backlinks()
            return current.as_meta()

    async def delete(self, filename: str) -> bool:
        async with self._lock:
            await self._ensure_loaded()
            note = self._cache.pop(filename, None)
            if note is None:
                return False
            path = self._layout.note_path(note.category or "root", filename)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self._explicit_links.pop(filename, None)
            for tgts in self._explicit_links.values():
                tgts.discard(filename)
            self._refresh_backlinks()
            return True

    async def link(self, source: str, target: str) -> bool:
        async with self._lock:
            await self._ensure_loaded()
            if source not in self._cache:
                raise KeyError(f"source note {source!r} not found")
            self._explicit_links.setdefault(source, set()).add(target)
            self._refresh_backlinks()
            return True

    async def graph(self) -> NoteGraph:
        async with self._lock:
            await self._ensure_loaded()
            nodes = [n.as_meta() for n in self._cache.values()]
            edges: List[Tuple[str, str]] = []
            for fname, note in self._cache.items():
                for tgt in note.links_out:
                    edges.append((fname, tgt))
                for tgt in self._explicit_links.get(fname, ()):
                    if tgt not in note.links_out:
                        edges.append((fname, tgt))
            return NoteGraph(nodes=nodes, edges=edges)

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
        keyword_set = set(keywords)

        async with self._lock:
            await self._ensure_loaded()
            scored: List[Tuple[float, MemoryChunk]] = []
            for fname, note in self._cache.items():
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
                            key=fname,
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

    # ── snapshot / restore helpers ─────────────────────────────────

    async def all(self) -> List[Note]:
        async with self._lock:
            await self._ensure_loaded()
            return [_clone(n) for n in self._cache.values() if n is not None]

    async def replace_all(self, notes: Iterable[Note]) -> None:
        async with self._lock:
            # Wipe every file we currently know about
            for fname, note in list(self._cache.items()):
                path = self._layout.note_path(note.category or "root", fname)
                path.unlink(missing_ok=True)
            self._cache.clear()
            self._explicit_links.clear()
            for note in notes:
                self._cache[note.ref.filename] = note
                self._write_to_disk(note)
            self._refresh_backlinks()
            self._loaded = True

    async def clear_cache(self) -> None:
        """Invalidate the in-memory cache so the next op rescans disk.
        Useful after a snapshot restore that rewrote files directly.
        """
        async with self._lock:
            self._cache.clear()
            self._explicit_links.clear()
            self._loaded = False

    # ── internal ────────────────────────────────────────────────────

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._cache.clear()
        self._explicit_links.clear()
        for category_dir in self._layout.category_dirs():
            if not category_dir.exists():
                continue
            for path in sorted(category_dir.glob("*.md")):
                rel = path.relative_to(self._layout.memory)
                if self._layout.is_reserved(rel):
                    continue
                note = self._load_note(path)
                if note is not None:
                    self._cache[note.ref.filename] = note
        self._refresh_backlinks()
        self._loaded = True

    def _resolve_filename(self, draft: NoteDraft) -> str:
        if draft.filename:
            return draft.filename
        base = _slug(draft.title) + ".md"
        if base not in self._cache:
            return base
        return f"{_slug(draft.title)}-{uuid.uuid4().hex[:6]}.md"

    def _refresh_backlinks(self) -> None:
        link_map: Dict[str, List[str]] = {}
        for fname, note in self._cache.items():
            for tgt in note.links_out:
                link_map.setdefault(tgt, []).append(fname)
            for tgt in self._explicit_links.get(fname, ()):
                if tgt not in note.links_out:
                    link_map.setdefault(tgt, []).append(fname)
        for fname, note in self._cache.items():
            note.links_in = list(dict.fromkeys(link_map.get(fname, [])))

    def _write_to_disk(self, note: Note) -> None:
        category = note.category or "root"
        dir_path = self._layout.note_dir(category)
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / note.ref.filename
        meta = _note_to_frontmatter(note, tz=self._tz)
        header = frontmatter.dump(meta)
        body = note.body.rstrip() + "\n"
        path.write_text(header + body, encoding="utf-8")

    def _load_note(self, path: Path) -> Optional[Note]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        meta, body = frontmatter.split(text)
        body = body.rstrip("\n")
        raw_cat = meta.get("category")
        fs_cat = self._layout.category_of(path)
        if raw_cat and str(raw_cat) != "root":
            category_value: Optional[str] = str(raw_cat)
        elif fs_cat and fs_cat != "root":
            category_value = fs_cat
        else:
            category_value = None
        importance = _parse_importance(meta.get("importance"))
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        title = meta.get("title") or _fallback_title(body) or path.stem
        created = _parse_ts(meta.get("created"))
        modified = _parse_ts(meta.get("modified"))
        links_out_meta = meta.get("links_to") or []
        if isinstance(links_out_meta, str):
            links_out_meta = [links_out_meta]
        links_out = list(links_out_meta) if isinstance(links_out_meta, list) else []
        if not links_out:
            links_out = _extract_links(body)
        rel = path.relative_to(self._layout.memory)
        # Notes directly under memory/ have rel.parent == "."
        filename = path.name if len(rel.parts) <= 1 else rel.parts[-1]
        return Note(
            ref=NoteRef(
                filename=filename,
                scope=self._scope,
                category=category_value,
                backend="filesystem",
            ),
            title=str(title),
            body=body,
            importance=importance,
            tags=[str(t) for t in tags],
            category=category_value,
            frontmatter={
                k: v
                for k, v in meta.items()
                if k
                not in {
                    "title",
                    "tags",
                    "category",
                    "importance",
                    "created",
                    "modified",
                    "links_to",
                    "linked_from",
                }
            },
            links_out=[str(t) for t in links_out],
            links_in=[],
            created_at=created,
            updated_at=modified,
        )


# ── module helpers ───────────────────────────────────────────────────


def _slug(title: str) -> str:
    slug = _SLUG_RE.sub("-", title).strip("-").lower() or "note"
    return slug[:80]


def _extract_links(body: str) -> List[str]:
    """Extract wikilink targets from `body`, preserving first-seen
    order and deduping. Aliases (`[[target|alias]]`) collapse to the
    target.
    """
    seen: List[str] = []
    for match in _WIKILINK.finditer(body):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def _note_to_frontmatter(note: Note, *, tz: tzinfo) -> Dict[str, Any]:
    """Produce the YAML-like frontmatter mapping for `note`."""
    created = note.created_at or now_in(tz)
    modified = note.updated_at or created
    meta: Dict[str, Any] = {
        "title": note.title,
        "tags": list(note.tags),
        "category": note.category or "root",
        "importance": note.importance.value,
        "created": created.astimezone(tz).isoformat(),
        "modified": modified.astimezone(tz).isoformat(),
    }
    if note.links_out:
        meta["links_to"] = list(note.links_out)
    # Preserve any caller-supplied frontmatter keys not already claimed
    for k, v in (note.frontmatter or {}).items():
        if k not in meta:
            meta[k] = v
    return meta


def _parse_importance(raw: Any) -> Importance:
    if isinstance(raw, Importance):
        return raw
    if not raw:
        return Importance.MEDIUM
    try:
        return Importance(str(raw).lower())
    except ValueError:
        return Importance.MEDIUM


def _parse_ts(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _fallback_title(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line.strip():
            return line.strip()
    return ""


def _clone(note: Optional[Note]) -> Optional[Note]:
    if note is None:
        return None
    return Note(
        ref=note.ref,
        title=note.title,
        body=note.body,
        importance=note.importance,
        tags=list(note.tags),
        category=note.category,
        frontmatter=dict(note.frontmatter or {}),
        links_out=list(note.links_out),
        links_in=list(note.links_in),
        created_at=note.created_at,
        updated_at=note.updated_at,
    )
