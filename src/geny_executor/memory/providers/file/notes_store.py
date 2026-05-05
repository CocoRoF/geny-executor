"""Structured notes plane backed by markdown + YAML frontmatter.

Notes live under ``memory/{category}/*.md`` (or directly under
``memory/`` for category ``root``). Each file has a ``---``-delimited
frontmatter block (see `frontmatter.py`) followed by a markdown body.

Matches Geny's on-disk format so a legacy Geny reader can consume the
output (or vice versa). No Geny code is imported.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.provider import (
    Importance,
    MemoryHooks,
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

logger = logging.getLogger(__name__)


# Callback signature for the auto-vector indexing hook. The notes
# store invokes this after every successful ``write`` / ``update`` so
# the vector layer can keep its index in lockstep with markdown disk
# state. Returning the chunk count is informational; failures are
# swallowed (logger.warning) to keep markdown writes authoritative.
VectorIndexer = Callable[[NoteRef, str], Awaitable[int]]


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
        vector_indexer: Optional[VectorIndexer] = None,
        hooks: Optional[MemoryHooks] = None,
    ) -> None:
        self._layout = layout
        self._tz = tz
        self._scope = scope
        self._lock = LoopAgnosticLock()
        self._cache: Dict[str, Note] = {}
        self._loaded = False
        self._explicit_links: Dict[str, Set[str]] = {}
        self._vector_indexer = vector_indexer
        self._hooks = hooks or MemoryHooks()

    def attach_vector_indexer(self, indexer: Optional[VectorIndexer]) -> None:
        """Wire (or detach) the auto-vector indexing callback.

        Provided so the surrounding `MemoryProvider` can build the
        notes store before the vector store (avoids a circular
        construction order) and then plug the indexer in.
        """
        self._vector_indexer = indexer

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
                metadata=dict(draft.metadata or {}),
                event_id=draft.event_id,
                linked_event_id=draft.linked_event_id,
                kind=draft.kind,
                direction=draft.direction,
                counterpart_id=draft.counterpart_id,
                counterpart_role=draft.counterpart_role,
                session_id=draft.session_id,
            )
            self._write_to_disk(note)
            self._cache[filename] = note
            self._refresh_backlinks()
            indexer = self._vector_indexer
            ref_for_index, body_for_index = note.ref, note.body
            note_meta = note.as_meta()
        # Run auto-vector outside the write lock — embedding the body
        # is an HTTP round-trip and must never block other note ops.
        if indexer is not None and body_for_index:
            await self._safe_index(indexer, ref_for_index, body_for_index)
        await _fire_hook(
            self._hooks.after_note_write,
            "after_note_write",
            note_meta,
        )
        return note_meta

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
            if patch.metadata is not None:
                current.metadata = dict(patch.metadata)
            current.links_out = _extract_links(current.body)
            current.updated_at = now_in(self._tz)

            self._write_to_disk(current)
            self._refresh_backlinks()
            indexer = self._vector_indexer
            ref_for_index, body_for_index = current.ref, current.body
            note_meta = current.as_meta()
        if indexer is not None and body_for_index:
            await self._safe_index(indexer, ref_for_index, body_for_index)
        await _fire_hook(
            self._hooks.after_note_update,
            "after_note_update",
            note_meta,
        )
        return note_meta

    @staticmethod
    async def _safe_index(indexer: VectorIndexer, ref: NoteRef, body: str) -> None:
        """Best-effort indexer call — markdown writes win on any
        embedding failure. Logs at WARNING so misconfigured embedding
        keys surface without breaking note CRUD.
        """
        try:
            await indexer(ref, body)
        except Exception:  # noqa: BLE001
            logger.warning(
                "auto-vector index failed for %s; markdown write retained",
                ref.filename,
                exc_info=True,
            )

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
            # Drop the metadata sidecar alongside the note so a
            # subsequent same-filename write doesn't pick up stale
            # extension metadata.
            try:
                _metadata_sidecar_path(path).unlink(missing_ok=True)
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

    async def load_pinned(
        self,
        *,
        category: str = "critical",
        max_chars: int = 3000,
    ) -> str:
        """Concatenate every note in `category` into a prompt-injectable
        string, ordered by importance (CRITICAL → HIGH → MEDIUM → LOW)
        and then by recency. Stops once the running char total reaches
        `max_chars`.

        Hosts use this for the system prompt's pinned-facts section
        without having to walk `list(category=...)` + `read` + manual
        char-budgeting themselves.
        """
        if max_chars <= 0:
            return ""
        async with self._lock:
            await self._ensure_loaded()
            pool = [
                _clone(note) for note in self._cache.values() if (note.category or "") == category
            ]

        # Importance descending, then most-recently modified first.
        def _key(note: Note) -> Tuple[float, str]:
            modified = note.updated_at.isoformat() if note.updated_at else ""
            return (-note.importance.boost, "" if modified is None else "")  # placeholder

        pool.sort(
            key=lambda n: (
                -n.importance.boost,
                -(n.updated_at.timestamp() if n.updated_at else 0.0),
            )
        )

        parts: List[str] = []
        used = 0
        for note in pool:
            block = note.body.strip()
            if not block:
                continue
            header = f"## {note.title}" if note.title else ""
            piece = f"{header}\n{block}".strip() if header else block
            cost = len(piece) + (2 if parts else 0)  # blank-line separator
            if used + cost > max_chars and parts:
                break
            parts.append(piece)
            used += cost
        return "\n\n".join(parts)

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
        """Recompute `links_in` for every cached note.

        Wikilink syntax is bare (`[[target]]`) — `target` has no `.md`
        suffix — but the cache keys notes by their on-disk filename
        (`target.md`). Without normalisation `link_map[target]` and
        `cache[target.md]` never align and `target.links_in` stays
        empty regardless of how many notes link to it.

        The fix: when a wikilink target lacks `.md`, also probe the
        cache for `<target>.md` so the resolved filename is what the
        cache stores. Hosts that pass full-filename wikilinks
        (`[[target.md]]`) keep working as before.
        """
        link_map: Dict[str, List[str]] = {}
        for fname, note in self._cache.items():
            for tgt in note.links_out:
                resolved = self._resolve_link_target(tgt)
                link_map.setdefault(resolved, []).append(fname)
            for tgt in self._explicit_links.get(fname, ()):
                if tgt in note.links_out:
                    continue
                resolved = self._resolve_link_target(tgt)
                link_map.setdefault(resolved, []).append(fname)
        for fname, note in self._cache.items():
            note.links_in = list(dict.fromkeys(link_map.get(fname, [])))

    def _resolve_link_target(self, target: str) -> str:
        """Map a wikilink target to its cache key.

        Tries the verbatim target first, then `<target>.md`. Falls
        back to the verbatim target so an unresolved link still
        appears in the link map (caller can detect via cache lookup).
        """
        if target in self._cache:
            return target
        with_md = target if target.endswith(".md") else f"{target}.md"
        if with_md in self._cache:
            return with_md
        return target

    def _write_to_disk(self, note: Note) -> None:
        category = note.category or "root"
        dir_path = self._layout.note_dir(category)
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / note.ref.filename
        meta = _note_to_frontmatter(note, tz=self._tz)
        header = frontmatter.dump(meta)
        body = note.body.rstrip() + "\n"
        path.write_text(header + body, encoding="utf-8")
        # Host-extension metadata sidecar — JSON dump alongside the
        # note. Empty dict → remove sidecar so we don't leave dead
        # files behind after a metadata clear.
        sidecar = _metadata_sidecar_path(path)
        if note.metadata:
            import json as _json

            sidecar.write_text(
                _json.dumps(note.metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            try:
                sidecar.unlink(missing_ok=True)
            except OSError:
                pass

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
        # Host-extension metadata is persisted in the sidecar JSON
        # rather than the YAML frontmatter (the hand-rolled parser
        # only handles flat scalar / list values). Read the sidecar if
        # it exists; missing → empty dict.
        host_metadata: Dict[str, Any] = {}
        sidecar = _metadata_sidecar_path(path)
        if sidecar.exists():
            try:
                import json as _json

                decoded = _json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(decoded, dict):
                    host_metadata = decoded
            except (OSError, ValueError):
                host_metadata = {}
        # Lift interaction fields out of frontmatter into typed slots.
        interaction_values: Dict[str, Optional[str]] = {}
        for fname in _INTERACTION_FIELDS:
            raw = meta.get(f"interaction.{fname}")
            if isinstance(raw, str) and raw.strip():
                interaction_values[fname] = raw.strip()
            else:
                interaction_values[fname] = None
        # Strip claimed keys from the carry-through frontmatter dict.
        reserved = {
            "title",
            "tags",
            "category",
            "importance",
            "created",
            "modified",
            "links_to",
            "linked_from",
            "_metadata",
        } | {f"interaction.{f}" for f in _INTERACTION_FIELDS}
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
            frontmatter={k: v for k, v in meta.items() if k not in reserved},
            links_out=[str(t) for t in links_out],
            links_in=[],
            created_at=created,
            updated_at=modified,
            metadata=host_metadata,
            event_id=interaction_values["event_id"],
            linked_event_id=interaction_values["linked_event_id"],
            kind=interaction_values["kind"],
            direction=interaction_values["direction"],
            counterpart_id=interaction_values["counterpart_id"],
            counterpart_role=interaction_values["counterpart_role"],
            session_id=interaction_values["session_id"],
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


_INTERACTION_FIELDS: Tuple[str, ...] = (
    "event_id",
    "linked_event_id",
    "kind",
    "direction",
    "counterpart_id",
    "counterpart_role",
    "session_id",
)


def _note_to_frontmatter(note: Note, *, tz: tzinfo) -> Dict[str, Any]:
    """Produce the YAML-like frontmatter mapping for `note`.

    The host-extension ``note.metadata`` is **not** embedded in the
    YAML — the hand-rolled frontmatter parser only handles flat
    scalar / list values, so a nested business dict would corrupt
    on round-trip. Instead it is persisted to a sidecar JSON file
    alongside the note (see `_metadata_sidecar_path`).

    Interaction fields (event_id / kind / counterpart_* / …) are
    typed first-class on every note dataclass and serialised to
    flat ``interaction.<name>`` keys here so a hand-edited note
    keeps the cross-event references readable in the ``---`` block.
    """
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
    # Promote interaction fields onto frontmatter as `interaction.<name>`.
    for fname in _INTERACTION_FIELDS:
        value = getattr(note, fname, None)
        if value is None:
            continue
        meta[f"interaction.{fname}"] = str(value)
    # Preserve any caller-supplied frontmatter keys not already claimed.
    for k, v in (note.frontmatter or {}).items():
        if k in meta:
            continue
        # Caller's dotted `interaction.*` keys also get rejected if the
        # typed field already wrote them — typed wins.
        meta[k] = v
    return meta


def _metadata_sidecar_path(note_path: Path) -> Path:
    """Return the sidecar JSON path that carries ``note.metadata``.

    Sidecar lives next to the note (`<note>.md.meta.json`) so an
    operator can spot it adjacent to the note in the file panel. The
    file is owned by the executor; hand edits should be rare.
    """
    return note_path.with_suffix(note_path.suffix + ".meta.json")


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
        metadata=dict(note.metadata or {}),
        event_id=note.event_id,
        linked_event_id=note.linked_event_id,
        kind=note.kind,
        direction=note.direction,
        counterpart_id=note.counterpart_id,
        counterpart_role=note.counterpart_role,
        session_id=note.session_id,
    )


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
        logger.debug("memory hook %s raised; skipping", name, exc_info=True)
