"""EphemeralMemoryProvider — in-memory reference implementation.

Purpose:
- Provide the canonical, dependency-free implementation of every
  `MemoryProvider` Protocol method so the rest of the system (stages,
  contract tests, web service) has something to develop against
  before Phase 2 native providers land.
- Demonstrate capability gating: this provider does NOT implement the
  Vector / Curated / Global layers (returns `None` from those handles)
  and the cross-layer `retrieve()` skips them gracefully. The intent
  is to prove the contract degrades cleanly per `MemoryDescriptor`.
- Pass C1 (six-layer retrieval round-trip) for the layers it declares
  as required.

Nothing here touches disk. State lives only as long as the provider
instance does.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.memory.provider import (
    BackendInfo,
    Capability,
    ExecutionSummary,
    Importance,
    Insight,
    Layer,
    MemoryDescriptor,
    MemoryProvider,
    MemorySnapshot,
    Note,
    NoteDraft,
    NoteGraph,
    NoteMeta,
    NotePatch,
    NoteRef,
    NotesHandle,
    RecordReceipt,
    ReflectionContext,
    RetrievalQuery,
    RetrievalResult,
    Scope,
    Turn,
)
from geny_executor.stages.s02_context.types import MemoryChunk


# ─────────────────────────────────────────────────────────────────────
# In-memory layer impls
# ─────────────────────────────────────────────────────────────────────


class _STMStore:
    """Append-only turn buffer with a flat scan search."""

    def __init__(self) -> None:
        self._turns: List[Turn] = []

    async def append(self, turn: Turn) -> None:
        self._turns.append(turn)

    async def recent(self, n: int = 20) -> List[Turn]:
        if n <= 0:
            return []
        return list(self._turns[-n:])

    async def search(self, text: str, *, limit: int = 10) -> List[Turn]:
        needle = text.lower().strip()
        if not needle:
            return []
        out: List[Turn] = []
        for turn in reversed(self._turns):
            haystack = (
                turn.content.lower() if isinstance(turn.content, str) else str(turn.content).lower()
            )
            if needle in haystack:
                out.append(turn)
                if len(out) >= limit:
                    break
        return out

    async def truncate(self, *, keep_last: int) -> int:
        if keep_last < 0 or keep_last >= len(self._turns):
            return 0
        dropped = len(self._turns) - keep_last
        self._turns = self._turns[-keep_last:]
        return dropped

    def all_turns(self) -> List[Turn]:
        return list(self._turns)


class _LTMStore:
    """In-memory long-term memory: main body + dated entries + topics."""

    def __init__(self) -> None:
        self._main: List[str] = []
        self._dated: Dict[str, str] = {}  # YYYY-MM-DD → body
        self._topics: Dict[str, str] = {}  # slug → body

    async def append(self, body: str, *, heading: Optional[str] = None) -> NoteRef:
        chunk = f"## {heading}\n\n{body}\n" if heading else f"{body}\n"
        self._main.append(chunk)
        return NoteRef(filename="MEMORY.md", scope=Scope.SESSION, category="ltm-main")

    async def write_dated(self, body: str, *, day: Optional[datetime] = None) -> NoteRef:
        when = (day or datetime.now(timezone.utc)).date().isoformat()
        prev = self._dated.get(when, "")
        self._dated[when] = (prev + "\n\n" + body).strip() if prev else body
        return NoteRef(filename=f"{when}.md", scope=Scope.SESSION, category="ltm-dated")

    async def write_topic(self, slug: str, body: str) -> NoteRef:
        prev = self._topics.get(slug, "")
        self._topics[slug] = (prev + "\n\n" + body).strip() if prev else body
        return NoteRef(filename=f"topics/{slug}.md", scope=Scope.SESSION, category="ltm-topic")

    async def read_main(self) -> str:
        return "\n".join(self._main).strip()

    async def search(self, text: str, *, limit: int = 5) -> List[MemoryChunk]:
        needle = text.lower().strip()
        if not needle:
            return []
        candidates: List[Tuple[str, str, str]] = []  # (key, body, source)
        if self._main:
            candidates.append(("MEMORY.md", "\n".join(self._main), "long_term"))
        for day, body in self._dated.items():
            candidates.append((f"{day}.md", body, "long_term_dated"))
        for slug, body in self._topics.items():
            candidates.append((f"topics/{slug}.md", body, "long_term_topic"))
        scored: List[Tuple[float, MemoryChunk]] = []
        for key, body, source in candidates:
            hits = body.lower().count(needle)
            if hits == 0:
                continue
            scored.append(
                (
                    float(hits),
                    MemoryChunk(
                        key=key,
                        content=body[:1000],
                        source=source,
                        relevance_score=float(hits),
                    ),
                )
            )
        scored.sort(key=lambda pair: -pair[0])
        return [chunk for _, chunk in scored[:limit]]


class _NotesStore(NotesHandle):
    """Structured notes with frontmatter, tags, importance, wikilinks."""

    _WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

    def __init__(self, *, scope: Scope = Scope.SESSION) -> None:
        self._scope = scope
        self._notes: Dict[str, Note] = {}
        self._explicit_links: Dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _slug(title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9가-힣\-]+", "-", title).strip("-").lower() or "note"
        return slug[:80]

    def _resolve_filename(self, draft: NoteDraft) -> str:
        if draft.filename:
            return draft.filename
        base = self._slug(draft.title) + ".md"
        if base not in self._notes:
            return base
        return f"{self._slug(draft.title)}-{uuid.uuid4().hex[:6]}.md"

    def _extract_links_out(self, body: str) -> List[str]:
        return list(dict.fromkeys(self._WIKILINK.findall(body)))

    def _refresh_backlinks(self) -> None:
        link_map: Dict[str, List[str]] = defaultdict(list)
        for fname, note in self._notes.items():
            for target in note.links_out:
                link_map[target].append(fname)
            for target in self._explicit_links.get(fname, ()):
                if target not in note.links_out:
                    link_map[target].append(fname)
        for fname, note in self._notes.items():
            note.links_in = list(dict.fromkeys(link_map.get(fname, [])))

    async def list(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        importance: Optional[Importance] = None,
    ) -> List[NoteMeta]:
        out: List[NoteMeta] = []
        for note in self._notes.values():
            if category is not None and note.category != category:
                continue
            if tag is not None and tag not in note.tags:
                continue
            if importance is not None and note.importance != importance:
                continue
            out.append(note.as_meta())
        return out

    async def read(self, filename: str) -> Optional[Note]:
        return self._notes.get(filename)

    async def write(self, draft: NoteDraft) -> NoteMeta:
        filename = self._resolve_filename(draft)
        now = datetime.now(timezone.utc)
        note = Note(
            ref=NoteRef(filename=filename, scope=self._scope, category=draft.category),
            title=draft.title,
            body=draft.body,
            importance=draft.importance,
            tags=list(draft.tags),
            category=draft.category,
            frontmatter=dict(draft.frontmatter),
            links_out=self._extract_links_out(draft.body),
            created_at=now,
            updated_at=now,
            metadata=dict(draft.metadata or {}),
        )
        self._notes[filename] = note
        self._refresh_backlinks()
        return note.as_meta()

    async def update(self, filename: str, patch: NotePatch) -> NoteMeta:
        note = self._notes.get(filename)
        if note is None:
            raise KeyError(f"note {filename!r} not found")
        if patch.title is not None:
            note.title = patch.title
        if patch.append_body is not None:
            note.body = (note.body + "\n\n" + patch.append_body).strip()
        elif patch.body is not None:
            note.body = patch.body
        if patch.importance is not None:
            note.importance = patch.importance
        if patch.tags is not None:
            note.tags = list(patch.tags)
        if patch.category is not None:
            note.category = patch.category
        if patch.frontmatter is not None:
            note.frontmatter = dict(patch.frontmatter)
        if patch.metadata is not None:
            note.metadata = dict(patch.metadata)
        note.links_out = self._extract_links_out(note.body)
        note.updated_at = datetime.now(timezone.utc)
        self._refresh_backlinks()
        return note.as_meta()

    async def delete(self, filename: str) -> bool:
        existed = filename in self._notes
        self._notes.pop(filename, None)
        self._explicit_links.pop(filename, None)
        for targets in self._explicit_links.values():
            targets.discard(filename)
        self._refresh_backlinks()
        return existed

    async def link(self, source: str, target: str) -> bool:
        if source not in self._notes:
            raise KeyError(f"source note {source!r} not found")
        self._explicit_links[source].add(target)
        self._refresh_backlinks()
        return True

    async def graph(self) -> NoteGraph:
        nodes = [n.as_meta() for n in self._notes.values()]
        edges: List[Tuple[str, str]] = []
        for fname, note in self._notes.items():
            for tgt in note.links_out:
                edges.append((fname, tgt))
            for tgt in self._explicit_links.get(fname, ()):
                if tgt not in note.links_out:
                    edges.append((fname, tgt))
        return NoteGraph(nodes=nodes, edges=edges)

    async def search(
        self, text: str, *, limit: int = 5, importance_floor: Importance = Importance.LOW
    ) -> List[MemoryChunk]:
        needle = text.lower().strip()
        if not needle:
            return []
        floor = importance_floor.boost
        keywords = [t for t in re.split(r"\s+", needle) if t]
        keyword_set = set(keywords)
        scored: List[Tuple[float, MemoryChunk]] = []
        for fname, note in self._notes.items():
            if note.importance.boost < floor:
                continue
            haystack = (note.title + "\n" + note.body).lower()
            keyword_hits = sum(haystack.count(k) for k in keywords)
            if keyword_hits == 0 and not keyword_set.intersection({t.lower() for t in note.tags}):
                continue
            tag_overlap = len(keyword_set.intersection({t.lower() for t in note.tags}))
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

    # used by snapshot/restore
    def all(self) -> List[Note]:
        return list(self._notes.values())

    def replace_all(self, notes: Iterable[Note]) -> None:
        self._notes.clear()
        self._explicit_links.clear()
        for n in notes:
            self._notes[n.ref.filename] = n
        self._refresh_backlinks()


class _IndexCache:
    """Derived index handle. Computes everything on demand."""

    def __init__(self, notes: _NotesStore) -> None:
        self._notes = notes

    async def snapshot(self) -> Dict[str, Any]:
        files = [n.ref.filename for n in self._notes.all()]
        return {
            "files": files,
            "count": len(files),
            "tags": await self.tag_counts(),
        }

    async def tag_counts(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for n in self._notes.all():
            counter.update(n.tags)
        return dict(counter)

    async def graph(self) -> NoteGraph:
        return await self._notes.graph()

    async def rebuild(self) -> None:
        self._notes._refresh_backlinks()  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────


class EphemeralMemoryProvider(MemoryProvider):
    """In-memory `MemoryProvider`. Vector / Curated / Global return `None`.

    Construction is dependency-free; safe to instantiate from a unit
    test fixture or as a default for sessions that must not persist
    (e.g. anonymous probes).
    """

    NAME = "ephemeral"
    VERSION = "1.0.0"

    def __init__(self, *, scope: Scope = Scope.SESSION) -> None:
        self._scope = scope
        self._stm = _STMStore()
        self._ltm = _LTMStore()
        self._notes = _NotesStore(scope=scope)
        self._index = _IndexCache(self._notes)
        self._closed = False
        self._descriptor = MemoryDescriptor(
            name=self.NAME,
            version=self.VERSION,
            layers={Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX},
            capabilities={
                Capability.READ,
                Capability.WRITE,
                Capability.SEARCH,
                Capability.LINK,
                Capability.SNAPSHOT,
            },
            backends=[
                BackendInfo(layer=Layer.STM, backend="memory"),
                BackendInfo(layer=Layer.LTM, backend="memory"),
                BackendInfo(layer=Layer.NOTES, backend="memory"),
                BackendInfo(layer=Layer.INDEX, backend="memory"),
            ],
            scope=scope,
            config_schema=_ephemeral_config_schema(),
            description=(
                "In-memory reference provider. STM/LTM/Notes/Index only; "
                "vector/curated/global intentionally absent."
            ),
        )

    @property
    def descriptor(self) -> MemoryDescriptor:
        return self._descriptor

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        self._closed = True

    # ── Layer handles ───────────────────────────────────────────────
    def stm(self) -> _STMStore:
        return self._stm

    def ltm(self) -> _LTMStore:
        return self._ltm

    def notes(self) -> NotesHandle:
        return self._notes

    def vector(self) -> None:
        return None

    def curated(self) -> None:
        return None

    def global_(self) -> None:
        return None

    def index(self) -> _IndexCache:
        return self._index

    # ── Cross-layer ─────────────────────────────────────────────────
    async def record_turn(self, turn: Turn) -> None:
        await self._stm.append(turn)

    async def record_execution(self, summary: ExecutionSummary) -> RecordReceipt:
        files: List[str] = []
        receipt = RecordReceipt()

        if summary.final_text:
            ref_dated = await self._ltm.write_dated(
                f"## Q\n{summary.user_input}\n\n## A\n{summary.final_text}",
            )
            files.append(ref_dated.filename)

            draft = NoteDraft(
                title=(summary.user_input or "execution")[:80],
                body=summary.final_text,
                importance=Importance.MEDIUM,
                tags=list(summary.tags),
                category="reflection",
                scope=self._scope,
            )
            meta = await self._notes.write(draft)
            files.append(meta.ref.filename)
            receipt.notes_written = 1

        receipt.files_updated = files
        return receipt

    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]:
        # Ephemeral provider has no LLM; concrete providers override.
        # Returns empty by default — reflection is opt-in.
        return ()

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        chunks: List[MemoryChunk] = []
        breakdown: Dict[Layer, int] = {}

        if Layer.STM in query.layers:
            recent = await self._stm.recent(n=query.max_per_layer)
            recent_chunks = [
                MemoryChunk(
                    key=f"stm-{i}",
                    content=_turn_to_text(t),
                    source="recent_message",
                    relevance_score=0.0,
                )
                for i, t in enumerate(recent)
            ]
            chunks.extend(recent_chunks)
            breakdown[Layer.STM] = len(recent_chunks)

        if Layer.LTM in query.layers:
            main_text = await self._ltm.read_main()
            ltm_chunks: List[MemoryChunk] = []
            if main_text:
                ltm_chunks.append(
                    MemoryChunk(
                        key="MEMORY.md",
                        content=main_text,
                        source="long_term",
                        relevance_score=1.0,
                    )
                )
            if query.text:
                ltm_chunks.extend(await self._ltm.search(query.text, limit=query.max_per_layer))
            chunks.extend(ltm_chunks)
            breakdown[Layer.LTM] = len(ltm_chunks)

        if Layer.NOTES in query.layers and query.text:
            note_chunks = await self._notes.search(
                query.text,
                limit=query.max_per_layer,
                importance_floor=query.importance_floor,
            )
            chunks.extend(note_chunks)
            breakdown[Layer.NOTES] = len(note_chunks)

        # Trim to char budget while preserving order
        kept: List[MemoryChunk] = []
        used = 0
        for c in chunks:
            cost = len(c.content)
            if used + cost > query.max_chars and kept:
                break
            kept.append(c)
            used += cost

        return RetrievalResult(
            chunks=kept,
            layer_breakdown=breakdown,
            total_chars=used,
        )

    async def snapshot(self) -> MemorySnapshot:
        payload = {
            "scope": self._scope.value,
            "stm": [_turn_to_dict(t) for t in self._stm.all_turns()],
            "ltm_main": await self._ltm.read_main(),
            "ltm_dated": dict(self._ltm._dated),  # type: ignore[attr-defined]
            "ltm_topics": dict(self._ltm._topics),  # type: ignore[attr-defined]
            "notes": [_note_to_dict(n) for n in self._notes.all()],
        }
        import json

        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return MemorySnapshot(
            provider=self.NAME,
            version=self.VERSION,
            layers=[Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX],
            payload=payload,
            size_bytes=len(blob),
            checksum=hashlib.sha256(blob).hexdigest(),
        )

    async def restore(self, snap: MemorySnapshot) -> None:
        if snap.provider != self.NAME:
            raise ValueError(f"snapshot from {snap.provider!r} cannot restore into {self.NAME!r}")
        payload = snap.payload
        # Reset
        self._stm = _STMStore()
        self._ltm = _LTMStore()
        self._notes = _NotesStore(scope=self._scope)
        self._index = _IndexCache(self._notes)

        for d in payload.get("stm", []):
            await self._stm.append(_turn_from_dict(d))
        if payload.get("ltm_main"):
            await self._ltm.append(payload["ltm_main"])
        for day, body in payload.get("ltm_dated", {}).items():
            self._ltm._dated[day] = body  # type: ignore[attr-defined]
        for slug, body in payload.get("ltm_topics", {}).items():
            self._ltm._topics[slug] = body  # type: ignore[attr-defined]
        notes = [_note_from_dict(d, default_scope=self._scope) for d in payload.get("notes", [])]
        self._notes.replace_all(notes)

    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef:
        # Ephemeral has no curated/global plane; promotion is a no-op
        # rename. Concrete providers replicate to a wider scope.
        if to == ref.scope:
            return ref
        note = await self._notes.read(ref.filename)
        if note is None:
            raise KeyError(f"cannot promote: {ref.filename!r} not found")
        note.ref = ref.with_scope(to)
        return note.ref


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _ephemeral_config_schema() -> ConfigSchema:
    """Minimal config schema. Real providers expand this to cover all
    21 R-F fields; ephemeral exposes the master switch only.
    """
    return ConfigSchema(
        name="ephemeral_memory",
        fields=[
            ConfigField(
                name="master_enabled",
                type="boolean",
                label="Memory enabled",
                default=True,
            ),
        ],
    )


def _turn_to_text(turn: Turn) -> str:
    if isinstance(turn.content, str):
        return f"[{turn.role}] {turn.content}"
    return f"[{turn.role}] {turn.content!r}"


def _turn_to_dict(turn: Turn) -> Dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        "timestamp": turn.timestamp.isoformat(),
        "metadata": dict(turn.metadata),
    }


def _turn_from_dict(d: Dict[str, Any]) -> Turn:
    ts = d.get("timestamp")
    if isinstance(ts, str):
        try:
            stamp = datetime.fromisoformat(ts)
        except ValueError:
            stamp = datetime.now(timezone.utc)
    else:
        stamp = datetime.now(timezone.utc)
    return Turn(
        role=str(d.get("role", "user")),
        content=d.get("content", ""),
        timestamp=stamp,
        metadata=dict(d.get("metadata", {})),
    )


def _note_to_dict(note: Note) -> Dict[str, Any]:
    return {
        "filename": note.ref.filename,
        "scope": note.ref.scope.value,
        "category": note.category,
        "title": note.title,
        "body": note.body,
        "importance": note.importance.value,
        "tags": list(note.tags),
        "frontmatter": dict(note.frontmatter),
        "links_out": list(note.links_out),
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
        "metadata": dict(note.metadata),
    }


def _note_from_dict(d: Dict[str, Any], *, default_scope: Scope) -> Note:
    def _parse_dt(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    return Note(
        ref=NoteRef(
            filename=d["filename"],
            scope=Scope(d.get("scope", default_scope.value)),
            category=d.get("category"),
        ),
        title=d.get("title", ""),
        body=d.get("body", ""),
        importance=Importance(d.get("importance", Importance.MEDIUM.value)),
        tags=list(d.get("tags", [])),
        category=d.get("category"),
        frontmatter=dict(d.get("frontmatter", {})),
        links_out=list(d.get("links_out", [])),
        links_in=[],
        created_at=_parse_dt(d.get("created_at")),
        updated_at=_parse_dt(d.get("updated_at")),
        metadata=dict(d.get("metadata", {})),
    )
