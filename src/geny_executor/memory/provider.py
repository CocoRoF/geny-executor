"""MemoryProvider — unified memory contract for geny-executor.

This module is the **public memory interface** for the executor. Every
concrete memory implementation (`EphemeralMemoryProvider`,
`FileMemoryProvider`, `SQLMemoryProvider`, `CompositeMemoryProvider`,
the validation-only `GenyManagerAdapter`, and any future provider)
satisfies the `MemoryProvider` Protocol defined here.

Architectural background lives in
`geny-executor-web/docs/MEMORY_ARCHITECTURE.md` (§3) and the frozen
contract in `docs/MEMORY_SPEC.yaml`. The 4-axis model is:

    Layer × Capability × Backend × Scope

Each method on a handle (`STMHandle`, `LTMHandle`, `NotesHandle`,
`VectorHandle`, `CuratedHandle`, `GlobalHandle`, `IndexHandle`) maps
to one (Layer, Capability) pair from the spec's `requirements`
catalogue. Handles for optional layers may legitimately return `None`
from the provider; callers must capability-gate.

The Protocols are `@runtime_checkable` so existing adapters can be
wrapped progressively.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from geny_executor.core.schema import ConfigSchema
    from geny_executor.core.state import PipelineState
    from geny_executor.stages.s02_context.types import MemoryChunk


# ─────────────────────────────────────────────────────────────────────
# 1. Enums (4-axis model)
# ─────────────────────────────────────────────────────────────────────


class Layer(str, enum.Enum):
    """Storage plane. Mirrors `MEMORY_SPEC.yaml::layers`."""

    STM = "stm"
    LTM = "ltm"
    NOTES = "notes"
    VECTOR = "vector"
    INDEX = "index"
    CURATED = "curated"
    GLOBAL = "global"


class Capability(str, enum.Enum):
    """Operation kind, orthogonal to layer.
    Mirrors `MEMORY_SPEC.yaml::capabilities`.
    """

    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    LINK = "link"
    PROMOTE = "promote"
    REINDEX = "reindex"
    SNAPSHOT = "snapshot"
    REFLECT = "reflect"
    SUMMARIZE = "summarize"


class Scope(str, enum.Enum):
    """Multi-tenancy / visibility axis."""

    EPHEMERAL = "ephemeral"
    SESSION = "session"
    USER = "user"
    TENANT = "tenant"
    GLOBAL = "global"


class Importance(str, enum.Enum):
    """Note importance grade. Drives retrieval boost
    (`MEMORY_SPEC.yaml::retrieval.importance_boosts`)."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def boost(self) -> float:
        return {
            Importance.CRITICAL: 2.0,
            Importance.HIGH: 1.5,
            Importance.MEDIUM: 1.0,
            Importance.LOW: 0.5,
        }[self]


# ─────────────────────────────────────────────────────────────────────
# 2. Identity / value types
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoteRef:
    """Stable handle to a note across providers. Filename + scope are
    the canonical identity key; `category` is advisory and may be
    derived from filename layout by the provider.
    """

    filename: str
    scope: Scope = Scope.SESSION
    category: Optional[str] = None
    backend: Optional[str] = None  # e.g. "filesystem", "postgres" — for diagnostics

    def with_scope(self, scope: Scope) -> "NoteRef":
        return NoteRef(
            filename=self.filename, scope=scope, category=self.category, backend=self.backend
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "scope": self.scope.value,
            "category": self.category,
            "backend": self.backend,
        }


@dataclass
class BackendInfo:
    """Describes which physical backend serves a layer in a provider."""

    layer: Layer
    backend: str  # e.g. "filesystem", "sqlite", "postgres", "faiss"
    location: str = ""  # path / DSN / URL — opaque to consumers
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingDescriptor:
    """Active embedding model snapshot. Used by C5 compatibility checks."""

    provider: str  # "openai" | "voyage" | "google" | "local"
    model: str
    dimension: int
    metric: str = "cosine"
    api_key_present: bool = False  # never leak the key itself

    def matches(self, other: "EmbeddingDescriptor") -> bool:
        return (
            self.provider == other.provider
            and self.model == other.model
            and self.dimension == other.dimension
            and self.metric == other.metric
        )


@dataclass
class CostEvent:
    """Structured cost event. Emitted by reflect/embed/summarize paths.
    Schema mirrors `MEMORY_SPEC.yaml::events::memory.cost`.
    """

    kind: str  # "embedding" | "reflection" | "summary"
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "usd": self.usd,
            "model": self.model,
            "metadata": dict(self.metadata),
        }


@dataclass
class CostModel:
    """Per-1k-token unit cost for a provider's model invocations.
    Concrete providers may extend with per-call surcharge etc.
    """

    embedding_usd_per_1k_tokens: float = 0.0
    reflection_usd_per_1k_in: float = 0.0
    reflection_usd_per_1k_out: float = 0.0
    summary_usd_per_1k_in: float = 0.0
    summary_usd_per_1k_out: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# 3. Note dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class InteractionFields:
    """Optional **first-class** interaction fields shared by every
    note-shaped dataclass (``NoteDraft``, ``Note``, ``NoteMeta``)
    and ``Turn``.

    The decision (D4): event_id / linked_event_id / kind / direction
    / counterpart_id / counterpart_role / session_id are common
    enough to be promoted out of the host-extension ``metadata``
    dict into typed optional fields. Hosts that don't use them set
    them to ``None`` and pay nothing; hosts that do (Geny's
    InteractionEvent stream, web-mirror dashboards, anyone wiring
    cross-event references) get a typed surface and frontmatter
    serialisation for free.

    Truly host-specific keys (Geny's bucket router, VTuber LOGS
    routing hints, etc.) stay on the ``metadata`` dict with a
    ``geny.*`` prefix per the cross-cutting convention.
    """

    event_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    kind: Optional[str] = None  # e.g. "user_chat", "agent_dm", "tool_run_summary"
    direction: Optional[str] = None  # "inbound" | "outbound"
    counterpart_id: Optional[str] = None
    counterpart_role: Optional[str] = None  # "user" | "agent"
    session_id: Optional[str] = None  # original session that produced the artefact


@dataclass
class NoteMeta:
    """Lightweight projection of a note for list/graph operations.

    `metadata` is a host-defined extension channel — providers store
    and round-trip it verbatim, never inspect the contents. Use a
    namespaced key prefix (e.g. ``geny.*``) to avoid collisions.

    The interaction fields (``event_id``, ``linked_event_id``,
    ``kind``, ``direction``, ``counterpart_id``, ``counterpart_role``,
    ``session_id``) are first-class and serialised to frontmatter
    when present. See ``InteractionFields``.
    """

    ref: NoteRef
    title: str = ""
    importance: Importance = Importance.MEDIUM
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    size_bytes: int = 0
    backlinks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Interaction fields (D4 — promoted from metadata to typed surface)
    event_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    kind: Optional[str] = None
    direction: Optional[str] = None
    counterpart_id: Optional[str] = None
    counterpart_role: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class Note:
    """Full note: frontmatter + body + metadata.

    `frontmatter` is the disk YAML payload (serialised to the .md
    header). `metadata` is the ephemeral host-extension channel —
    not persisted to the YAML, but round-tripped through writes
    and read responses for routing / business hints.

    Interaction fields (``event_id``, ``linked_event_id``, etc.) are
    first-class and serialised to frontmatter under the
    ``interaction.*`` namespace.
    """

    ref: NoteRef
    title: str
    body: str
    importance: Importance = Importance.MEDIUM
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    links_out: List[str] = field(default_factory=list)
    links_in: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Interaction fields (D4)
    event_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    kind: Optional[str] = None
    direction: Optional[str] = None
    counterpart_id: Optional[str] = None
    counterpart_role: Optional[str] = None
    session_id: Optional[str] = None

    def as_meta(self) -> NoteMeta:
        return NoteMeta(
            ref=self.ref,
            title=self.title,
            importance=self.importance,
            tags=list(self.tags),
            category=self.category,
            created_at=self.created_at,
            updated_at=self.updated_at,
            size_bytes=len(self.body.encode("utf-8")),
            backlinks=len(self.links_in),
            metadata=dict(self.metadata),
            event_id=self.event_id,
            linked_event_id=self.linked_event_id,
            kind=self.kind,
            direction=self.direction,
            counterpart_id=self.counterpart_id,
            counterpart_role=self.counterpart_role,
            session_id=self.session_id,
        )


@dataclass
class NoteDraft:
    """Input payload for `NotesHandle.write`. Provider assigns the
    canonical filename if `filename` is empty.

    `frontmatter` is the disk YAML payload. `metadata` is the
    ephemeral host-extension channel — not serialised, but
    available to `MemoryHooks` and downstream callers for business
    routing.

    Interaction fields are first-class — providers serialise them
    to ``interaction.*`` frontmatter keys when present.
    """

    title: str
    body: str
    importance: Importance = Importance.MEDIUM
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    filename: str = ""  # empty → provider-generated slug
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    scope: Scope = Scope.SESSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Interaction fields (D4)
    event_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    kind: Optional[str] = None
    direction: Optional[str] = None
    counterpart_id: Optional[str] = None
    counterpart_role: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class NotePatch:
    """Partial update payload. Unset fields are left untouched.

    `metadata` patches replace the existing extension dict (same
    semantics as `frontmatter`). Pass an empty dict to clear, or
    omit (None) to leave untouched.
    """

    title: Optional[str] = None
    body: Optional[str] = None
    importance: Optional[Importance] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    frontmatter: Optional[Dict[str, Any]] = None
    append_body: Optional[str] = None  # convenience: append to existing body
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NoteSummary:
    """Lightweight per-note summary for progressive-disclosure listings.

    Returned by ``IndexHandle.list_notes`` — hosts use this to render
    a category folder view (filename + title + first paragraph + tag
    chips + size + modified) without parsing every note's body.
    """

    filename: str
    title: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    importance: str = "medium"
    char_count: int = 0
    modified: str = ""
    first_paragraph: str = ""


@dataclass
class OutlineNode:
    """One heading in a markdown outline tree.

    ``level`` is the markdown heading depth (1 → ``#``, 2 → ``##``, …).
    ``line_start`` / ``line_end`` are 1-indexed line numbers in the
    note body that bound this section's content (between this heading
    and the next heading at the same or shallower level). Children
    are nested headings of strictly greater depth.
    """

    level: int
    heading: str
    line_start: int
    line_end: int
    children: List["OutlineNode"] = field(default_factory=list)


@dataclass
class NoteOutline:
    """Markdown outline of a single note.

    Hosts call ``IndexHandle.read_outline(filename)`` after a
    ``list_notes`` selection to see the heading tree, then
    ``read_section(filename, heading)`` for the body of a chosen
    heading. This is the third step of the progressive-disclosure
    chain (categories → notes → outline → section).
    """

    filename: str
    title: str = ""
    headings: List[OutlineNode] = field(default_factory=list)


@dataclass
class NoteGraph:
    """Wikilink graph snapshot. Edge: (source_filename → target_filename).

    `metadata` carries optional host-side annotations (graph build
    timestamp, derived stats, etc.).

    Query helpers (1-hop / k-hop / connected-component / linked-chain
    / notes-with-tag) operate on the in-memory snapshot — hosts that
    need fresh results re-snapshot via ``IndexHandle.graph()``.
    """

    nodes: List[NoteMeta] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def neighbours(self, filename: str) -> List[str]:
        return [b for a, b in self.edges if a == filename]

    def k_hop(self, filename: str, k: int) -> List[str]:
        """Every node reachable from ``filename`` in **at most** ``k``
        hops, excluding ``filename`` itself. ``k=0`` returns ``[]``,
        ``k=1`` is equivalent to ``neighbours``. Order: BFS level,
        deterministic by edge order.
        """
        if k <= 0 or not filename:
            return []
        adj = self._adjacency()
        seen: Set[str] = {filename}
        frontier: List[str] = [filename]
        out: List[str] = []
        for _ in range(k):
            next_frontier: List[str] = []
            for node in frontier:
                for nbr in adj.get(node, []):
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    out.append(nbr)
                    next_frontier.append(nbr)
            if not next_frontier:
                break
            frontier = next_frontier
        return out

    def connected_component(self, filename: str) -> Set[str]:
        """Closure under the directed edge relation (treated as
        undirected) starting from ``filename``. Includes the seed
        node itself when it appears anywhere in the graph.
        """
        if not filename:
            return set()
        # Build undirected adjacency once per call.
        adj_und: Dict[str, Set[str]] = {}
        for src, tgt in self.edges:
            adj_und.setdefault(src, set()).add(tgt)
            adj_und.setdefault(tgt, set()).add(src)
        if filename not in adj_und:
            # Lone node — empty graph component unless it's a node
            # known to the graph.
            known = {n.ref.filename for n in self.nodes}
            if filename in known:
                return {filename}
            return set()
        seen: Set[str] = {filename}
        stack: List[str] = [filename]
        while stack:
            node = stack.pop()
            for nbr in adj_und.get(node, ()):
                if nbr in seen:
                    continue
                seen.add(nbr)
                stack.append(nbr)
        return seen

    def linked_chain(self, start: str, end: str) -> Optional[List[str]]:
        """Shortest directed path from ``start`` to ``end`` (BFS).
        Returns ``None`` if no path exists; ``[start]`` when
        ``start == end``.
        """
        if not start or not end:
            return None
        if start == end:
            return [start]
        adj = self._adjacency()
        prev: Dict[str, str] = {}
        seen: Set[str] = {start}
        frontier: List[str] = [start]
        while frontier:
            next_frontier: List[str] = []
            for node in frontier:
                for nbr in adj.get(node, []):
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    prev[nbr] = node
                    if nbr == end:
                        # Reconstruct path.
                        path = [end]
                        cur = end
                        while cur != start:
                            cur = prev[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    next_frontier.append(nbr)
            frontier = next_frontier
        return None

    def notes_with_tag(self, tag: str) -> List[str]:
        """Filenames of every node whose ``tags`` (case-insensitive)
        contains ``tag``. Empty when no node carries the tag.
        """
        if not tag:
            return []
        needle = tag.lower()
        out: List[str] = []
        for n in self.nodes:
            tags = getattr(n, "tags", None) or ()
            if any(str(t).lower() == needle for t in tags):
                out.append(n.ref.filename)
        return out

    def _adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for src, tgt in self.edges:
            adj.setdefault(src, []).append(tgt)
        return adj


# ─────────────────────────────────────────────────────────────────────
# 4. Turn / Reflection / Execution
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """One conversational turn for STM recording.

    Interaction fields (``event_id``, ``linked_event_id``, etc.) are
    first-class — STM stores serialise them onto the jsonl row so
    cross-event references survive read-back. Truly host-specific
    routing hints stay on ``metadata``.
    """

    role: str  # "user" | "assistant" | "system" | "tool"
    content: Any  # str or Anthropic structured content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Interaction fields (D4 — promoted from metadata to typed surface)
    event_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    kind: Optional[str] = None
    direction: Optional[str] = None
    counterpart_id: Optional[str] = None
    counterpart_role: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def bytes(self) -> int:
        if isinstance(self.content, str):
            return len(self.content.encode("utf-8"))
        try:
            import json

            return len(json.dumps(self.content, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError):
            return len(repr(self.content).encode("utf-8"))

    @classmethod
    def from_state_message(cls, message: Mapping[str, Any]) -> "Turn":
        """Lift a `state.messages` entry into a `Turn`.

        Honours an optional ``message["metadata"]`` dict. Hosts that
        stamp interaction fields onto a message before stage 18 may
        either:

        - Use the typed keys directly on the message (``event_id``,
          ``linked_event_id``, ``kind``, …) — promoted to the matching
          ``Turn`` field.
        - Or place them inside ``metadata["interaction"]`` /
          ``metadata["geny.interaction.*"]`` — preserved on
          ``Turn.metadata`` for downstream routing.

        Either form survives the round-trip; STM stores serialise
        the typed surface to dedicated jsonl columns so cross-event
        traversal stays first-class.
        """
        raw_meta = message.get("metadata") or {}
        meta_dict: Dict[str, Any]
        if isinstance(raw_meta, Mapping):
            meta_dict = dict(raw_meta)
        else:
            meta_dict = {}

        def _take(name: str) -> Optional[str]:
            # Prefer top-level key on the message, then a top-level
            # key in metadata, then `metadata['interaction'][name]`.
            v = message.get(name)
            if isinstance(v, str) and v:
                return v
            v = meta_dict.get(name)
            if isinstance(v, str) and v:
                return v
            inter = meta_dict.get("interaction")
            if isinstance(inter, Mapping):
                v = inter.get(name)
                if isinstance(v, str) and v:
                    return v
            return None

        return cls(
            role=str(message.get("role", "user")),
            content=message.get("content", ""),
            metadata=meta_dict,
            event_id=_take("event_id"),
            linked_event_id=_take("linked_event_id"),
            kind=_take("kind"),
            direction=_take("direction"),
            counterpart_id=_take("counterpart_id"),
            counterpart_role=_take("counterpart_role"),
            session_id=_take("session_id"),
        )


@dataclass
class ExecutionSummary:
    """Snapshot of a completed execution, fed to `record_execution`."""

    session_id: str
    user_input: str
    final_text: str
    iterations: int = 1
    duration_ms: int = 0
    completion_signal: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    turns: List[Turn] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: "PipelineState", *, user_input: str = "") -> "ExecutionSummary":
        return cls(
            session_id=state.session_id,
            user_input=user_input or _extract_first_user_text(state),
            final_text=state.final_text,
            iterations=state.iteration,
            completion_signal=state.completion_signal,
            metadata=dict(state.metadata),
            turns=[Turn.from_state_message(m) for m in state.messages],
        )


@dataclass
class RecordReceipt:
    """Result of `record_execution`. Drives the
    `memory.execution_recorded` event payload.

    `metadata` is the host-extension channel for downstream hooks
    (e.g. emit-event payload customisation, business audit fields).
    """

    notes_written: int = 0
    vector_chunks: int = 0
    files_updated: List[str] = field(default_factory=list)
    cost: Optional[CostEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> Dict[str, Any]:
        return {
            "notes_written": self.notes_written,
            "vector_chunks": self.vector_chunks,
            "files_updated": list(self.files_updated),
            "cost": self.cost.to_dict() if self.cost else None,
        }


@dataclass
class Insight:
    """LLM-extracted memory insight. Auto-promotion checks
    `importance >= HIGH`.

    `metadata` carries host extension (reflection prompt fingerprint,
    confidence score, business labels) — never inspected by the
    executor.
    """

    title: str
    content: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    importance: Importance = Importance.MEDIUM
    ref: Optional[NoteRef] = None  # filled in once promoted
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_auto_promote(self) -> bool:
        return self.importance in (Importance.HIGH, Importance.CRITICAL)

    def to_event(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "importance": self.importance.value,
            "category": self.category,
            "tags": list(self.tags),
        }


@dataclass
class ReflectionContext:
    """Input bundle for `MemoryProvider.reflect`.

    `metadata` is the host-extension channel for business hints
    (current speaker persona, session phase, etc.). Replaces the
    legacy ``extra`` field; the rename is intentional so every
    stage I/O dataclass shares the ``metadata`` name.
    """

    session_id: str
    recent_turns: List[Turn] = field(default_factory=list)
    user_focus: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: "PipelineState", *, focus: str = "") -> "ReflectionContext":
        return cls(
            session_id=state.session_id,
            recent_turns=[Turn.from_state_message(m) for m in state.messages[-10:]],
            user_focus=focus or _extract_first_user_text(state),
        )


# ─────────────────────────────────────────────────────────────────────
# 5. Retrieval
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalQuery:
    """Cross-layer query. Mirrors §5 Appendix A."""

    text: str
    layers: Set[Layer] = field(
        default_factory=lambda: {Layer.STM, Layer.LTM, Layer.NOTES, Layer.VECTOR}
    )
    max_chars: int = 8000
    max_per_layer: int = 5
    importance_floor: Importance = Importance.LOW
    tag_filter: Set[str] = field(default_factory=set)
    use_llm_gate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: "PipelineState") -> "RetrievalQuery":
        return cls(text=_extract_first_user_text(state))


@dataclass
class RetrievalResult:
    """Result of a cross-layer query.

    `metadata` is the host-extension channel for business stats
    (e.g. counterpart filter applied, retrieval latency breakdown).
    """

    chunks: List[MemoryChunk] = field(default_factory=list)
    layer_breakdown: Dict[Layer, int] = field(default_factory=dict)
    total_chars: int = 0
    cost: Optional[CostEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_prompt_block(self) -> str:
        """Render chunks into the single string the system prompt
        builder injects under `<memory>` — providers may override.
        """
        if not self.chunks:
            return ""
        parts: List[str] = []
        for c in self.chunks:
            tag = c.source or "memory"
            parts.append(f'<{tag} score="{c.relevance_score:.3f}">\n{c.content}\n</{tag}>')
        return "\n".join(parts)

    def to_event(self) -> Dict[str, Any]:
        return {
            "chunks": [
                {
                    "key": c.key,
                    "source": c.source,
                    "score": c.relevance_score,
                }
                for c in self.chunks
            ],
            "total_chars": self.total_chars,
            "layer_breakdown": {layer.value: n for layer, n in self.layer_breakdown.items()},
        }


# ─────────────────────────────────────────────────────────────────────
# 6. Snapshot / migration
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MemorySnapshot:
    """Portable export. `payload` is provider-specific (filesystem
    path, tarball bytes, JSON document, ...). Always carry a checksum
    so callers can verify integrity.

    `metadata` carries host-extension fields (originating cycle id,
    retention policy, business audit) — never inspected by the
    executor.
    """

    provider: str
    version: str
    layers: List[Layer]
    payload: Any
    size_bytes: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> Dict[str, Any]:
        return {
            "size_bytes": self.size_bytes,
            "layers": [layer.value for layer in self.layers],
            "checksum": self.checksum,
        }


@dataclass
class ReindexPlan:
    """Output of `MemoryDescriptor.compatibility_check` / a vector
    handle's plan-mode reindex. Surfaces enough info for a UI confirm
    dialog (C5).
    """

    layer: Layer
    reason: str
    chunks_to_reindex: int = 0
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_duration_ms: int = 0
    requires_explicit_approval: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# 7. MemoryDescriptor — provider self-description
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MemoryDescriptor:
    """Every provider returns one of these from its `descriptor`
    property. The descriptor is the synthesis material for both:
        - executor-side capability gating (`provider.vector() is None`)
        - executor-web's auto-generated UI (config form, layer cards)
    """

    name: str
    version: str
    layers: Set[Layer]
    capabilities: Set[Capability]
    backends: List[BackendInfo]
    scope: Scope = Scope.SESSION
    config_schema: Optional["ConfigSchema"] = None
    cost_model: Optional[CostModel] = None
    embedding: Optional[EmbeddingDescriptor] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_layer(self, layer: Layer) -> bool:
        return layer in self.layers

    def has_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def compatibility_check(self, target: EmbeddingDescriptor) -> Optional[ReindexPlan]:
        """Decide whether moving to `target` requires a reindex. Returns
        `None` when the new descriptor matches the current embedding;
        a `ReindexPlan` otherwise.

        Concrete providers should override when they can give better
        cost/duration estimates.
        """
        if self.embedding is None:
            return ReindexPlan(
                layer=Layer.VECTOR,
                reason="No embedding configured; full index required.",
                requires_explicit_approval=True,
            )
        if self.embedding.matches(target):
            return None
        return ReindexPlan(
            layer=Layer.VECTOR,
            reason=(
                f"Embedding swap "
                f"{self.embedding.provider}/{self.embedding.model}({self.embedding.dimension}) → "
                f"{target.provider}/{target.model}({target.dimension}). "
                "Dimension or metric change requires reindex."
            ),
            requires_explicit_approval=True,
        )


# ─────────────────────────────────────────────────────────────────────
# 8. Event names + helpers
# ─────────────────────────────────────────────────────────────────────


class MemoryEvent(str, enum.Enum):
    """Canonical event type strings emitted by Stage 2 / Stage 15.
    Mirrors `MEMORY_SPEC.yaml::events`.
    """

    CONTEXT_BUILT = "context.built"
    CONTEXT_COMPACTED = "context.compacted"
    TURN_RECORDED = "memory.turn_recorded"
    EXECUTION_RECORDED = "memory.execution_recorded"
    INSIGHT = "memory.insight"
    PROMOTED = "memory.promoted"
    REINDEXED = "memory.reindexed"
    COST = "memory.cost"
    SNAPSHOT = "memory.snapshot"


# ─────────────────────────────────────────────────────────────────────
# 9. Layer handles — narrow, capability-gated Protocols
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class STMHandle(Protocol):
    """Short-Term Memory plane. Append-only stream of turns + events.

    The plane also exposes a session-summary slot — a single markdown
    string written once at session close (see ``MemoryStage`` /
    Stage 19 Summarizer) and read back on resume. Hosts that pre-1.20
    wrote ``transcripts/summary.md`` directly should switch to
    ``write_summary`` / ``read_summary`` so the plane stays the
    single source of truth.
    """

    async def append(self, turn: Turn) -> None: ...
    async def append_event(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    async def recent(self, n: int = 20) -> List[Turn]: ...
    async def search(self, text: str, *, limit: int = 10) -> List[Turn]: ...
    async def truncate(self, *, keep_last: int) -> int: ...
    async def read_summary(self) -> Optional[str]: ...
    async def write_summary(self, body: str) -> None: ...


@runtime_checkable
class LTMHandle(Protocol):
    """Long-Term Memory plane. Markdown narrative."""

    async def append(self, body: str, *, heading: Optional[str] = None) -> NoteRef: ...
    async def write_dated(self, body: str, *, day: Optional[datetime] = None) -> NoteRef: ...
    async def write_topic(self, slug: str, body: str) -> NoteRef: ...
    async def read_main(self) -> str: ...
    async def search(self, text: str, *, limit: int = 5) -> List[MemoryChunk]: ...


@runtime_checkable
class NotesHandle(Protocol):
    """Structured notes. CRUD + wikilink + graph."""

    async def list(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        importance: Optional[Importance] = None,
    ) -> List[NoteMeta]: ...
    async def read(self, filename: str) -> Optional[Note]: ...
    async def write(self, draft: NoteDraft) -> NoteMeta: ...
    async def update(self, filename: str, patch: NotePatch) -> NoteMeta: ...
    async def delete(self, filename: str) -> bool: ...
    async def link(self, source: str, target: str) -> bool: ...
    async def graph(self) -> NoteGraph: ...
    async def search(
        self, text: str, *, limit: int = 5, importance_floor: Importance = Importance.LOW
    ) -> List[MemoryChunk]: ...
    async def load_pinned(
        self,
        *,
        category: str = "critical",
        max_chars: int = 3000,
    ) -> str: ...


@runtime_checkable
class VectorHandle(Protocol):
    """Embedding-backed similarity index."""

    @property
    def descriptor(self) -> EmbeddingDescriptor: ...
    async def index(self, ref: NoteRef, text: str) -> int: ...
    async def index_batch(self, items: Sequence[Tuple[NoteRef, str]]) -> int: ...
    async def search(
        self, text: str, *, top_k: int = 5, threshold: float = 0.0
    ) -> List[MemoryChunk]: ...
    async def reindex(self, *, plan: Optional[ReindexPlan] = None) -> ReindexPlan: ...
    async def remove(self, ref: NoteRef) -> bool: ...


@runtime_checkable
class CuratedHandle(Protocol):
    """Per-user curated knowledge plane.
    `notes()` returns a `NotesHandle` scoped to the user; `vector()`
    is optional.
    """

    @property
    def user_id(self) -> str: ...
    def notes(self) -> NotesHandle: ...
    def vector(self) -> Optional[VectorHandle]: ...
    async def promote_from_session(self, ref: NoteRef) -> NoteRef: ...


@runtime_checkable
class GlobalHandle(Protocol):
    """Cross-session global plane."""

    def notes(self) -> NotesHandle: ...
    def vector(self) -> Optional[VectorHandle]: ...
    async def promote_from(self, ref: NoteRef) -> NoteRef: ...


@runtime_checkable
class IndexHandle(Protocol):
    """Derived index/graph cache.

    Distinct from the Notes graph because it includes provider-side
    materialised aggregates (tag counts, importance histogram, link
    graph, file inventory).

    Supports a 4-step **progressive disclosure** read path so hosts
    (or LLM agents) can drill from the high-level vault structure down
    to a single section without paying for a full body load on every
    step:

    1. ``list_categories()``       — every category folder + file count
    2. ``list_notes(category)``    — note summaries within one category
    3. ``read_outline(filename)``  — heading tree of one note
    4. ``read_section(file, hd)``  — body of one heading
    """

    async def snapshot(self) -> Dict[str, Any]: ...
    async def tag_counts(self) -> Dict[str, int]: ...
    async def graph(self) -> NoteGraph: ...
    async def rebuild(self) -> None: ...
    async def list_categories(self) -> List[Dict[str, Any]]: ...
    async def list_notes(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[NoteSummary]: ...
    async def read_outline(self, filename: str) -> Optional[NoteOutline]: ...
    async def read_section(self, filename: str, heading: str) -> Optional[str]: ...
    async def build_vault_map(
        self,
        *,
        recent_limit: int = 5,
        top_tags: int = 10,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]: ...
    async def render_vault_map(
        self,
        *,
        recent_limit: int = 5,
        top_tags: int = 10,
        category_descriptions: Optional[Dict[str, str]] = None,
    ) -> str: ...


# ─────────────────────────────────────────────────────────────────────
# 10. The MemoryProvider Protocol
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class MemoryProvider(Protocol):
    """Unified memory contract.

    Layer handles return `None` when the layer is not supported; cross-
    layer methods (`retrieve`, `record_turn`, ...) MUST be safe to call
    on every provider — implementors degrade gracefully (e.g.,
    `retrieve` skips layers whose handle is None).
    """

    @property
    def descriptor(self) -> MemoryDescriptor: ...

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    # Layer access — `None` means "this provider does not implement
    # that layer". STM, LTM, Notes, Index are required by spec; the
    # remaining three are optional.
    def stm(self) -> STMHandle: ...
    def ltm(self) -> LTMHandle: ...
    def notes(self) -> NotesHandle: ...
    def vector(self) -> Optional[VectorHandle]: ...
    def curated(self) -> Optional[CuratedHandle]: ...
    def global_(self) -> Optional[GlobalHandle]: ...
    def index(self) -> IndexHandle: ...

    # Cross-layer
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult: ...
    async def record_turn(self, turn: Turn) -> None: ...
    async def record_execution(self, summary: ExecutionSummary) -> RecordReceipt: ...
    async def reflect(self, ctx: ReflectionContext) -> Sequence[Insight]: ...
    async def snapshot(self) -> MemorySnapshot: ...
    async def restore(self, snap: MemorySnapshot) -> None: ...
    async def promote(self, ref: NoteRef, to: Scope) -> NoteRef: ...

    # Hook installation — every provider exposes this surface so hosts
    # can attach `MemoryHooks.after_*` callbacks (and future hook
    # bundles) without doing a `hasattr` dance. Composite providers
    # forward to every distinct delegate; concrete providers route
    # the hook bag to their own store layers (STM / Notes); providers
    # that don't fire any hook yet (e.g. SQL backend) still implement
    # the method as `self._hooks = hooks` so the contract holds.
    def set_hooks(self, hooks: "MemoryHooks") -> None: ...


# ─────────────────────────────────────────────────────────────────────
# 11. MemoryHooks — pluggable policy attached to a provider / stage
# ─────────────────────────────────────────────────────────────────────


# Default per-layer budget ratios for ``MemoryAwareRetriever``. Each
# value is the maximum share of the total ``max_inject_chars`` budget
# that the corresponding retrieval layer is allowed to consume. The
# layers run in order so an early layer hitting its cap leaves the
# rest of the budget for downstream layers — no layer is forced to
# fill its full ratio.
_DEFAULT_LAYER_BUDGET_RATIO: Dict[str, float] = {
    "recent_turns": 0.40,
    "session_summary": 0.10,
    "pinned": 0.30,
    "vault_map": 0.05,
    "ltm_main": 0.20,
    "vector": 0.40,
    "keyword": 0.40,
    "backlink": 0.20,
    "curated": 0.20,
}


_DEFAULT_IMPORTANCE_BOOST: Dict[str, float] = {
    "critical": 2.0,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


@dataclass
class MemoryHooks:
    """Pluggable policy + callback bag for the executor's memory plane.

    Attached to a ``MemoryProvider`` via ``provider.set_hooks(hooks)``.
    The same instance is consulted by:

    1. **Stage 18 MemoryStage** — `should_record_execution`,
       `should_reflect`, `should_auto_promote` gate the
       per-turn record / reflect / promote chain.
    2. **Stage 18 post-write fan-out** — `after_record_turn`,
       `after_record_execution`, `after_note_write`,
       `after_note_update` fire fire-and-forget so hosts can layer
       business logic (DM bundle archiver, conversation bucket
       router, VTuber LOGS emit, pin policy decisions) on top of
       the executor's STM/Notes/LTM plane.
    3. **Stage 2 MemoryAwareRetriever** — every retrieval-policy
       field below (`vault_descriptions`, `importance_boost`,
       `layer_budget_ratio`, `pin_category`, `recent_turns`,
       `slim_mode`, `enable_vector_search`, `max_results`,
       `max_inject_chars`, `search_chars`, `vault_map_max_chars`)
       is read live so hosts can adjust retrieval shape from a
       single attach point.

    Construction is a plain dataclass so tests inline-build one with
    only the fields they care about.
    """

    # ── Stage 18 gate callbacks ─────────────────────────────────────
    should_record_execution: Callable[["PipelineState"], bool] = lambda s: bool(s.final_text)
    should_reflect: Callable[["PipelineState"], bool] = lambda s: False
    should_auto_promote: Callable[[Insight], bool] = lambda i: i.should_auto_promote()

    # ── Stage 18 post-write callbacks (fire-and-forget) ─────────────
    after_record_turn: Optional[Callable[[Turn, RecordReceipt], "Awaitable[None]"]] = None
    after_record_execution: Optional[
        Callable[[ExecutionSummary, RecordReceipt], "Awaitable[None]"]
    ] = None
    after_note_write: Optional[Callable[[NoteMeta], "Awaitable[None]"]] = None
    after_note_update: Optional[Callable[[NoteMeta], "Awaitable[None]"]] = None

    # ── Stage 2 retrieval policy ────────────────────────────────────
    # Host-supplied category labels. Used by IndexHandle.render_vault_map
    # so the rendered block matches the host's operator-prompt layout.
    vault_descriptions: Dict[str, str] = field(default_factory=dict)
    # Multiplicative score boost applied to keyword-search results.
    # Indexed by `Importance` value. Boost > 1 promotes, < 1 demotes.
    importance_boost: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_IMPORTANCE_BOOST)
    )
    # Multiplicative score boost applied by category. Empty dict disables.
    category_boosts: Dict[str, float] = field(default_factory=dict)
    # Per-layer fraction of `max_inject_chars`. See
    # `_DEFAULT_LAYER_BUDGET_RATIO`. Hosts may override only the
    # layers they want to clamp — missing keys fall back to defaults.
    layer_budget_ratio: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_LAYER_BUDGET_RATIO)
    )
    # Notes category that holds always-pinned facts. Read by
    # `NotesHandle.load_pinned(category=...)` and stamped on retrieved
    # chunks as `metadata["host_layer"]`.
    pin_category: str = "critical"
    # STM tail size injected as the L0 chunk regardless of query overlap.
    recent_turns: int = 6
    # When True, MemoryAwareRetriever returns only L0/L1/L1.5/L1.7 and
    # leaves heavy semantic / keyword layers to the host's progressive
    # disclosure tools (memory_search / memory_read).
    slim_mode: bool = False
    # When True, the lightweight vault map is injected even outside slim mode.
    always_render_vault_map: bool = True
    # Cap applied to the rendered vault map block.
    vault_map_max_chars: int = 500
    # Vector search switches.
    enable_vector_search: bool = True
    # Per-layer max chunk count (vector / keyword / backlink).
    max_results: int = 5
    # Total character budget for one retrieval call.
    max_inject_chars: int = 10000
    # Cap on the query text actually sent to keyword/vector layers.
    search_chars: int = 500


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _extract_first_user_text(state: "PipelineState") -> str:
    """Best-effort: pull the latest user-turn text from PipelineState."""
    for msg in reversed(state.messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        return str(content)
    return ""


# ─────────────────────────────────────────────────────────────────────
# Public exports
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    # enums
    "Layer",
    "Capability",
    "Scope",
    "Importance",
    "MemoryEvent",
    # value types
    "NoteRef",
    "BackendInfo",
    "EmbeddingDescriptor",
    "CostEvent",
    "CostModel",
    # notes
    "Note",
    "NoteMeta",
    "NoteDraft",
    "NotePatch",
    "NoteGraph",
    "NoteSummary",
    "NoteOutline",
    "OutlineNode",
    "InteractionFields",
    # turn / reflection / execution
    "Turn",
    "ExecutionSummary",
    "RecordReceipt",
    "Insight",
    "ReflectionContext",
    # retrieval
    "RetrievalQuery",
    "RetrievalResult",
    # snapshot / migration
    "MemorySnapshot",
    "ReindexPlan",
    # descriptor
    "MemoryDescriptor",
    # handles
    "STMHandle",
    "LTMHandle",
    "NotesHandle",
    "VectorHandle",
    "CuratedHandle",
    "GlobalHandle",
    "IndexHandle",
    # provider
    "MemoryProvider",
    # policy
    "MemoryHooks",
]
