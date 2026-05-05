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
class NoteMeta:
    """Lightweight projection of a note for list/graph operations.

    `metadata` is a host-defined extension channel — providers store
    and round-trip it verbatim, never inspect the contents. Use a
    namespaced key prefix (e.g. ``geny.*``) to avoid collisions.
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


@dataclass
class Note:
    """Full note: frontmatter + body + metadata.

    `frontmatter` is the disk YAML payload (serialised to the .md
    header). `metadata` is the ephemeral host-extension channel —
    not persisted to the YAML, but round-tripped through writes
    and read responses for routing / business hints.
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
        )


@dataclass
class NoteDraft:
    """Input payload for `NotesHandle.write`. Provider assigns the
    canonical filename if `filename` is empty.

    `frontmatter` is the disk YAML payload. `metadata` is the
    ephemeral host-extension channel — not serialised, but
    available to `MemoryHooks` and downstream callers for business
    routing.
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
class NoteGraph:
    """Wikilink graph snapshot. Edge: (source_filename → target_filename).

    `metadata` carries optional host-side annotations (graph build
    timestamp, derived stats, etc.).
    """

    nodes: List[NoteMeta] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def neighbours(self, filename: str) -> List[str]:
        return [b for a, b in self.edges if a == filename]


# ─────────────────────────────────────────────────────────────────────
# 4. Turn / Reflection / Execution
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """One conversational turn for STM recording."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: Any  # str or Anthropic structured content
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

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

        Honours an optional ``message["metadata"]`` dict — host code
        (Geny's `_pending_message_metadata` stamp etc.) writes
        InteractionEvent fields onto the message before stage 18 runs;
        this method preserves that dict on `Turn.metadata` so the
        executor's record_turn path can route it to STM without the
        host having to maintain a parallel write trail.
        """
        raw_meta = message.get("metadata") or {}
        meta_dict: Dict[str, Any]
        if isinstance(raw_meta, Mapping):
            meta_dict = dict(raw_meta)
        else:
            meta_dict = {}
        return cls(
            role=str(message.get("role", "user")),
            content=message.get("content", ""),
            metadata=meta_dict,
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
    """Short-Term Memory plane. Append-only stream of turns + events."""

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
    """

    async def snapshot(self) -> Dict[str, Any]: ...
    async def tag_counts(self) -> Dict[str, int]: ...
    async def graph(self) -> NoteGraph: ...
    async def rebuild(self) -> None: ...
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
# 11. MemoryHooks — pluggable policy attached to a stage instance
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MemoryHooks:
    """Policy callbacks consulted by the rewritten `MemoryStage`. Kept
    as a plain dataclass so test stubs can inline-construct one.

    The ``after_*`` callbacks fire after the corresponding
    ``MemoryProvider`` operation completes (record_turn,
    record_execution, notes.write, notes.update). They run as
    fire-and-forget tasks scheduled on the current event loop —
    failures are logged but never block the primary memory write.
    Hosts use these to layer business logic (DM bundle archiver,
    conversation bucket router, VTuber LOGS emit, pin policy
    decisions) on top of the executor's STM/LTM/notes plane without
    maintaining a parallel pipeline path.
    """

    should_record_execution: Callable[["PipelineState"], bool] = lambda s: bool(s.final_text)
    should_reflect: Callable[["PipelineState"], bool] = lambda s: False
    should_auto_promote: Callable[[Insight], bool] = lambda i: i.should_auto_promote()
    # Post-write callbacks. Default: None (no-op). Awaited inside a
    # detached task by the provider; raise inside the callback to
    # log + drop, never to abort the write.
    after_record_turn: Optional[Callable[[Turn, RecordReceipt], "Awaitable[None]"]] = None
    after_record_execution: Optional[
        Callable[[ExecutionSummary, RecordReceipt], "Awaitable[None]"]
    ] = None
    after_note_write: Optional[Callable[[NoteMeta], "Awaitable[None]"]] = None
    after_note_update: Optional[Callable[[NoteMeta], "Awaitable[None]"]] = None


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
