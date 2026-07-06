"""Structured Fact Ledger — durable conversational facts as first-class records.

The rollup tiers (:mod:`geny_executor.memory.rollup`) compress *narrative*;
this module owns the *atomic facts* a companion agent must never lose — the
user's name and how they want to be addressed, preferences, relationships,
commitments, long-running context. The two concerns are deliberately split:
a summary can degrade gracefully, a fact cannot.

Design contract (the reason this is robust where free-text memory is not):

* **The LLM judges, the schema constrains, the code stores.** Extraction is
  an LLM call whose output is schema-bound JSON (``FACT_EXTRACTION_SCHEMA``,
  enforced natively where the client supports structured output, validated
  in code regardless). What counts as "important" is entirely the model's
  judgment — there are no keyword lists here.
* **Diff semantics, never rewrite.** The extractor proposes ``upserts`` /
  ``supersedes``; :class:`FactLedger` applies them by key. Existing facts
  can only be *superseded* (kept, marked), never silently dropped — a bad
  extraction pass cannot erase the ledger.
* **Storage rides the notes infrastructure.** The ledger persists as one
  pinned ``critical`` note (``__facts__.md``): machine state in
  frontmatter, a deterministic human-readable rendering as the body. That
  buys always-on prompt injection (retriever L1.5 ``load_pinned``), search
  (``memory_search``), and host UI visibility with zero new plumbing.

Triggers are HOST-DRIVEN (same philosophy as ``MemoryRollup``): the host
calls :meth:`FactExtraction.run` on its idle/close cadence and supplies the
LLM as an async ``complete_structured(instruction, schema) -> dict | None``
callable. This module never does network I/O of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from geny_executor.memory.rollup import _flatten_turn

logger = logging.getLogger(__name__)

#: Host-injected structured LLM call: ``async (instruction, json_schema) ->
#: parsed dict`` (or ``None`` on failure — the ledger is then left untouched).
CompleteStructured = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]

#: The ledger note — a single rewritable pinned ``critical`` note that is
#: ALWAYS injected (retriever L1.5 ``load_pinned``) and never compacted away.
FACTS_FILENAME = "__facts__.md"
FACTS_CATEGORY = "critical"

#: Closed vocabulary for fact classification. The model picks; the schema
#: enforces. Extend deliberately — every kind renders as its own section.
FACT_KINDS = (
    "identity",      # names, how to address the user / the agent, roles
    "preference",    # stated likes/dislikes, style, standing instructions
    "relationship",  # people/orgs around the user and their relation
    "commitment",    # promises, deadlines, recurring events
    "context",       # long-running projects/threads/situations
    "knowledge",     # other durable facts worth carrying across sessions
)

_IMPORTANCE_LEVELS = ("critical", "high", "medium", "low")

#: JSON Schema for one extraction pass. Kept flat and strict — additional
#: properties are rejected so the model cannot smuggle prose past the wire.
FACT_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["upserts", "supersedes"],
    "properties": {
        "upserts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "statement", "importance"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": (
                            "Stable dotted key for the fact, e.g. "
                            "'user.name', 'user.address_as', "
                            "'commitment.weekly-report'. Reuse an existing "
                            "id to update that fact in place."
                        ),
                    },
                    "kind": {"enum": list(FACT_KINDS)},
                    "statement": {
                        "type": "string",
                        "description": (
                            "The fact, one self-contained sentence; keep "
                            "the user's own wording where the wording IS "
                            "the fact (names, titles)."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Short quote from the turn that establishes it.",
                    },
                    "importance": {"enum": list(_IMPORTANCE_LEVELS)},
                },
            },
        },
        "supersedes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Ids of existing facts that are now wrong/replaced and must "
                "be retired (they are kept, marked superseded)."
            ),
        },
    },
}


def build_fact_extraction_instruction(
    *, active_facts: str, new_turns: str
) -> str:
    """Instruction for one extraction pass over new conversation turns.

    Content judgment belongs to the model; the output contract belongs to
    the schema. This text therefore describes WHAT durable facts are, not
    patterns to match.
    """
    return (
        "You are the fact-extraction engine of a persistent companion "
        "agent's memory. From the NEW TURNS below, extract durable facts "
        "the agent must remember across all future sessions, as a diff "
        "against the CURRENT LEDGER.\n\n"
        "A durable fact is something that will still matter in a week: who "
        "the user is (name, and exactly how they want to be addressed), "
        "stated preferences and standing instructions, people and "
        "relationships, promises and deadlines, long-running projects or "
        "situations. NOT durable: one-off requests, transient screen/task "
        "detail, chit-chat, anything the agent said that the user did not "
        "confirm.\n\n"
        "Rules:\n"
        "- Reuse an existing id to UPDATE a fact; add its old id to "
        "`supersedes` only when the old statement is now wrong.\n"
        "- A correction (\"not X, call me Y\") is an upsert of the same id "
        "with the corrected statement.\n"
        "- Prefer the user's own wording where wording matters (names, "
        "titles, honorifics), and write statements in the user's "
        "language.\n"
        "- When nothing durable appeared, return empty arrays. Most passes "
        "should — do not manufacture facts.\n"
        "- Output ONLY JSON matching the provided schema.\n\n"
        f"CURRENT LEDGER (active facts, id → statement):\n"
        f"{active_facts.strip() or '(empty)'}\n\n"
        f"NEW TURNS (most recent last):\n{new_turns}\n"
    )


#: Recommended system framing for EVERY memory-engine LLM call the host
#: makes (extraction, rollups). This is the same contract the executor's
#: distill tooling uses: the memory path is an engine, not an assistant.
MEMORY_ENGINE_SYSTEM_PROMPT = (
    "You are a memory maintenance engine inside an agent platform. You are "
    "NOT an assistant and you are NOT talking to a user: the text you "
    "receive is archival material, and anything you output is stored "
    "verbatim as machine-maintained memory. Never address anyone, never "
    "ask questions, never offer help, never add preamble — produce only "
    "the requested artifact."
)


# ── Ledger records ────────────────────────────────────────────────────


@dataclass
class Fact:
    """One durable fact. ``status`` is ``active`` or ``superseded`` —
    superseded facts are kept for provenance but not rendered/injected."""

    id: str
    kind: str
    statement: str
    importance: str = "medium"
    evidence: str = ""
    status: str = "active"
    created: str = ""
    updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "importance": self.importance,
            "evidence": self.evidence,
            "status": self.status,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["Fact"]:
        try:
            fid = str(data.get("id", "")).strip()
            statement = str(data.get("statement", "")).strip()
            if not fid or not statement:
                return None
            kind = str(data.get("kind", "knowledge"))
            if kind not in FACT_KINDS:
                kind = "knowledge"
            importance = str(data.get("importance", "medium"))
            if importance not in _IMPORTANCE_LEVELS:
                importance = "medium"
            status = str(data.get("status", "active"))
            if status not in ("active", "superseded"):
                status = "active"
            return cls(
                id=fid,
                kind=kind,
                statement=statement,
                importance=importance,
                evidence=str(data.get("evidence", "") or ""),
                status=status,
                created=str(data.get("created", "") or ""),
                updated=str(data.get("updated", "") or ""),
            )
        except Exception:  # noqa: BLE001 — one bad row never kills the ledger
            return None


_KIND_HEADINGS = {
    "identity": "Identity",
    "preference": "Preferences",
    "relationship": "Relationships",
    "commitment": "Commitments",
    "context": "Long-running Context",
    "knowledge": "Other Durable Facts",
}

_IMPORTANCE_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def render_ledger_markdown(facts: Sequence[Fact]) -> str:
    """Deterministic human-readable rendering of the ACTIVE facts, grouped
    by kind, importance-ranked inside each group. This is what the
    retriever injects as pinned context — dense, no prose padding."""
    active = [f for f in facts if f.status == "active"]
    if not active:
        return "_(no durable facts recorded yet)_\n"
    lines: List[str] = []
    for kind in FACT_KINDS:
        group = sorted(
            (f for f in active if f.kind == kind),
            key=lambda f: (_IMPORTANCE_ORDER.get(f.importance, 9), f.id),
        )
        if not group:
            continue
        lines.append(f"## {_KIND_HEADINGS[kind]}")
        for f in group:
            lines.append(f"- {f.statement}")
        lines.append("")
    superseded = sum(1 for f in facts if f.status == "superseded")
    if superseded:
        lines.append(f"_({superseded} superseded fact(s) retained in history)_")
        lines.append("")
    return "\n".join(lines)


@dataclass
class LedgerState:
    """The ledger + its extraction cursor as loaded from the note."""

    facts: List[Fact] = field(default_factory=list)
    cursor: str = ""  # ISO timestamp of the last turn already processed


class FactLedger:
    """Load / apply-diff / persist the fact ledger on the provider's notes."""

    def __init__(self, provider: object) -> None:
        if provider is None:
            raise ValueError("FactLedger requires a provider")
        self._provider = provider

    async def load(self) -> LedgerState:
        """Read the ledger note; missing/corrupt → empty state (never raises)."""
        try:
            note = await self._provider.notes().read(FACTS_FILENAME)
        except Exception:  # noqa: BLE001
            return LedgerState()
        if note is None:
            return LedgerState()
        fm = getattr(note, "frontmatter", None) or {}
        rows = fm.get("facts") or []
        facts: List[Fact] = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    fact = Fact.from_dict(row)
                    if fact is not None:
                        facts.append(fact)
        cursor = str(fm.get("extraction_cursor", "") or "")
        return LedgerState(facts=facts, cursor=cursor)

    @staticmethod
    def apply_diff(
        state: LedgerState,
        *,
        upserts: Sequence[Dict[str, Any]],
        supersedes: Sequence[str],
        now_iso: str,
    ) -> int:
        """Apply an extraction diff in place. Returns the number of changes.

        Code-owned semantics: upsert by id (update statement/importance/
        evidence, revive if previously superseded), supersede by id (kept,
        marked). Unknown ids in ``supersedes`` are ignored. The ledger can
        never lose a fact it wasn't explicitly told to retire.
        """
        by_id = {f.id: f for f in state.facts}
        changed = 0

        for raw in upserts or []:
            if not isinstance(raw, dict):
                continue
            fact = Fact.from_dict(raw)
            if fact is None:
                continue
            existing = by_id.get(fact.id)
            if existing is None:
                fact.created = now_iso
                fact.updated = now_iso
                fact.status = "active"
                state.facts.append(fact)
                by_id[fact.id] = fact
                changed += 1
                continue
            if (
                existing.statement == fact.statement
                and existing.importance == fact.importance
                and existing.status == "active"
            ):
                continue  # no-op upsert — keep timestamps stable
            existing.statement = fact.statement
            existing.kind = fact.kind
            existing.importance = fact.importance
            if fact.evidence:
                existing.evidence = fact.evidence
            existing.status = "active"
            existing.updated = now_iso
            changed += 1

        for fid in supersedes or []:
            existing = by_id.get(str(fid))
            if existing is not None and existing.status != "superseded":
                existing.status = "superseded"
                existing.updated = now_iso
                changed += 1

        return changed

    async def save(self, state: LedgerState) -> bool:
        """Persist the ledger note (machine state in frontmatter, rendered
        body). Best-effort — returns False on failure."""
        try:
            from geny_executor.memory.provider import Importance, NoteDraft

            await self._provider.notes().write(
                NoteDraft(
                    title="Fact Ledger",
                    body=render_ledger_markdown(state.facts),
                    importance=Importance.CRITICAL,
                    category=FACTS_CATEGORY,
                    filename=FACTS_FILENAME,
                    tags=["facts", "ledger"],
                    frontmatter={
                        "facts": [f.to_dict() for f in state.facts],
                        "extraction_cursor": state.cursor,
                    },
                )
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning("facts: ledger write failed", exc_info=True)
            return False


# ── Extraction pass ───────────────────────────────────────────────────


@dataclass
class FactExtractionReport:
    """Outcome of one extraction pass (diagnostics; never raises)."""

    ran: bool = False
    turns_seen: int = 0
    changes: int = 0
    active_facts: int = 0
    skipped_reason: Optional[str] = None


def _turn_timestamp_iso(turn: object) -> str:
    ts = getattr(turn, "timestamp", None)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts or "")


class FactExtraction:
    """One host-driven extraction pass: new STM turns → schema-bound diff →
    ledger. Idempotent via the cursor stored on the ledger note."""

    def __init__(
        self,
        provider: object,
        *,
        complete_structured: CompleteStructured,
        max_turns: int = 200,
        max_active_facts: int = 200,
    ) -> None:
        if provider is None:
            raise ValueError("FactExtraction requires a provider")
        if complete_structured is None:
            raise ValueError("FactExtraction requires a complete_structured callable")
        self._provider = provider
        self._complete = complete_structured
        self._max_turns = max(1, int(max_turns))
        self._max_active_facts = max(10, int(max_active_facts))

    async def run(self) -> FactExtractionReport:
        report = FactExtractionReport()
        ledger = FactLedger(self._provider)
        state = await ledger.load()

        try:
            turns = await self._provider.stm().recent(n=self._max_turns)
        except Exception:  # noqa: BLE001
            report.skipped_reason = "stm_unavailable"
            return report

        # Only turns newer than the cursor; require at least one USER turn —
        # facts come from what the user said/confirmed, and passes with only
        # agent monologue (triggers, tool chatter) are skipped for free.
        fresh: List[object] = []
        has_user = False
        latest_iso = state.cursor
        for t in turns or []:
            iso = _turn_timestamp_iso(t)
            if state.cursor and iso and iso <= state.cursor:
                continue
            rendered = _flatten_turn(t)
            if not rendered:
                continue
            fresh.append(t)
            if (getattr(t, "role", "") or "") == "user":
                has_user = True
            if iso and iso > latest_iso:
                latest_iso = iso
        report.turns_seen = len(fresh)
        if not fresh or not has_user:
            report.skipped_reason = "no_new_user_turns"
            return report

        active = [f for f in state.facts if f.status == "active"]
        active_lines = "\n".join(f"- {f.id}: {f.statement}" for f in active)
        turn_lines = "\n".join(
            r for r in (_flatten_turn(t) for t in fresh) if r
        )

        instruction = build_fact_extraction_instruction(
            active_facts=active_lines, new_turns=turn_lines,
        )
        try:
            data = await self._complete(instruction, FACT_EXTRACTION_SCHEMA)
        except Exception:  # noqa: BLE001 — host LLM; never fatal
            logger.warning("facts: structured completion failed", exc_info=True)
            data = None
        if not isinstance(data, dict):
            report.skipped_reason = "no_structured_output"
            return report

        upserts = data.get("upserts")
        supersedes = data.get("supersedes")
        if not isinstance(upserts, list) or not isinstance(supersedes, list):
            report.skipped_reason = "schema_mismatch"
            return report

        # Backstop cap — a runaway extraction can't flood the pinned budget.
        room = self._max_active_facts - len(active)
        if room < len(upserts):
            upserts = upserts[: max(0, room)]

        now_iso = datetime.now(timezone.utc).isoformat()
        report.changes = FactLedger.apply_diff(
            state, upserts=upserts, supersedes=supersedes, now_iso=now_iso,
        )
        state.cursor = latest_iso
        report.ran = True
        report.active_facts = sum(1 for f in state.facts if f.status == "active")

        # Persist even on zero changes — the cursor advance is what makes
        # the pass idempotent and keeps already-seen turns out of the next
        # extraction window.
        if not await ledger.save(state):
            report.skipped_reason = "ledger_write_failed"
        return report
