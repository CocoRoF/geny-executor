"""Tiered semantic memory compaction (rollup) — the compressed view a host serves first.

The executor keeps raw turns/events (STM) and structured notes; what was missing is a
*proper* semantic compression that is served first and lets the agent drill down. This
module builds that compressed tier:

    L1 rolling digest  →  STMHandle.write_summary()   (retriever L1, always injected)

from the raw STM the executor already keeps. It FOLDS the prior digest together with the
new raw turns into an updated digest (bounded size, history preserved in compressed
form), rather than re-summarizing from scratch or listing turns mechanically.

Design decisions (see Geny docs/analysis/memory-compression-pipeline.md):
  * Triggers are HOST-DRIVEN (idle / context-pressure / lazy) — this module never does
    network I/O on pipeline build. The host calls ``run()`` when appropriate.
  * Summarization is a host-injected async callable (the host owns model + transport).
    This module owns the ORCHESTRATION + the PRESERVATION-FOCUSED instruction, so every
    host gets the same thorough, important-info-preserving behavior. Cost is not
    optimized away: the instruction asks for a thorough, structured digest.
  * IMPORTANT INFO IS ALWAYS PRESERVED — the instruction names the categories that must
    never be dropped (facts, decisions, entities, user preferences/commitments, open
    threads, relationship/affect). Raw is never deleted here — only compressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Host-injected summarizer: ``async (instruction) -> digest_markdown``. The host wires
#: its LLM (model + transport) here; this module supplies the full instruction.
Summarize = Callable[[str], Awaitable[str]]

#: Host-injected STRUCTURED summarizer: ``async (instruction, json_schema) ->
#: parsed dict | None``. When supplied, digests are produced as schema-bound
#: JSON and rendered to markdown by code — a conversational assistant reply
#: is a schema violation and simply leaves the previous digest in place.
CompleteStructured = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]


def _string_array() -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


#: Schema for the L1 rolling digest (structured mode).
SEGMENT_DIGEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "facts_decisions",
        "entities",
        "preferences_commitments",
        "open_threads",
        "relationship_mood",
    ],
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-5 sentences: what this session is about + where it stands",
        },
        "facts_decisions": _string_array(),
        "entities": _string_array(),
        "preferences_commitments": _string_array(),
        "open_threads": _string_array(),
        "relationship_mood": _string_array(),
    },
}

#: Schema for the L3 evergreen (structured mode). Narrative tiers only —
#: atomic facts belong to the Fact Ledger (``geny_executor.memory.facts``).
EVERGREEN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "identity",
        "user",
        "durable_facts",
        "preferences_commitments",
        "long_running_threads",
    ],
    "properties": {
        "identity": _string_array(),
        "user": _string_array(),
        "durable_facts": _string_array(),
        "preferences_commitments": _string_array(),
        "long_running_threads": _string_array(),
    },
}


def _bullets(items: Any) -> List[str]:
    out: List[str] = []
    for item in items if isinstance(items, list) else []:
        text = str(item).strip()
        if text:
            out.append(f"- {text}")
    return out


def render_segment_digest(data: Dict[str, Any]) -> str:
    """Deterministic markdown for a schema-bound rolling digest."""
    sections = [
        ("Summary", None),
        ("Facts & Decisions", "facts_decisions"),
        ("Entities", "entities"),
        ("User Preferences & Commitments", "preferences_commitments"),
        ("Open Threads", "open_threads"),
        ("Relationship & Mood", "relationship_mood"),
    ]
    lines: List[str] = []
    summary = str(data.get("summary", "")).strip()
    if summary:
        lines += ["## Summary", summary, ""]
    for title, key in sections[1:]:
        rows = _bullets(data.get(key))
        if rows:
            lines += [f"## {title}", *rows, ""]
    return "\n".join(lines).strip()


def render_evergreen(data: Dict[str, Any]) -> str:
    """Deterministic markdown for a schema-bound evergreen note."""
    sections = [
        ("Identity", "identity"),
        ("User", "user"),
        ("Durable Facts", "durable_facts"),
        ("Preferences & Commitments", "preferences_commitments"),
        ("Long-running Threads", "long_running_threads"),
    ]
    lines: List[str] = []
    for title, key in sections:
        rows = _bullets(data.get(key))
        if rows:
            lines += [f"## {title}", *rows, ""]
    return "\n".join(lines).strip()


#: The non-negotiable preservation clause shared by every rollup instruction. These are
#: the things a digest must carry forward — verbatim where wording matters.
PRESERVE_CLAUSE = (
    "ALWAYS PRESERVE (never drop; keep verbatim where exact wording matters):\n"
    "- concrete facts, figures, identifiers, file paths\n"
    "- decisions made and their rationale\n"
    "- named entities (people, projects, tools, places)\n"
    "- the user's stated preferences, instructions, and commitments\n"
    "- open threads, TODOs, and unresolved questions\n"
    "- relationship and emotional/affective state (for a persona / VTuber)\n"
)


def _flatten_turn(turn: object) -> Optional[str]:
    """Render one STM ``Turn`` as ``[role] text``, flattening block content."""
    content = getattr(turn, "content", "") or ""
    if isinstance(content, list):
        pieces = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        content = "\n".join(p for p in pieces if p)
    text = str(content).strip()
    if not text:
        return None
    role = getattr(turn, "role", "user") or "user"
    return f"[{role}] {text}"


def build_segment_instruction(
    *, prior_digest: str, raw_turns: str, max_chars: int
) -> str:
    """The proper, preservation-focused rolling-digest instruction.

    Folds ``prior_digest`` + ``raw_turns`` into an updated digest. Structured (so it is
    drillable), thorough (not a one-liner), and bounded to ``max_chars``.
    """
    prior_block = (
        f"PRIOR DIGEST (carry everything still relevant forward):\n{prior_digest}\n\n"
        if prior_digest.strip()
        else "PRIOR DIGEST: (none yet)\n\n"
    )
    return (
        "You are maintaining a session's rolling MEMORY DIGEST — the compressed view "
        "that is served to the agent first, before any raw history. Update it by folding "
        "the new raw turns into the prior digest.\n\n"
        f"{PRESERVE_CLAUSE}\n"
        "FORMAT — return Markdown with these sections (omit a section only if truly "
        "empty):\n"
        "## Summary\n(2-5 sentences: what this session is about + where it stands)\n"
        "## Facts & Decisions\n(durable facts, decisions + rationale, identifiers)\n"
        "## Entities\n(people / projects / tools / files, with one-line context each)\n"
        "## User Preferences & Commitments\n## Open Threads\n## Relationship & Mood\n\n"
        "RULES: be faithful — never invent; prefer the user's own wording for "
        "preferences/commitments; keep it information-dense; do NOT list turns "
        "mechanically; merge duplicates; drop only chit-chat with no lasting value. "
        f"Keep the whole digest under ~{max_chars} characters.\n\n"
        f"{prior_block}"
        f"NEW RAW TURNS (most recent last):\n{raw_turns}\n"
    )


def build_segment_instruction_structured(
    *, prior_digest: str, raw_turns: str, max_chars: int
) -> str:
    """Structured-mode rolling-digest instruction — same preservation
    contract, but the output is schema-bound JSON (rendered by code)."""
    prior_block = (
        f"PRIOR DIGEST (carry everything still relevant forward):\n{prior_digest}\n\n"
        if prior_digest.strip()
        else "PRIOR DIGEST: (none yet)\n\n"
    )
    return (
        "You maintain a session's rolling MEMORY DIGEST — the compressed view "
        "served to the agent before any raw history. Fold the new raw turns "
        "into the prior digest.\n\n"
        f"{PRESERVE_CLAUSE}\n"
        "RULES: be faithful — never invent; prefer the user's own wording for "
        "preferences/commitments; keep it information-dense; merge duplicates; "
        "drop only chit-chat with no lasting value. Keep the total content "
        f"under ~{max_chars} characters. Output ONLY JSON matching the "
        "provided schema (empty arrays where a section is truly empty).\n\n"
        f"{prior_block}"
        f"NEW RAW TURNS (most recent last):\n{raw_turns}\n"
    )


#: The evergreen note — a single rewritable pinned ``critical`` note that is
#: ALWAYS injected (retriever L1.5 ``load_pinned``) and never compacted away.
EVERGREEN_FILENAME = "__evergreen__.md"
EVERGREEN_CATEGORY = "critical"


def build_evergreen_instruction(
    *, current: str, recent_digest: str, max_chars: int
) -> str:
    """Instruction for the L3 EVERGREEN merge — the durable, always-loaded core.

    Merges the latest rolling digest into the current evergreen, keeping only
    durable knowledge (identity, who the user is, long-running facts /
    preferences / commitments / threads), never losing a load-bearing fact.
    """
    return (
        "You maintain the EVERGREEN MEMORY — the durable, always-loaded core an "
        "agent carries across ALL sessions: stable identity, who the user is, "
        "long-running facts, preferences, commitments, and threads that persist "
        "beyond any single session. Update it by merging the latest rolling "
        "digest into the current evergreen.\n\n"
        f"{PRESERVE_CLAUSE}\n"
        "RULES: keep ONLY durable knowledge — drop ephemeral / one-off chatter "
        "(the rolling digest already holds the recent detail); merge duplicates; "
        "NEVER lose a durable fact, name, preference, or commitment; prefer the "
        "user's own wording. Return structured Markdown (## Identity / ## User / "
        "## Durable Facts / ## Preferences & Commitments / ## Long-running Threads "
        "— omit empty sections). "
        f"Keep the whole evergreen under ~{max_chars} characters.\n\n"
        f"CURRENT EVERGREEN (+ pinned critical facts):\n"
        f"{current.strip() or '(none yet)'}\n\n"
        f"LATEST ROLLING DIGEST:\n{recent_digest.strip() or '(none)'}\n"
    )


def build_evergreen_instruction_structured(
    *, current: str, recent_digest: str, max_chars: int
) -> str:
    """Structured-mode evergreen merge. Narrative-tier only: atomic facts
    (names, preferences as records) are owned by the Fact Ledger; the
    evergreen carries the durable NARRATIVE that gives them context."""
    return (
        "You maintain the EVERGREEN MEMORY — the durable, always-loaded core "
        "an agent carries across ALL sessions: stable identity, who the user "
        "is, long-running facts, preferences, commitments, and threads that "
        "persist beyond any single session. Merge the latest rolling digest "
        "into the current evergreen.\n\n"
        f"{PRESERVE_CLAUSE}\n"
        "RULES: keep ONLY durable knowledge — drop ephemeral / one-off "
        "chatter; merge duplicates; NEVER lose a durable fact, name, "
        "preference, or commitment; prefer the user's own wording. Keep the "
        f"total content under ~{max_chars} characters. Output ONLY JSON "
        "matching the provided schema (empty arrays where a section is "
        "truly empty).\n\n"
        f"CURRENT EVERGREEN (+ pinned critical facts):\n"
        f"{current.strip() or '(none yet)'}\n\n"
        f"LATEST ROLLING DIGEST:\n{recent_digest.strip() or '(none)'}\n"
    )


@dataclass
class RollupReport:
    """Outcome of a rollup pass (diagnostics; never raises to the caller)."""

    segment_written: bool = False
    evergreen_written: bool = False
    daily_written: bool = False
    turns_seen: int = 0
    chars_in: int = 0
    chars_out: int = 0
    error: Optional[str] = None


class MemoryRollup:
    """Builds the compressed memory tiers from raw STM via a host LLM.

    Args:
        provider: live ``MemoryProvider``.
        summarize: host-injected ``async (instruction) -> digest`` LLM callable.
        max_turns: how many recent raw turns to fold per pass.
        digest_max_chars: target ceiling for the rolling digest.
    """

    def __init__(
        self,
        provider: object,
        *,
        summarize: Summarize,
        complete_structured: Optional[CompleteStructured] = None,
        max_turns: int = 300,
        digest_max_chars: int = 6000,
        evergreen_max_chars: int = 8000,
    ) -> None:
        if provider is None:
            raise ValueError("MemoryRollup requires a provider")
        if summarize is None:
            raise ValueError("MemoryRollup requires a summarize callable")
        self._provider = provider
        self._summarize = summarize
        # Structured mode: when the host supplies a schema-bound completion,
        # digests are produced as validated JSON and rendered by code — a
        # freeform/conversational reply cannot overwrite good state.
        self._complete_structured = complete_structured
        self._max_turns = max(1, int(max_turns))
        self._digest_max_chars = max(500, int(digest_max_chars))
        self._evergreen_max_chars = max(1000, int(evergreen_max_chars))

    async def summarize_segment(self) -> Optional[str]:
        """Fold the prior digest + recent raw turns into an updated rolling digest and
        persist it to the summary slot (retriever L1). Returns the digest, or None when
        there is nothing to compress. Never raises — best-effort."""
        try:
            stm = self._provider.stm()
            turns = await stm.recent(n=self._max_turns)
        except Exception:  # noqa: BLE001
            logger.debug("rollup: stm.recent failed", exc_info=True)
            return None

        lines: List[str] = []
        for t in turns or []:
            rendered = _flatten_turn(t)
            if rendered:
                lines.append(rendered)
        if not lines:
            return None
        raw_turns = "\n".join(lines)

        try:
            prior = (await stm.read_summary()) or ""
        except Exception:  # noqa: BLE001
            prior = ""

        if self._complete_structured is not None:
            instruction = build_segment_instruction_structured(
                prior_digest=prior,
                raw_turns=raw_turns,
                max_chars=self._digest_max_chars,
            )
            try:
                data = await self._complete_structured(
                    instruction, SEGMENT_DIGEST_SCHEMA
                )
            except Exception:  # noqa: BLE001 — host LLM; never fatal
                logger.warning("rollup: structured segment failed", exc_info=True)
                data = None
            if not isinstance(data, dict):
                # Contract violation → keep the existing digest untouched.
                return None
            digest = render_segment_digest(data).strip()
            if not digest:
                return None
        else:
            instruction = build_segment_instruction(
                prior_digest=prior,
                raw_turns=raw_turns,
                max_chars=self._digest_max_chars,
            )
            try:
                digest = (await self._summarize(instruction) or "").strip()
            except Exception:  # noqa: BLE001 — summarizer is host code; never fatal
                logger.warning("rollup: summarize callable failed", exc_info=True)
                return None
            if not digest:
                return None

        try:
            await stm.write_summary(digest)
        except Exception:  # noqa: BLE001
            logger.warning("rollup: write_summary failed", exc_info=True)
            return None
        return digest

    async def rollup_evergreen(self) -> Optional[str]:
        """Maintain the L3 EVERGREEN — merge the latest rolling digest into the
        durable, always-injected evergreen note (a rewritable pinned ``critical``
        note). Returns the merged evergreen or None. Never raises — best-effort."""
        try:
            notes = self._provider.notes()
        except Exception:  # noqa: BLE001
            return None

        try:
            current = await notes.load_pinned(
                category=EVERGREEN_CATEGORY, max_chars=self._evergreen_max_chars
            )
        except Exception:  # noqa: BLE001
            current = ""
        try:
            recent = (await self._provider.stm().read_summary()) or ""
        except Exception:  # noqa: BLE001
            recent = ""
        if not (current or "").strip() and not (recent or "").strip():
            return None

        if self._complete_structured is not None:
            instruction = build_evergreen_instruction_structured(
                current=current or "", recent_digest=recent or "",
                max_chars=self._evergreen_max_chars,
            )
            try:
                data = await self._complete_structured(instruction, EVERGREEN_SCHEMA)
            except Exception:  # noqa: BLE001 — host LLM; never fatal
                logger.warning("rollup: structured evergreen failed", exc_info=True)
                data = None
            if not isinstance(data, dict):
                # Contract violation → keep the existing evergreen untouched.
                return None
            merged = render_evergreen(data).strip()
            if not merged:
                return None
        else:
            instruction = build_evergreen_instruction(
                current=current or "", recent_digest=recent or "",
                max_chars=self._evergreen_max_chars,
            )
            try:
                merged = (await self._summarize(instruction) or "").strip()
            except Exception:  # noqa: BLE001 — host LLM; never fatal
                logger.warning("rollup: evergreen summarize failed", exc_info=True)
                return None
            if not merged:
                return None

        try:
            from geny_executor.memory.provider import Importance, NoteDraft

            await notes.write(
                NoteDraft(
                    title="Evergreen Memory",
                    body=merged,
                    importance=Importance.CRITICAL,
                    category=EVERGREEN_CATEGORY,
                    filename=EVERGREEN_FILENAME,
                    tags=["evergreen", "memory"],
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("rollup: evergreen write failed", exc_info=True)
            return None
        return merged

    async def rollup_daily(self, *, day: str) -> Optional[str]:
        """Persist the current rolling digest as the L2 DAILY digest note for
        ``day`` (``YYYY-MM-DD``) — idempotent per day (overwrites as the day
        progresses). Gives a date-navigable series of per-day compressed digests
        the agent (and Opsidian) can drill into. Host supplies ``day`` (the
        library takes no wall-clock). Best-effort."""
        if not day:
            return None
        try:
            recent = (await self._provider.stm().read_summary()) or ""
        except Exception:  # noqa: BLE001
            recent = ""
        if not recent.strip():
            return None
        try:
            from geny_executor.memory.provider import Importance, NoteDraft

            await self._provider.notes().write(
                NoteDraft(
                    title=f"Daily Digest · {day}",
                    body=recent,
                    importance=Importance.HIGH,
                    category="daily",
                    filename=f"__digest_{day}__.md",
                    tags=["digest", "daily"],
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("rollup: daily write failed", exc_info=True)
            return None
        return recent

    async def run(
        self, *, evergreen: bool = False, daily_key: Optional[str] = None
    ) -> RollupReport:
        """Run a rollup pass (host calls this on idle / context-pressure / lazily).

        Always folds the L1 rolling digest. ``evergreen=True`` (slower cadence)
        merges the L3 durable evergreen; ``daily_key`` (``YYYY-MM-DD``) writes the
        L2 daily digest note. The compressed map is layered on separately.
        """
        report = RollupReport()
        try:
            digest = await self.summarize_segment()
            if digest:
                report.segment_written = True
                report.chars_out = len(digest)
            if evergreen:
                ever = await self.rollup_evergreen()
                report.evergreen_written = bool(ever)
            if daily_key:
                day = await self.rollup_daily(day=daily_key)
                report.daily_written = bool(day)
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            report.error = str(exc)
            logger.debug("rollup.run failed", exc_info=True)
        return report


__all__ = [
    "MemoryRollup",
    "RollupReport",
    "Summarize",
    "PRESERVE_CLAUSE",
    "build_segment_instruction",
    "build_evergreen_instruction",
    "EVERGREEN_FILENAME",
    "EVERGREEN_CATEGORY",
]
