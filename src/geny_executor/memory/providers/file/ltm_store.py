"""LTM plane backed by markdown files under `memory/`.

Storage modes (matching Geny's `LTMConfig`):

  * ``memory/MEMORY.md`` — evergreen, append-only narrative.
  * ``memory/YYYY-MM-DD.md`` — per-day journal. One file per date
    (`day` argument normalised via the provider's timezone).
  * ``memory/topics/{slug}.md`` — per-topic knowledge file.

All writes are append-only. Search is a keyword scan across the three
stores, with the same scoring heuristic as Geny: keyword density
(0.7 weight) plus recency bonus (0.3 weight, half-life 30 days).
The main file gets an evergreen bonus of 1.0 with no decay.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import List, Optional, Tuple

from geny_executor.memory._locks import LoopAgnosticLock
from geny_executor.memory.provider import NoteRef, Scope
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.timezone import now_in
from geny_executor.stages.s02_context.types import MemoryChunk


_RECENCY_HALFLIFE_DAYS = 30.0
_SLUG_RE = re.compile(r"[^a-zA-Z0-9가-힣\-_]+")


class _MarkdownLTMStore:
    """Markdown-on-disk LTM."""

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
        self._lock = LoopAgnosticLock()

    # ── LTMHandle contract ──────────────────────────────────────────

    async def append(self, body: str, *, heading: Optional[str] = None) -> NoteRef:
        """Append to `memory/MEMORY.md`, prefixed by an HTML-comment
        timestamp and an optional `##` heading.
        """
        stamp = now_in(self._tz)
        block = _render_evergreen_block(body, stamp=stamp, heading=heading)
        async with self._lock:
            self._layout.ensure()
            _append(self._layout.main_ltm, block)
        return NoteRef(
            filename="MEMORY.md",
            scope=self._scope,
            category="ltm-main",
        )

    async def write_dated(self, body: str, *, day: Optional[datetime] = None) -> NoteRef:
        when = (day or now_in(self._tz)).astimezone(self._tz)
        date_key = when.date().isoformat()
        block = _render_dated_block(body, stamp=when)
        async with self._lock:
            self._layout.ensure()
            _append(self._layout.dated_ltm(date_key), block)
        return NoteRef(
            filename=f"{date_key}.md",
            scope=self._scope,
            category="ltm-dated",
        )

    async def write_topic(self, slug: str, body: str) -> NoteRef:
        slug = _slugify(slug)
        stamp = now_in(self._tz)
        block = _render_dated_block(body, stamp=stamp)
        async with self._lock:
            self._layout.ensure()
            _append(self._layout.topic_ltm(slug), block)
        return NoteRef(
            filename=f"topics/{slug}.md",
            scope=self._scope,
            category="ltm-topic",
        )

    async def read_main(self) -> str:
        async with self._lock:
            return _read(self._layout.main_ltm)

    async def search(self, text: str, *, limit: int = 5) -> List[MemoryChunk]:
        needle = text.lower().strip()
        if not needle or limit <= 0:
            return []
        keywords = [t for t in re.split(r"\s+", needle) if t]
        if not keywords:
            return []
        async with self._lock:
            candidates = self._list_candidates()
        scored: List[Tuple[float, MemoryChunk]] = []
        for kind, ref_name, path in candidates:
            content = _read(path)
            if not content:
                continue
            density = _keyword_density(content, keywords)
            if density <= 0:
                continue
            recency = 1.0 if kind == "long_term" else _recency_bonus(ref_name, tz=self._tz)
            score = density * 0.7 + recency * 0.3
            chunk = MemoryChunk(
                key=ref_name,
                content=_excerpt(content, keywords, max_chars=1200),
                source=kind,
                relevance_score=score,
                metadata={"ltm_kind": kind, "filename": ref_name},
            )
            scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])
        return [chunk for _, chunk in scored[:limit]]

    # ── housekeeping ────────────────────────────────────────────────

    async def list_files(self) -> List[Tuple[str, str, Path]]:
        async with self._lock:
            return list(self._list_candidates())

    # ── internal ────────────────────────────────────────────────────

    def _list_candidates(self) -> List[Tuple[str, str, Path]]:
        """Return (kind, reference-name, path) triples for every LTM
        file that should participate in search. Order: main → dated
        newest-first → topics alphabetically.
        """
        out: List[Tuple[str, str, Path]] = []
        memory_dir = self._layout.memory
        main = self._layout.main_ltm
        if main.exists():
            out.append(("long_term", "MEMORY.md", main))

        dated: List[Tuple[str, Path]] = []
        for p in memory_dir.glob("*.md"):
            if p.name in {"MEMORY.md", "summary.md"}:
                continue
            if not _looks_like_date(p.stem):
                continue
            dated.append((p.stem, p))
        dated.sort(key=lambda t: t[0], reverse=True)
        for stem, path in dated:
            out.append(("long_term_dated", f"{stem}.md", path))

        topics_dir = self._layout.topics_dir
        if topics_dir.exists():
            topics = sorted(topics_dir.glob("*.md"), key=lambda p: p.name)
            for path in topics:
                out.append(("long_term_topic", f"topics/{path.name}", path))

        return out


# ── helpers ───────────────────────────────────────────────────────────


def _append(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read(path)
    separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(separator + block.rstrip() + "\n")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def _render_evergreen_block(body: str, *, stamp: datetime, heading: Optional[str] = None) -> str:
    lines: List[str] = [f"<!-- {stamp.strftime('%Y-%m-%d %H:%M %Z')} -->"]
    if heading:
        lines.append(f"## {heading}")
        lines.append("")
    lines.append(body.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def _render_dated_block(body: str, *, stamp: datetime) -> str:
    lines = [
        "---",
        f"_({stamp.strftime('%H:%M')} {stamp.strftime('%Z')})_",
        "",
        body.rstrip(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _slugify(raw: str) -> str:
    raw = raw.strip().lower()
    slug = _SLUG_RE.sub("_", raw).strip("_")
    return (slug or "topic")[:64]


def _looks_like_date(stem: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", stem))


def _keyword_density(content: str, keywords: List[str]) -> float:
    body = content.lower()
    total_words = max(1, len(body.split()))
    hits = sum(body.count(k) for k in keywords)
    return hits / total_words if hits > 0 else 0.0


def _recency_bonus(ref_name: str, *, tz: tzinfo) -> float:
    stem = ref_name.rsplit("/", 1)[-1][: -len(".md")] if ref_name.endswith(".md") else ref_name
    try:
        when = datetime.fromisoformat(stem).replace(tzinfo=tz)
    except ValueError:
        return 0.5  # unknown date — medium decay
    today = now_in(tz)
    age = today - when
    if age < timedelta(0):
        age = timedelta(0)
    age_days = age.total_seconds() / 86400.0
    return float(math.pow(2.0, -age_days / _RECENCY_HALFLIFE_DAYS))


def _excerpt(content: str, keywords: List[str], *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    low = content.lower()
    for kw in keywords:
        pos = low.find(kw)
        if pos < 0:
            continue
        start = max(0, pos - max_chars // 4)
        end = min(len(content), start + max_chars)
        snippet = content[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(content):
            snippet = snippet + "…"
        return snippet
    return content[:max_chars] + "…"
