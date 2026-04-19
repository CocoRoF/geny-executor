"""Directory layout for FileMemoryProvider.

Encapsulates the Geny-compatible on-disk paths so the provider code
never open-codes a path string. Constants here are the *format
contract* with the legacy reader; changes must be evaluated against
Geny's on-disk expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ── directory structure ──────────────────────────────────────────────

TRANSCRIPTS_DIR = "transcripts"
MEMORY_DIR = "memory"
VECTORDB_DIR = "vectordb"

TOPICS_SUBDIR = "topics"

# Hardcoded in Geny — structured notes are organised under one of six
# category directories under `memory/`. `root` means "direct under
# memory/", i.e. no subfolder.
NOTE_CATEGORIES = ("daily", "topics", "entities", "projects", "insights", "root")

# File names that are canonical artefacts of the memory subsystem and
# must not be treated as notes when scanning `memory/`.
RESERVED_FILENAMES = frozenset({"MEMORY.md", "_index.json", "summary.md"})


@dataclass(frozen=True)
class DirectoryLayout:
    """Resolved filesystem paths for one session's memory tree.

    All paths are absolute once `root` is absolute. Construction does
    not create anything on disk — call `ensure()` for that.
    """

    root: Path

    # ── Resolved paths ──────────────────────────────────────────────

    @property
    def transcripts(self) -> Path:
        return self.root / TRANSCRIPTS_DIR

    @property
    def stm_jsonl(self) -> Path:
        return self.transcripts / "session.jsonl"

    @property
    def summary_md(self) -> Path:
        return self.transcripts / "summary.md"

    @property
    def memory(self) -> Path:
        return self.root / MEMORY_DIR

    @property
    def main_ltm(self) -> Path:
        return self.memory / "MEMORY.md"

    @property
    def topics_dir(self) -> Path:
        return self.memory / TOPICS_SUBDIR

    @property
    def index_json(self) -> Path:
        return self.memory / "_index.json"

    @property
    def vectordb(self) -> Path:
        return self.root / VECTORDB_DIR

    @property
    def vector_index(self) -> Path:
        return self.vectordb / "index.faiss"

    @property
    def vector_metadata(self) -> Path:
        return self.vectordb / "metadata.json"

    # ── Helpers ─────────────────────────────────────────────────────

    def dated_ltm(self, day: str) -> Path:
        """Path to `memory/YYYY-MM-DD.md`. `day` is the ISO date."""
        return self.memory / f"{day}.md"

    def topic_ltm(self, slug: str) -> Path:
        return self.topics_dir / f"{slug}.md"

    def note_dir(self, category: str) -> Path:
        """Return the directory for notes of `category`. `root` maps
        to `memory/` itself (no subfolder).
        """
        if category == "root" or not category:
            return self.memory
        return self.memory / category

    def note_path(self, category: str, filename: str) -> Path:
        """Absolute path to `memory/{category}/{filename}`, respecting
        the `root` special case.
        """
        return self.note_dir(category) / filename

    def category_dirs(self) -> Iterable[Path]:
        """All structured-note category directories. `root` is the
        memory directory itself (notes directly under `memory/`).
        """
        for cat in NOTE_CATEGORIES:
            yield self.note_dir(cat)

    def ensure(self) -> None:
        """Create the root directory tree if any piece is missing.
        Idempotent; safe to call on every provider init.
        """
        for d in (
            self.root,
            self.transcripts,
            self.memory,
            self.topics_dir,
            *[self.note_dir(c) for c in NOTE_CATEGORIES if c != "root"],
        ):
            d.mkdir(parents=True, exist_ok=True)

    def is_reserved(self, rel: Path) -> bool:
        """True if `rel` (a path relative to `memory/`) is a
        provider-owned artefact (main LTM body, index cache, summary)
        and therefore not a user-editable note.
        """
        if rel.parts and rel.parts[0].startswith("."):
            return True
        return rel.name in RESERVED_FILENAMES

    def category_of(self, path: Path) -> str:
        """Infer the note category from a path under `memory/`.
        Returns `root` if the note is directly under `memory/`,
        `topics` for `memory/topics/*`, etc.
        """
        try:
            rel = path.relative_to(self.memory)
        except ValueError:
            return ""
        parts = rel.parts
        if len(parts) <= 1:
            return "root"
        first = parts[0]
        if first in NOTE_CATEGORIES:
            return first
        return "root"
