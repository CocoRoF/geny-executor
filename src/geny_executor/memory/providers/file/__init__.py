"""FileMemoryProvider — disk-persistent memory provider.

Format-compatible with Geny's `SessionMemoryManager` on-disk layout
(STM JSONL in `transcripts/`, LTM markdown in `memory/`, notes
under `memory/{category}/`, vector index under `vectordb/`,
derived index at `memory/_index.json`) — but zero Geny imports.
The compatibility is a directory-level contract, not a code
dependency.

Layers delivered in Phase 2a:
  - STM (JSONL)
  - LTM (markdown with main / dated / topic split)
  - Notes (markdown + hand-rolled YAML frontmatter, wikilinks, tags,
    importance boosts)
  - Index (derivable cache at `memory/_index.json`)

Vector / Curated / Global return None in Phase 2a. VectorHandle
wiring lands in Phase 2b together with the EmbeddingClient Protocol.
"""

from __future__ import annotations

from geny_executor.memory.providers.file.provider import FileMemoryProvider

__all__ = ["FileMemoryProvider"]
