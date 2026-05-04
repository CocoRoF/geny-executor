"""Auto-vector indexing on `NotesHandle.write` / `update`.

`FileMemoryProvider.__init__` plugs the vector store's `index`
method into the notes store as a `vector_indexer` callback. Every
successful note write/update therefore embeds the body and lands a
row in the vector index without the caller (or a stage author)
having to remember a second `vector.index()` call.

Tests use `LocalHashEmbeddingClient` so the embedding round-trip is
deterministic and offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    NoteDraft,
    NotePatch,
    Scope,
)
from geny_executor.memory.providers import FileMemoryProvider


async def _build_provider(root: Path) -> FileMemoryProvider:
    provider = FileMemoryProvider(
        root=root,
        scope=Scope.SESSION,
        embedding_client=LocalHashEmbeddingClient(model="hash-v1", dimension=64),
    )
    await provider.initialize()
    return provider


class TestAutoVectorOnWrite:
    async def test_write_creates_vector_row(self, tmp_path: Path) -> None:
        provider = await _build_provider(tmp_path)

        meta = await provider.notes().write(
            NoteDraft(
                title="Auto-indexed",
                body="The vector store should pick this up automatically.",
                category="topics",
                scope=Scope.SESSION,
            )
        )

        results = await provider.vector().search(
            "vector store should pick this up", top_k=3
        )
        # The deterministic hash embedding always returns *something*
        # for matching n-grams; require the auto-indexed note to be
        # one of the hits.
        assert any(r.key == meta.ref.filename for r in results)

    async def test_update_replaces_vector_row(self, tmp_path: Path) -> None:
        provider = await _build_provider(tmp_path)

        meta = await provider.notes().write(
            NoteDraft(
                title="Living doc",
                body="Initial content describing turnips.",
                category="topics",
                scope=Scope.SESSION,
                filename="living-doc.md",
            )
        )

        await provider.notes().update(
            meta.ref.filename,
            NotePatch(body="Replaced content describing parsnips entirely."),
        )

        # Search for the *new* body — the indexer must have replaced
        # the row in lockstep with the markdown rewrite, otherwise a
        # query against the new keyword would miss it.
        results = await provider.vector().search(
            "parsnips entirely", top_k=3
        )
        assert any(r.key == meta.ref.filename for r in results)

    async def test_no_indexer_when_no_embedding(self, tmp_path: Path) -> None:
        # Without `embedding_client`, the provider should never wire an
        # indexer, and vector() returns None.
        provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
        await provider.initialize()
        await provider.notes().write(
            NoteDraft(
                title="No embed",
                body="No vector layer at all.",
                category="topics",
                scope=Scope.SESSION,
            )
        )
        assert provider.vector() is None

    async def test_indexer_failure_does_not_fail_write(
        self, tmp_path: Path
    ) -> None:
        # Hand-rolled provider that injects a failing indexer.
        from geny_executor.memory.providers.file.notes_store import (
            _FilesystemNotesStore,
        )

        provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
        await provider.initialize()

        async def boom(_ref, _text):
            raise RuntimeError("simulated embedding outage")

        # Reach into the notes store and replace the indexer to
        # exercise the resilience path. A real-world scenario would
        # be an OpenAI 5xx; the markdown write must still succeed.
        notes = provider.notes()
        assert isinstance(notes, _FilesystemNotesStore)
        notes.attach_vector_indexer(boom)

        meta = await provider.notes().write(
            NoteDraft(
                title="Resilient",
                body="Markdown should land even when embedding errors.",
                category="topics",
                scope=Scope.SESSION,
            )
        )
        # Read-back must succeed → markdown side committed
        loaded = await provider.notes().read(meta.ref.filename)
        assert loaded is not None
        assert "Markdown should land" in loaded.body
