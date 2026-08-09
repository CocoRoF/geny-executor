"""Deleting a note must delete its index row — effect-proving tests.

Writes have had an auto-vector hook since the beginning; deletes never did.
So a deleted note kept its vector row: search went on scoring a memory that
no longer existed, returned it, and then could not resolve a body for it. A
forward scan at boot cannot find these either — iterating the files that
exist never visits the ones that don't — so they only accumulate. One
production vault reached 36% of its index being notes whose files were gone.
"""

from __future__ import annotations

from datetime import timezone
from typing import List

import pytest

from geny_executor.memory.provider import Importance, NoteDraft, Scope
from geny_executor.memory.providers.file.layout import DirectoryLayout
from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore


class _Vector:
    def __init__(self) -> None:
        self.indexed: List[str] = []
        self.removed: List[str] = []
        self.fail = False

    async def index(self, ref, text) -> int:
        self.indexed.append(ref.filename)
        return 1

    async def remove(self, ref) -> bool:
        if self.fail:
            raise RuntimeError("index unavailable")
        self.removed.append(ref.filename)
        return True


@pytest.fixture
def store(tmp_path):
    layout = DirectoryLayout(tmp_path)
    layout.ensure()
    s = _FilesystemNotesStore(layout=layout, tz=timezone.utc, scope=Scope.SESSION)
    return s


async def _write(store, name: str) -> str:
    meta = await store.write(NoteDraft(
        title=name, body=f"{name} 본문", category="observations",
        importance=Importance.LOW,
    ))
    return meta.ref.filename


@pytest.mark.asyncio
async def test_deleting_a_note_removes_it_from_the_index(store):
    """THE property."""
    vec = _Vector()
    store.attach_vector_indexer(vec.index)
    store.attach_vector_remover(vec.remove)
    filename = await _write(store, "사라질 노트")
    assert vec.indexed == [filename]

    assert await store.delete(filename) is True

    assert vec.removed == [filename], "the index kept a note that is gone"


@pytest.mark.asyncio
async def test_deleting_a_missing_note_touches_nothing(store):
    vec = _Vector()
    store.attach_vector_remover(vec.remove)

    assert await store.delete("nope.md") is False
    assert vec.removed == []


@pytest.mark.asyncio
async def test_surviving_notes_keep_their_index_rows(store):
    vec = _Vector()
    store.attach_vector_indexer(vec.index)
    store.attach_vector_remover(vec.remove)
    keep = await _write(store, "남을 노트")
    drop = await _write(store, "지울 노트")

    await store.delete(drop)

    assert vec.removed == [drop]
    assert keep not in vec.removed


@pytest.mark.asyncio
async def test_a_failing_index_does_not_fail_the_delete(store):
    """The markdown delete already happened and is authoritative. Raising
    here would report failure for work that succeeded, and the leftover row
    is exactly what boot reconciliation is for."""
    vec = _Vector()
    vec.fail = True
    store.attach_vector_indexer(vec.index)
    store.attach_vector_remover(vec.remove)
    filename = await _write(store, "실패 노트")

    assert await store.delete(filename) is True
    assert await store.read(filename) is None, "the note survived the delete"


@pytest.mark.asyncio
async def test_a_store_with_no_remover_still_deletes(store):
    """Vector-less deployments must not break."""
    filename = await _write(store, "벡터 없음")
    assert await store.delete(filename) is True
    assert await store.read(filename) is None


@pytest.mark.asyncio
async def test_the_remover_can_be_detached(store):
    vec = _Vector()
    store.attach_vector_remover(vec.remove)
    store.attach_vector_remover(None)
    filename = await _write(store, "분리됨")

    await store.delete(filename)

    assert vec.removed == []
