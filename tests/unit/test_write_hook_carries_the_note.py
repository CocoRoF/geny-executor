"""A note's write hook must offer the note, not just its body.

Production 2026-08-18: freshly written notes showed a blank title in the
catalogue while reconciled ones had it. Two write paths, two shapes —
the auto-vector hook sent ``(ref, body)`` while the host's periodic
reconciliation sent the whole note.

That is not merely a lost field. A vector store whose idempotence digest
covers title/tags/importance will never settle: the hook writes digest A,
the reconciliation rewrites digest B, the next edit writes A again, and
the note re-indexes forever.

Indexers are asked by SIGNATURE, never by try/except, so a genuine
TypeError raised inside an indexer is never mistaken for "wrong arity"
and silently retried with less information.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.memory.providers.file.notes_store import _indexer_takes_note


# ── arity detection ──────────────────────────────────────────────────

def test_a_two_argument_indexer_is_left_alone():
    async def old(ref, text):
        return 1

    assert _indexer_takes_note(old) is False


def test_a_three_argument_indexer_is_offered_the_note():
    async def new(ref, text, note=None):
        return 1

    assert _indexer_takes_note(new) is True


def test_a_bound_method_is_measured_without_self():
    """`signature` on a bound method already omits `self`; unwrapping to
    `__func__` puts it back and made every two-arg store look three-arg."""
    class _Old:
        async def index(self, ref, text):
            return 1

    class _New:
        async def index(self, ref, text, note=None):
            return 1

    assert _indexer_takes_note(_Old().index) is False
    assert _indexer_takes_note(_New().index) is True


def test_star_args_counts_as_room():
    async def flexible(ref, *args):
        return 1

    assert _indexer_takes_note(flexible) is True


def test_an_unreadable_signature_is_treated_as_narrow():
    """Never guess wide: an indexer we cannot measure gets the old shape."""
    assert _indexer_takes_note(object()) is False


# ── end-to-end through the store ─────────────────────────────────────

@pytest.mark.asyncio
async def test_the_write_hook_hands_the_note_to_a_willing_indexer(tmp_path):
    from datetime import timezone

    from geny_executor.memory.provider import NoteDraft, Scope
    from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore
    from geny_executor.memory.providers.file.layout import DirectoryLayout

    got = []

    async def indexer(ref, text, note=None):
        got.append(note)
        return 1

    layout = DirectoryLayout(tmp_path)
    layout.ensure()
    store = _FilesystemNotesStore(layout=layout, tz=timezone.utc, scope=Scope.SESSION)
    store.attach_vector_indexer(indexer)

    await store.write(NoteDraft(title="제목", body="본문입니다", category="daily",
                                tags=["work"]))
    assert got and got[0] is not None, "the hook indexed without the note"
    assert got[0].title == "제목"
    assert list(got[0].tags) == ["work"]


@pytest.mark.asyncio
async def test_an_old_two_argument_indexer_still_works(tmp_path):
    """Every VectorStore in this package takes (ref, text) — the common
    path must not break to serve the new one."""
    from datetime import timezone

    from geny_executor.memory.provider import NoteDraft, Scope
    from geny_executor.memory.providers.file.notes_store import _FilesystemNotesStore
    from geny_executor.memory.providers.file.layout import DirectoryLayout

    calls = []

    async def old_indexer(ref, text):
        calls.append(ref.filename)
        return 1

    layout = DirectoryLayout(tmp_path)
    layout.ensure()
    store = _FilesystemNotesStore(layout=layout, tz=timezone.utc, scope=Scope.SESSION)
    store.attach_vector_indexer(old_indexer)

    await store.write(NoteDraft(title="제목", body="본문입니다", category="daily"))
    assert len(calls) == 1
