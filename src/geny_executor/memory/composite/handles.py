"""Composite-owned `CuratedHandle` / `GlobalHandle` wrappers.

The two non-required handles (`curated`, `global_`) are intentionally
kept off the native `FileMemoryProvider` / `SQLMemoryProvider` because
they are inherently *cross-scope* — a curated note belongs to a
specific user, a global note crosses every session — and a single-root
provider has no business knowing about other scopes.

`CompositeMemoryProvider` is the right place to attach them: the
composite already owns a `LayerRouting` table that can map
`Scope.USER` / `Scope.GLOBAL` to a dedicated underlying provider.
The wrappers in this module take that target provider's `NotesHandle`
plus its (optional) `VectorHandle` and pair them with the curated /
global semantics that the `CuratedHandle` / `GlobalHandle` Protocols
require.

Promotion is the only operation that needs to look at the *source*
side of the composite — copy a note from the session-scope provider
into the user-scope provider, then delete from the source. Both
handles share the same primitive (`_promote`), parameterised by the
target scope.
"""

from __future__ import annotations

import logging
from typing import Optional

from geny_executor.memory.provider import (
    CuratedHandle,
    GlobalHandle,
    MemoryProvider,
    NoteDraft,
    NoteRef,
    NotesHandle,
    Scope,
    VectorHandle,
)

logger = logging.getLogger(__name__)


async def _promote(
    source: MemoryProvider,
    target: MemoryProvider,
    ref: NoteRef,
    target_scope: Scope,
) -> NoteRef:
    """Copy a note from `source` to `target`, then delete the source.

    The new ref carries the requested target scope so callers can
    follow the promoted artefact downstream. Idempotency: if `source`
    no longer has the note (already promoted) a `KeyError` surfaces;
    that is the same shape every other handle uses for missing-note
    failures.
    """
    note = await source.notes().read(ref.filename)
    if note is None:
        raise KeyError(
            f"cannot promote: {ref.filename!r} not found in source provider"
        )
    meta = await target.notes().write(
        NoteDraft(
            title=note.title,
            body=note.body,
            importance=note.importance,
            tags=list(note.tags),
            category=note.category,
            filename=note.ref.filename,
            frontmatter=dict(note.frontmatter),
            scope=target_scope,
        )
    )
    await source.notes().delete(ref.filename)
    return meta.ref.with_scope(target_scope)


class _CompositeCuratedHandle(CuratedHandle):
    """`CuratedHandle` backed by a user-scoped delegate provider.

    The composite resolves `Scope.USER` to a dedicated provider whose
    `notes()` and `vector()` carry the curated knowledge for one user;
    this wrapper exposes those handles directly while owning the
    `user_id` identity and the `promote_from_session` semantics.
    """

    def __init__(
        self,
        *,
        user_id: str,
        target: MemoryProvider,
        source: MemoryProvider,
    ) -> None:
        self._user_id = user_id
        self._target = target
        self._source = source

    @property
    def user_id(self) -> str:
        return self._user_id

    def notes(self) -> NotesHandle:
        return self._target.notes()

    def vector(self) -> Optional[VectorHandle]:
        return self._target.vector()

    async def promote_from_session(self, ref: NoteRef) -> NoteRef:
        return await _promote(self._source, self._target, ref, Scope.USER)


class _CompositeGlobalHandle(GlobalHandle):
    """`GlobalHandle` backed by a global-scoped delegate provider.

    Same shape as `_CompositeCuratedHandle` minus `user_id`: a global
    note has no per-user identity. `promote_from` accepts any source
    ref — the composite hands the source provider in at construction
    time so the wrapper does not need to know the original scope.
    """

    def __init__(
        self,
        *,
        target: MemoryProvider,
        source: MemoryProvider,
    ) -> None:
        self._target = target
        self._source = source

    def notes(self) -> NotesHandle:
        return self._target.notes()

    def vector(self) -> Optional[VectorHandle]:
        return self._target.vector()

    async def promote_from(self, ref: NoteRef) -> NoteRef:
        return await _promote(self._source, self._target, ref, Scope.GLOBAL)


__all__ = [
    "_CompositeCuratedHandle",
    "_CompositeGlobalHandle",
]
