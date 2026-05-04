"""Curated / Global handle behaviour for `CompositeMemoryProvider`.

The composite resolves `Scope.USER` → curated handle and
`Scope.GLOBAL` → global handle by wrapping the underlying delegate's
notes (and optional vector) handles. These tests cover the wrapper
end-to-end without requiring a network embedding client:

  - curated handle exposes `user_id` from composite construction
  - curated `notes()` and `vector()` resolve to the user-scope
    delegate's handles; vector returns ``None`` when the delegate has
    no embedding wired
  - `promote_from_session(ref)` copies the source-scope note into the
    user-scope delegate and deletes the source row
  - global handle behaves the same shape minus `user_id`
  - `descriptor.layers` contains `CURATED` / `GLOBAL` once their
    `scope_providers` slot is populated, so callers can capability-gate
    via the descriptor
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.composite import CompositeMemoryProvider, LayerRouting
from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    Layer,
    NoteDraft,
    Scope,
)
from geny_executor.memory.providers import FileMemoryProvider


async def _build_provider(
    root: Path, *, with_embedding: bool = False, user_id: str = "alice"
) -> CompositeMemoryProvider:
    """Compose a session-scope delegate + a user-scope delegate.

    `with_embedding` flips the local hash embedding on so the curated
    handle's `vector()` returns a usable handle for the auto-vector
    path test; the deterministic SHA-256 backend keeps the test
    network-free.
    """
    embedding = (
        LocalHashEmbeddingClient(model="hash-v1", dimension=64)
        if with_embedding
        else None
    )
    session_root = root / "sessions" / "sess-1"
    user_root = root / "curated" / user_id
    session = FileMemoryProvider(
        root=session_root,
        scope=Scope.SESSION,
        embedding_client=embedding,
    )
    user = FileMemoryProvider(
        root=user_root,
        scope=Scope.USER,
        embedding_client=embedding,
    )
    routing = LayerRouting(
        layers={
            Layer.STM: session,
            Layer.LTM: session,
            Layer.NOTES: session,
            Layer.INDEX: session,
        },
        scope_providers={
            Scope.SESSION: session,
            Scope.USER: user,
        },
    )
    composite = CompositeMemoryProvider(
        routing=routing, user_id=user_id, session_id="sess-1"
    )
    await composite.initialize()
    return composite


class TestCompositeCuratedHandle:
    async def test_user_id_propagates(self, tmp_path: Path) -> None:
        provider = await _build_provider(tmp_path, user_id="alice")
        curated = provider.curated()
        assert curated is not None
        assert curated.user_id == "alice"

    async def test_notes_handle_routes_to_user_delegate(
        self, tmp_path: Path
    ) -> None:
        provider = await _build_provider(tmp_path)
        curated = provider.curated()
        assert curated is not None

        meta = await curated.notes().write(
            NoteDraft(
                title="Curated entry",
                body="Body for alice",
                category="topics",
                scope=Scope.USER,
            )
        )
        # Verify the note materialised under the user-scope root,
        # *not* under the session-scope root.
        user_dir = tmp_path / "curated" / "alice" / "memory"
        session_dir = tmp_path / "sessions" / "sess-1" / "memory"
        assert any(user_dir.rglob(meta.ref.filename))
        assert not any(session_dir.rglob(meta.ref.filename))

    async def test_vector_handle_resolves_when_embedding_present(
        self, tmp_path: Path
    ) -> None:
        provider = await _build_provider(tmp_path, with_embedding=True)
        curated = provider.curated()
        assert curated is not None
        assert curated.vector() is not None

    async def test_vector_handle_none_when_no_embedding(
        self, tmp_path: Path
    ) -> None:
        provider = await _build_provider(tmp_path, with_embedding=False)
        curated = provider.curated()
        assert curated is not None
        assert curated.vector() is None

    async def test_promote_from_session_copies_and_removes(
        self, tmp_path: Path
    ) -> None:
        provider = await _build_provider(tmp_path)

        # Write a note via the session-scope notes handle (the
        # composite's required `notes()` resolves to the session
        # delegate via the routing table).
        session_meta = await provider.notes().write(
            NoteDraft(
                title="Hot insight",
                body="Body to promote.",
                category="insights",
                scope=Scope.SESSION,
                filename="hot-insight.md",
            )
        )
        assert session_meta.ref.scope == Scope.SESSION

        curated = provider.curated()
        assert curated is not None
        promoted_ref = await curated.promote_from_session(session_meta.ref)

        assert promoted_ref.scope == Scope.USER
        # Target now has the file
        in_curated = await curated.notes().read(promoted_ref.filename)
        assert in_curated is not None
        assert in_curated.body == "Body to promote."
        # Source row is gone
        in_session = await provider.notes().read(session_meta.ref.filename)
        assert in_session is None


class TestCompositeGlobalHandle:
    async def test_global_handle_none_without_global_scope(
        self, tmp_path: Path
    ) -> None:
        # The default fixture only registers Scope.USER, so global_()
        # has nothing to wrap.
        provider = await _build_provider(tmp_path)
        assert provider.global_() is None

    async def test_global_handle_resolves_when_scope_provider_set(
        self, tmp_path: Path
    ) -> None:
        # Manually compose a routing that adds a global delegate.
        session = FileMemoryProvider(root=tmp_path / "sess", scope=Scope.SESSION)
        global_ = FileMemoryProvider(root=tmp_path / "glob", scope=Scope.GLOBAL)
        routing = LayerRouting(
            layers={
                Layer.STM: session,
                Layer.LTM: session,
                Layer.NOTES: session,
                Layer.INDEX: session,
            },
            scope_providers={
                Scope.SESSION: session,
                Scope.GLOBAL: global_,
            },
        )
        composite = CompositeMemoryProvider(routing=routing)
        await composite.initialize()

        gh = composite.global_()
        assert gh is not None

        meta = await gh.notes().write(
            NoteDraft(
                title="Shared",
                body="cross-session knowledge",
                category="reference",
                scope=Scope.GLOBAL,
            )
        )
        # Lands under the global delegate's root, not the session one.
        assert any((tmp_path / "glob" / "memory").rglob(meta.ref.filename))


class TestCompositeDescriptorLayers:
    async def test_curated_layer_in_descriptor_when_user_scope_set(
        self, tmp_path: Path
    ) -> None:
        provider = await _build_provider(tmp_path)
        assert Layer.CURATED in provider.descriptor.layers

    async def test_global_layer_in_descriptor_when_global_scope_set(
        self, tmp_path: Path
    ) -> None:
        session = FileMemoryProvider(root=tmp_path / "sess", scope=Scope.SESSION)
        global_ = FileMemoryProvider(root=tmp_path / "glob", scope=Scope.GLOBAL)
        routing = LayerRouting(
            layers={
                Layer.STM: session,
                Layer.LTM: session,
                Layer.NOTES: session,
                Layer.INDEX: session,
            },
            scope_providers={Scope.GLOBAL: global_},
        )
        composite = CompositeMemoryProvider(routing=routing)
        await composite.initialize()
        assert Layer.GLOBAL in composite.descriptor.layers
