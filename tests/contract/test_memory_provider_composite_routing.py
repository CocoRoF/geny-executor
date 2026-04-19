"""Routing-specific tests for `CompositeMemoryProvider`.

Distinct from the contract suite: these exercise the *routing* —
that the composite hands each layer to the right delegate, that
distinct backends survive a snapshot round-trip, that scope-bound
promote actually moves a note between providers, and that the
descriptor merges every delegate's surface correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.composite import CompositeMemoryProvider, LayerRouting
from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    Capability,
    Importance,
    Layer,
    NoteDraft,
    NoteRef,
    RetrievalQuery,
    Scope,
    Turn,
)
from geny_executor.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)


# ── routing validation ──────────────────────────────────────────────


def test_routing_requires_all_four_required_layers():
    file_provider = FileMemoryProvider(root=Path("/tmp/whatever"))
    with pytest.raises(ValueError, match="missing required layers"):
        LayerRouting(layers={Layer.STM: file_provider, Layer.LTM: file_provider})


def test_routing_distinct_providers_dedupes():
    p = FileMemoryProvider(root=Path("/tmp/whatever"))
    routing = LayerRouting(layers={Layer.STM: p, Layer.LTM: p, Layer.NOTES: p, Layer.INDEX: p})
    assert routing.distinct_providers() == [p]


def test_routing_distinct_providers_preserves_order(tmp_path: Path):
    a = EphemeralMemoryProvider()
    b = FileMemoryProvider(root=tmp_path / "b")
    routing = LayerRouting(layers={Layer.STM: a, Layer.LTM: b, Layer.NOTES: b, Layer.INDEX: b})
    distinct = routing.distinct_providers()
    assert distinct == [a, b]


# ── per-layer dispatch ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handles_dispatch_to_correct_delegate(tmp_path: Path):
    stm_delegate = EphemeralMemoryProvider()
    main_delegate = FileMemoryProvider(root=tmp_path / "main")
    await stm_delegate.initialize()
    await main_delegate.initialize()
    routing = LayerRouting(
        layers={
            Layer.STM: stm_delegate,
            Layer.LTM: main_delegate,
            Layer.NOTES: main_delegate,
            Layer.INDEX: main_delegate,
        }
    )
    composite = CompositeMemoryProvider(routing=routing)
    # The STM handle returned by the composite IS the ephemeral STM
    assert composite.stm() is stm_delegate.stm()
    # The Notes handle is the file-provider's notes — different object
    assert composite.notes() is main_delegate.notes()
    assert composite.stm() is not main_delegate.stm()


@pytest.mark.asyncio
async def test_record_turn_lands_in_stm_delegate_only(tmp_path: Path):
    stm_delegate = EphemeralMemoryProvider()
    main_delegate = FileMemoryProvider(root=tmp_path / "main")
    await stm_delegate.initialize()
    await main_delegate.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: stm_delegate,
                Layer.LTM: main_delegate,
                Layer.NOTES: main_delegate,
                Layer.INDEX: main_delegate,
            }
        )
    )
    await composite.record_turn(Turn(role="user", content="hello"))
    # STM owner saw the turn
    assert len(await stm_delegate.stm().recent(10)) == 1
    # The other delegate didn't
    assert len(await main_delegate.stm().recent(10)) == 0


# ── descriptor synthesis ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_descriptor_unions_layers_and_capabilities(tmp_path: Path):
    embed = LocalHashEmbeddingClient(model="test", dimension=32)
    sql_delegate = SQLMemoryProvider(tmp_path / "main.db", embedding_client=embed)
    file_delegate = FileMemoryProvider(root=tmp_path / "session")
    await sql_delegate.initialize()
    await file_delegate.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: file_delegate,
                Layer.LTM: sql_delegate,
                Layer.NOTES: sql_delegate,
                Layer.VECTOR: sql_delegate,
                Layer.INDEX: sql_delegate,
            }
        )
    )
    d = composite.descriptor
    assert {Layer.STM, Layer.LTM, Layer.NOTES, Layer.VECTOR, Layer.INDEX} <= d.layers
    assert Capability.SNAPSHOT in d.capabilities
    assert Capability.REINDEX in d.capabilities
    # Embedding propagates from the wrapped sql delegate
    assert d.embedding is not None
    # Backends aggregate every delegate's BackendInfo entries
    backends = {(info.layer, info.backend) for info in d.backends}
    assert (Layer.STM, "filesystem") in backends
    assert (Layer.NOTES, "sqlite") in backends


@pytest.mark.asyncio
async def test_descriptor_grants_promote_when_scope_router_present(tmp_path: Path):
    session_provider = FileMemoryProvider(root=tmp_path / "session")
    user_provider = FileMemoryProvider(root=tmp_path / "user")
    await session_provider.initialize()
    await user_provider.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: session_provider,
                Layer.LTM: session_provider,
                Layer.NOTES: session_provider,
                Layer.INDEX: session_provider,
            },
            scope_providers={
                Scope.SESSION: session_provider,
                Scope.USER: user_provider,
            },
        )
    )
    assert Capability.PROMOTE in composite.descriptor.capabilities


# ── retrieve composition across backends ────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_pulls_from_each_layer_owner(tmp_path: Path):
    stm_delegate = EphemeralMemoryProvider()
    main_delegate = FileMemoryProvider(root=tmp_path / "main")
    await stm_delegate.initialize()
    await main_delegate.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: stm_delegate,
                Layer.LTM: main_delegate,
                Layer.NOTES: main_delegate,
                Layer.INDEX: main_delegate,
            }
        )
    )
    await composite.record_turn(Turn(role="user", content="quick brown fox"))
    await composite.notes().write(
        NoteDraft(title="alpha", body="fox jumps", importance=Importance.HIGH)
    )
    result = await composite.retrieve(RetrievalQuery(text="fox", max_chars=2000))
    assert result.layer_breakdown.get(Layer.STM, 0) >= 1
    assert result.layer_breakdown.get(Layer.NOTES, 0) >= 1


# ── promote moves notes across scope-bound providers ────────────────


@pytest.mark.asyncio
async def test_promote_copies_note_into_target_scope_provider(tmp_path: Path):
    session_provider = FileMemoryProvider(root=tmp_path / "session")
    user_provider = FileMemoryProvider(root=tmp_path / "user")
    await session_provider.initialize()
    await user_provider.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: session_provider,
                Layer.LTM: session_provider,
                Layer.NOTES: session_provider,
                Layer.INDEX: session_provider,
            },
            scope_providers={
                Scope.SESSION: session_provider,
                Scope.USER: user_provider,
            },
        )
    )
    meta = await composite.notes().write(
        NoteDraft(title="rare-fact", body="42 is the answer", scope=Scope.SESSION)
    )
    new_ref = await composite.promote(meta.ref, Scope.USER)
    assert new_ref.scope == Scope.USER
    # The note now lives in the user provider...
    assert (await user_provider.notes().read(meta.ref.filename)) is not None
    # ...and is gone from the session provider
    assert (await session_provider.notes().read(meta.ref.filename)) is None


@pytest.mark.asyncio
async def test_promote_no_op_when_target_equals_source(tmp_path: Path):
    main = FileMemoryProvider(root=tmp_path / "main")
    await main.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: main,
                Layer.LTM: main,
                Layer.NOTES: main,
                Layer.INDEX: main,
            }
        )
    )
    meta = await composite.notes().write(NoteDraft(title="x", body="y", scope=Scope.SESSION))
    again = await composite.promote(meta.ref, Scope.SESSION)
    assert again.scope == Scope.SESSION


@pytest.mark.asyncio
async def test_promote_falls_back_to_source_when_no_target_provider(tmp_path: Path):
    main = FileMemoryProvider(root=tmp_path / "main")
    await main.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: main,
                Layer.LTM: main,
                Layer.NOTES: main,
                Layer.INDEX: main,
            }
        )
    )
    meta = await composite.notes().write(NoteDraft(title="x", body="y", scope=Scope.SESSION))
    new_ref = await composite.promote(meta.ref, Scope.USER)
    # Falls through to source's promote — same filename, new scope tag
    assert new_ref.scope == Scope.USER
    assert new_ref.filename == meta.ref.filename


# ── snapshot round-trip with multiple backends ──────────────────────


@pytest.mark.asyncio
async def test_snapshot_round_trip_across_two_backends(tmp_path: Path):
    stm_delegate_a = EphemeralMemoryProvider()
    main_delegate_a = FileMemoryProvider(root=tmp_path / "a-main")
    await stm_delegate_a.initialize()
    await main_delegate_a.initialize()
    composite_a = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: stm_delegate_a,
                Layer.LTM: main_delegate_a,
                Layer.NOTES: main_delegate_a,
                Layer.INDEX: main_delegate_a,
            }
        )
    )
    await composite_a.record_turn(Turn(role="user", content="hello"))
    await composite_a.notes().write(NoteDraft(title="alpha", body="snap me"))
    snap = await composite_a.snapshot()
    assert snap.size_bytes > 0
    assert snap.checksum

    stm_delegate_b = EphemeralMemoryProvider()
    main_delegate_b = FileMemoryProvider(root=tmp_path / "b-main")
    await stm_delegate_b.initialize()
    await main_delegate_b.initialize()
    composite_b = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: stm_delegate_b,
                Layer.LTM: main_delegate_b,
                Layer.NOTES: main_delegate_b,
                Layer.INDEX: main_delegate_b,
            }
        )
    )
    await composite_b.restore(snap)
    notes = await composite_b.notes().list()
    assert any(meta.title == "alpha" for meta in notes)
    recent = await composite_b.stm().recent(10)
    assert any("hello" in str(t.content) for t in recent)


@pytest.mark.asyncio
async def test_snapshot_tampered_checksum_raises(tmp_path: Path):
    main = FileMemoryProvider(root=tmp_path / "main")
    await main.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: main,
                Layer.LTM: main,
                Layer.NOTES: main,
                Layer.INDEX: main,
            }
        )
    )
    await composite.notes().write(NoteDraft(title="x", body="y"))
    snap = await composite.snapshot()
    snap.checksum = "deadbeef" * 8

    fresh_main = FileMemoryProvider(root=tmp_path / "fresh")
    await fresh_main.initialize()
    fresh = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: fresh_main,
                Layer.LTM: fresh_main,
                Layer.NOTES: fresh_main,
                Layer.INDEX: fresh_main,
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        await fresh.restore(snap)


@pytest.mark.asyncio
async def test_restore_rejects_non_composite_payload(tmp_path: Path):
    main = FileMemoryProvider(root=tmp_path / "main")
    await main.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: main,
                Layer.LTM: main,
                Layer.NOTES: main,
                Layer.INDEX: main,
            }
        )
    )
    file_snap = await main.snapshot()
    with pytest.raises(ValueError, match="composite"):
        await composite.restore(file_snap)


# ── vector wiring through composite ─────────────────────────────────


@pytest.mark.asyncio
async def test_vector_handle_visible_when_routed(tmp_path: Path):
    embed = LocalHashEmbeddingClient(model="test", dimension=32)
    sql_delegate = SQLMemoryProvider(tmp_path / "main.db", embedding_client=embed)
    file_delegate = FileMemoryProvider(root=tmp_path / "session")
    await sql_delegate.initialize()
    await file_delegate.initialize()
    composite = CompositeMemoryProvider(
        routing=LayerRouting(
            layers={
                Layer.STM: file_delegate,
                Layer.LTM: sql_delegate,
                Layer.NOTES: sql_delegate,
                Layer.VECTOR: sql_delegate,
                Layer.INDEX: sql_delegate,
            }
        )
    )
    assert composite.vector() is sql_delegate.vector()
    # Auto-index on record_execution surfaces a vector chunk in retrieve
    await composite.notes().write(NoteDraft(title="alpha", body="fox", filename="alpha.md"))
    await composite.vector().index(
        NoteRef(filename="alpha.md", scope=Scope.SESSION, backend="sqlite"), "fox"
    )
    result = await composite.retrieve(RetrievalQuery(text="fox", max_chars=4000))
    assert result.layer_breakdown.get(Layer.VECTOR, 0) >= 1
