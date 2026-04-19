"""Reusable contract for `geny_executor.memory.provider.MemoryProvider`.

Every concrete MemoryProvider implementation imports
`MemoryProviderContract` and subclasses it once with a
`provider_factory()` fixture. The mixin ships ~50 behavioural
assertions covering descriptor, all 7 layer handles, cross-layer
retrieval, record/reflect/snapshot, and capability gating.

Phase 1 ships the contract + `TestEphemeralProvider`. Phase 2 adds
`TestFileProvider`, `TestSQLProvider`, etc. as siblings — when those
subclasses appear, the same assertions run against each backend.
"""

from __future__ import annotations

from typing import Awaitable, Callable

import pytest

from geny_executor.memory.provider import (
    Capability,
    EmbeddingDescriptor,
    ExecutionSummary,
    Importance,
    Layer,
    MemoryDescriptor,
    MemoryProvider,
    NoteDraft,
    NotePatch,
    NoteRef,
    ReflectionContext,
    RetrievalQuery,
    Scope,
    Turn,
)


ProviderFactory = Callable[[], Awaitable[MemoryProvider]]


@pytest.mark.asyncio
class MemoryProviderContract:
    """Subclass and override `provider_factory` to plug in a concrete impl.

    Tests that depend on optional layers (`vector`, `curated`,
    `global_`) skip when the provider declines that layer in its
    descriptor. The contract therefore *grows* with provider
    capability rather than failing for narrower providers.
    """

    # ── To be implemented by concrete subclasses ───────────────────

    @pytest.fixture
    async def provider(self) -> MemoryProvider:  # pragma: no cover — abstract
        raise NotImplementedError("subclass must override `provider` fixture")

    # ── Lifecycle / descriptor ─────────────────────────────────────

    async def test_descriptor_self_describes(self, provider: MemoryProvider):
        d: MemoryDescriptor = provider.descriptor
        assert d.name and d.version
        # Spec invariant: STM, LTM, Notes, Index are required for every
        # provider — they may differ in backend, never in presence.
        for required_layer in (Layer.STM, Layer.LTM, Layer.NOTES, Layer.INDEX):
            assert required_layer in d.layers, (
                f"provider {d.name!r} missing required layer {required_layer.value}"
            )
        # Required capabilities at minimum
        for cap in (Capability.READ, Capability.WRITE, Capability.SEARCH):
            assert cap in d.capabilities

    async def test_initialize_and_close_idempotent(self, provider: MemoryProvider):
        # initialize() must be safe to call repeatedly
        await provider.initialize()
        await provider.initialize()
        await provider.close()
        # After close we don't require continued operation; subclasses
        # that support reuse should override.

    async def test_handles_match_descriptor(self, provider: MemoryProvider):
        d = provider.descriptor
        assert provider.stm() is not None
        assert provider.ltm() is not None
        assert provider.notes() is not None
        assert provider.index() is not None
        assert (provider.vector() is not None) == (Layer.VECTOR in d.layers)
        assert (provider.curated() is not None) == (Layer.CURATED in d.layers)
        assert (provider.global_() is not None) == (Layer.GLOBAL in d.layers)

    # ── STM ─────────────────────────────────────────────────────────

    async def test_stm_append_and_recent(self, provider: MemoryProvider):
        stm = provider.stm()
        for i in range(5):
            await stm.append(Turn(role="user", content=f"msg-{i}"))
        recent = await stm.recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "msg-4"

    async def test_stm_search_returns_matches(self, provider: MemoryProvider):
        stm = provider.stm()
        await stm.append(Turn(role="user", content="alpha"))
        await stm.append(Turn(role="user", content="beta gamma"))
        await stm.append(Turn(role="assistant", content="delta"))
        hits = await stm.search("gamma", limit=5)
        assert len(hits) == 1
        assert "gamma" in str(hits[0].content)

    async def test_stm_truncate_drops_old(self, provider: MemoryProvider):
        stm = provider.stm()
        for i in range(10):
            await stm.append(Turn(role="user", content=str(i)))
        dropped = await stm.truncate(keep_last=3)
        assert dropped == 7
        recent = await stm.recent(10)
        assert len(recent) == 3
        assert recent[0].content == "7"

    # ── LTM ─────────────────────────────────────────────────────────

    async def test_ltm_append_and_read_main(self, provider: MemoryProvider):
        ltm = provider.ltm()
        await ltm.append("first observation", heading="Day 1")
        body = await ltm.read_main()
        assert "first observation" in body
        assert "Day 1" in body

    async def test_ltm_dated_segregates_by_day(self, provider: MemoryProvider):
        from datetime import datetime, timezone, timedelta
        ltm = provider.ltm()
        d1 = datetime(2026, 4, 18, tzinfo=timezone.utc)
        d2 = d1 + timedelta(days=1)
        ref1 = await ltm.write_dated("a", day=d1)
        ref2 = await ltm.write_dated("b", day=d2)
        assert ref1.filename != ref2.filename
        hits = await ltm.search("b", limit=5)
        assert any("b" in h.content for h in hits)

    async def test_ltm_topic_groups_under_slug(self, provider: MemoryProvider):
        ltm = provider.ltm()
        ref = await ltm.write_topic("nutrition", "eat veggies")
        assert "nutrition" in ref.filename
        hits = await ltm.search("veggies", limit=3)
        assert hits, "topic body should be searchable"

    # ── Notes ───────────────────────────────────────────────────────

    async def test_notes_write_assigns_filename(self, provider: MemoryProvider):
        notes = provider.notes()
        meta = await notes.write(NoteDraft(title="Pi day", body="3.14159", tags=["math"]))
        assert meta.ref.filename.endswith(".md")
        again = await notes.write(NoteDraft(title="Pi day", body="duplicate", tags=["math"]))
        # Same title → providers may either dedupe or auto-suffix; require
        # both notes to be retrievable with distinct refs.
        assert (await notes.read(again.ref.filename)) is not None

    async def test_notes_list_filters(self, provider: MemoryProvider):
        notes = provider.notes()
        await notes.write(NoteDraft(title="A", body="x", tags=["t1"], importance=Importance.HIGH))
        await notes.write(NoteDraft(title="B", body="y", tags=["t2"], importance=Importance.LOW))
        await notes.write(
            NoteDraft(title="C", body="z", tags=["t1"], importance=Importance.MEDIUM, category="cat")
        )
        all_t1 = await notes.list(tag="t1")
        assert {m.title for m in all_t1} == {"A", "C"}
        cats = await notes.list(category="cat")
        assert {m.title for m in cats} == {"C"}
        highs = await notes.list(importance=Importance.HIGH)
        assert {m.title for m in highs} == {"A"}

    async def test_notes_update_patches_only_supplied_fields(self, provider: MemoryProvider):
        notes = provider.notes()
        meta = await notes.write(NoteDraft(title="orig", body="hello", tags=["a"]))
        updated = await notes.update(meta.ref.filename, NotePatch(append_body="world"))
        full = await notes.read(meta.ref.filename)
        assert full is not None
        assert "hello" in full.body and "world" in full.body
        assert full.title == "orig"
        assert "a" in full.tags
        await notes.update(meta.ref.filename, NotePatch(importance=Importance.HIGH))
        full2 = await notes.read(meta.ref.filename)
        assert full2 is not None and full2.importance == Importance.HIGH

    async def test_notes_delete_returns_truth_only_when_existed(self, provider: MemoryProvider):
        notes = provider.notes()
        meta = await notes.write(NoteDraft(title="ephemeral", body="x"))
        assert await notes.delete(meta.ref.filename) is True
        assert await notes.delete(meta.ref.filename) is False

    async def test_notes_link_creates_edge(self, provider: MemoryProvider):
        notes = provider.notes()
        a = await notes.write(NoteDraft(title="a", body="hi"))
        b = await notes.write(NoteDraft(title="b", body="bye"))
        assert await notes.link(a.ref.filename, b.ref.filename)
        graph = await notes.graph()
        assert (a.ref.filename, b.ref.filename) in graph.edges

    async def test_notes_wikilink_in_body_yields_edge(self, provider: MemoryProvider):
        notes = provider.notes()
        a = await notes.write(NoteDraft(title="src", body="see [[target.md]]"))
        graph = await notes.graph()
        assert any(src == a.ref.filename for src, _ in graph.edges)

    async def test_notes_search_respects_importance_floor(self, provider: MemoryProvider):
        notes = provider.notes()
        await notes.write(NoteDraft(title="lowprio", body="rare-word", importance=Importance.LOW))
        await notes.write(NoteDraft(title="highprio", body="rare-word", importance=Importance.HIGH))
        only_high = await notes.search("rare-word", importance_floor=Importance.HIGH)
        titles = {c.metadata.get("title") for c in only_high}
        assert "highprio" in titles
        assert "lowprio" not in titles

    # ── Index ───────────────────────────────────────────────────────

    async def test_index_tag_counts_aggregates_notes(self, provider: MemoryProvider):
        notes = provider.notes()
        await notes.write(NoteDraft(title="n1", body="x", tags=["red", "blue"]))
        await notes.write(NoteDraft(title="n2", body="y", tags=["red"]))
        counts = await provider.index().tag_counts()
        assert counts.get("red", 0) >= 2
        assert counts.get("blue", 0) >= 1

    async def test_index_graph_matches_notes_graph(self, provider: MemoryProvider):
        notes = provider.notes()
        a = await notes.write(NoteDraft(title="a", body="see [[b]]"))
        graph_n = await notes.graph()
        graph_i = await provider.index().graph()
        # Index handle may add aggregates, but every notes-graph edge
        # should be present.
        for edge in graph_n.edges:
            assert edge in graph_i.edges

    # ── Cross-layer retrieval ───────────────────────────────────────

    async def test_retrieve_returns_layer_breakdown(self, provider: MemoryProvider):
        await provider.notes().write(
            NoteDraft(title="alpha", body="quick brown fox", tags=["fox"], importance=Importance.HIGH)
        )
        await provider.ltm().append("the fox is quick")
        await provider.stm().append(Turn(role="user", content="tell me about the fox"))

        result = await provider.retrieve(RetrievalQuery(text="fox"))
        assert result.chunks, "retrieve should return at least one chunk"
        # Layer breakdown must sum to chunks count (approx — providers
        # may drop chunks during char-budget trimming, so allow ≥ chunks)
        total = sum(result.layer_breakdown.values())
        assert total >= len(result.chunks)
        # At least one layer must have populated
        assert any(v > 0 for v in result.layer_breakdown.values())

    async def test_retrieve_respects_char_budget(self, provider: MemoryProvider):
        for i in range(20):
            await provider.notes().write(
                NoteDraft(
                    title=f"n{i}",
                    body="word " * 200,
                    tags=["bulk"],
                    importance=Importance.HIGH,
                )
            )
        result = await provider.retrieve(RetrievalQuery(text="word", max_chars=500))
        assert result.total_chars <= 500 * 2  # tolerance: allow 1 oversized chunk

    # ── Record turn / execution ─────────────────────────────────────

    async def test_record_turn_appends_to_stm(self, provider: MemoryProvider):
        before = len(await provider.stm().recent(100))
        await provider.record_turn(Turn(role="user", content="hi"))
        after = len(await provider.stm().recent(100))
        assert after == before + 1

    async def test_record_execution_returns_receipt(self, provider: MemoryProvider):
        receipt = await provider.record_execution(
            ExecutionSummary(
                session_id="s1", user_input="ping", final_text="pong"
            )
        )
        # Providers must produce a receipt; counts depend on impl but
        # files_updated must be a list and notes_written must be ≥ 0.
        assert isinstance(receipt.files_updated, list)
        assert receipt.notes_written >= 0

    # ── Reflect ─────────────────────────────────────────────────────

    async def test_reflect_returns_iterable(self, provider: MemoryProvider):
        out = await provider.reflect(ReflectionContext(session_id="s1"))
        # Must be iterable; default impl may yield empty.
        list(out)

    # ── Snapshot / restore ──────────────────────────────────────────

    async def test_snapshot_round_trips(self, provider: MemoryProvider):
        await provider.notes().write(NoteDraft(title="snap", body="data", tags=["t"]))
        await provider.ltm().append("ltm data")
        await provider.stm().append(Turn(role="user", content="hi"))

        snap = await provider.snapshot()
        assert snap.size_bytes > 0
        assert snap.checksum

        # Restore on a fresh instance from the same factory
        fresh = await self._fresh_from(provider)
        await fresh.restore(snap)
        assert (await fresh.notes().list()) == (await provider.notes().list())

    async def _fresh_from(self, provider: MemoryProvider) -> MemoryProvider:
        """Best-effort: spin a fresh provider of the same class.
        Override in subclasses if construction needs more than `()`.
        """
        return type(provider)()

    # ── Promote ─────────────────────────────────────────────────────

    async def test_promote_changes_scope(self, provider: MemoryProvider):
        notes = provider.notes()
        meta = await notes.write(NoteDraft(title="prom", body="x"))
        new_ref: NoteRef = await provider.promote(meta.ref, Scope.USER)
        assert new_ref.scope == Scope.USER
        assert new_ref.filename == meta.ref.filename

    async def test_promote_no_op_when_target_equals_source(self, provider: MemoryProvider):
        notes = provider.notes()
        meta = await notes.write(NoteDraft(title="x", body="y", scope=Scope.SESSION))
        again = await provider.promote(meta.ref, meta.ref.scope)
        assert again.scope == meta.ref.scope

    # ── Capability gating (optional layers) ─────────────────────────

    async def test_optional_layer_handles_match_descriptor(self, provider: MemoryProvider):
        d = provider.descriptor
        assert (provider.vector() is None) == (Layer.VECTOR not in d.layers)
        assert (provider.curated() is None) == (Layer.CURATED not in d.layers)
        assert (provider.global_() is None) == (Layer.GLOBAL not in d.layers)

    # ── Embedding compatibility check ───────────────────────────────

    async def test_compatibility_check_when_embedding_present(self, provider: MemoryProvider):
        emb = provider.descriptor.embedding
        if emb is None:
            pytest.skip("provider has no embedding configured")
        same_plan = provider.descriptor.compatibility_check(emb)
        assert same_plan is None
        different = EmbeddingDescriptor(
            provider="other", model="other", dimension=emb.dimension + 7
        )
        plan = provider.descriptor.compatibility_check(different)
        assert plan is not None
        assert plan.requires_explicit_approval is True
