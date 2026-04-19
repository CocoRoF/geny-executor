"""C5 — Embedding migration safety.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[4]`):
Switching embedding provider triggers compatibility_check, surfaces a
reindex_plan, and only after explicit approval performs the reindex
in the background while emitting `memory.reindexed`. Silent rebuild
is forbidden.

C5 gates on the vector-capable native providers (file + sql). The
ephemeral + adapter providers do not expose an EmbeddingDescriptor,
so they are skipped for this test — the cross-backend surface that
C5 exercises is explicitly the embedding-pluggable one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.embedding.local import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    EmbeddingDescriptor,
    Importance,
    Layer,
    MemoryProvider,
    NoteDraft,
    Scope,
)
from geny_executor.memory.providers import FileMemoryProvider, SQLMemoryProvider


def _vector_capable_providers(tmp_path: Path):
    """Yield (name, factory) for each provider we can wire a vector
    layer into. Factories bake in a 64-dim local embedding client so
    the descriptor's compatibility_check has something concrete to
    compare against.
    """

    async def _file_factory(root: Path) -> MemoryProvider:
        provider = FileMemoryProvider(
            root=root / "file_root",
            embedding_client=LocalHashEmbeddingClient(model="c5-local", dimension=64),
        )
        await provider.initialize()
        return provider

    async def _sql_factory(root: Path) -> MemoryProvider:
        provider = SQLMemoryProvider(
            dsn=str(root / "c5.db"),
            embedding_client=LocalHashEmbeddingClient(model="c5-local", dimension=64),
        )
        await provider.initialize()
        return provider

    yield "file", _file_factory
    yield "sql", _sql_factory


async def test_c5_embedding_swap_requires_explicit_approval(
    tmp_path: Path,
    registered_providers,
):
    if not registered_providers:
        pytest.skip("no providers registered for C5")

    new_embedding = EmbeddingDescriptor(
        provider="local",
        model="c5-local-upgraded",
        dimension=128,  # dimension change — must surface a reindex plan
    )

    ran_at_least_one = False
    for name, factory in _vector_capable_providers(tmp_path):
        root = tmp_path / f"c5-{name}"
        root.mkdir()
        provider = await factory(root)
        try:
            assert provider.vector() is not None, (
                f"{name}: test precondition — vector handle must be wired"
            )
            assert Layer.VECTOR in provider.descriptor.layers, (
                f"{name}: descriptor must declare VECTOR when embedding wired"
            )

            # Seed one indexable note so the reindex path has work to do.
            meta = await provider.notes().write(
                NoteDraft(
                    title="c5-probe",
                    body="Baseline content for the C5 reindex probe.",
                    importance=Importance.MEDIUM,
                    category="insights",
                    scope=Scope.SESSION,
                )
            )
            await provider.vector().index(meta.ref, "Baseline content for the C5 reindex probe.")

            # 1. Silent rebuild is forbidden — compatibility_check surfaces a plan.
            plan = provider.descriptor.compatibility_check(new_embedding)
            assert plan is not None, (
                f"{name}: dimension change must produce a ReindexPlan, not None"
            )
            assert plan.requires_explicit_approval, (
                f"{name}: C5 forbids silent rebuild — plan must require approval"
            )
            assert plan.layer == Layer.VECTOR

            # 2. Without approval, the original index still works.
            results = await provider.vector().search("baseline", top_k=1)
            assert results, f"{name}: pre-approval vector search should still return hits"

            # 3. Explicit approval → reindex() executes and returns a fresh plan.
            applied = await provider.vector().reindex()
            assert applied.layer == Layer.VECTOR
            assert applied.chunks_to_reindex >= 1, f"{name}: reindex() reported no chunks rebuilt"

            ran_at_least_one = True
        finally:
            await provider.close()

    assert ran_at_least_one, "C5 needs at least one vector-capable provider"
