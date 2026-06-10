"""Vector-layer auth circuit breaker (audit §2.6).

The live prod incident: a revoked embedding key caused every note
write to retry the embed call and log a full traceback — per write,
forever, with no way to tell from the provider whether vectors were
even functional. The breaker contract pinned here:

  - 3 consecutive 'auth'-classified `EmbeddingError`s trip a
    session-long disabled flag; exactly ONE warning is logged across
    the whole spam scenario (the trip message);
  - markdown writes keep landing on disk throughout — the vector
    layer is an enhancement, never a gatekeeper;
  - after the trip, vector ops no-op fast (no embed calls, no logs);
  - 'transient' (and 'quota') failures never trip the breaker;
  - a success between auth failures resets the consecutive counter
    (the threshold exists to absorb stray 401s during key rotation);
  - the state is observable: `provider.vector_disabled` and
    `descriptor.metadata['vector_disabled']`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

from geny_executor.memory.embedding.client import EmbeddingError
from geny_executor.memory.provider import EmbeddingDescriptor, NoteDraft, Scope
from geny_executor.memory.providers import FileMemoryProvider


class _ScriptedEmbeddingClient:
    """EmbeddingClient whose embed() outcomes are a scripted sequence.

    Each entry in `script` is either 'ok' or an EmbeddingError
    category to raise. Once the script is exhausted the client keeps
    succeeding. `calls` counts how many times embed() was actually
    reached — the breaker's no-op guarantee is asserted against it.
    """

    def __init__(self, script: Sequence[str], dimension: int = 8) -> None:
        self._script = list(script)
        self._dimension = dimension
        self.calls = 0

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return EmbeddingDescriptor(
            provider="scripted",
            model="scripted-v1",
            dimension=self._dimension,
            metric="cosine",
            api_key_present=True,
        )

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self.calls += 1
        outcome = self._script.pop(0) if self._script else "ok"
        if outcome != "ok":
            raise EmbeddingError(f"scripted {outcome} failure", category=outcome)
        return [[1.0] * self._dimension for _ in texts]

    async def close(self) -> None:
        return None


async def _provider_with(client: _ScriptedEmbeddingClient, root: Path) -> FileMemoryProvider:
    provider = FileMemoryProvider(root=root, scope=Scope.SESSION, embedding_client=client)
    await provider.initialize()
    return provider


def _draft(i: int) -> NoteDraft:
    return NoteDraft(
        title=f"note {i}",
        body=f"body of note number {i}",
        category="topics",
        scope=Scope.SESSION,
    )


# ── auth trips the breaker ──────────────────────────────────────────


async def test_three_auth_failures_trip_breaker_markdown_persists(
    tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.DEBUG, logger="geny_executor.memory")
    client = _ScriptedEmbeddingClient(["auth"] * 10)
    provider = await _provider_with(client, tmp_path)

    for i in range(5):
        await provider.notes().write(_draft(i))

    # Breaker tripped after the 3rd consecutive auth failure: writes
    # 4 and 5 never reached the network.
    assert client.calls == 3
    assert provider.vector_disabled is True
    assert provider._vector.vector_disabled is True
    assert "auth" in (provider._vector.disabled_reason or "")

    # Markdown stayed authoritative throughout the spam.
    notes = await provider.notes().list(category="topics")
    assert len(notes) == 5

    # Exactly ONE warning across the whole scenario — the trip
    # message. Per-write tracebacks (the prod 401-spam) are gone.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {warnings!r}"
    assert "vector indexing disabled" in warnings[0].message
    assert "credentials" in warnings[0].message


async def test_tripped_breaker_noops_search_and_reindex(tmp_path: Path) -> None:
    client = _ScriptedEmbeddingClient(["auth"] * 3)
    provider = await _provider_with(client, tmp_path)

    for i in range(3):
        await provider.notes().write(_draft(i))
    assert provider.vector_disabled is True
    calls_at_trip = client.calls

    # Post-trip vector ops degrade silently and fast.
    assert await provider.vector().search("anything", top_k=3) == []
    assert await provider.vector().index_batch([]) == 0
    receipt = await provider.vector().reindex()
    assert receipt.chunks_to_reindex == 0
    assert receipt.metadata.get("vector_disabled") is True
    assert client.calls == calls_at_trip


async def test_descriptor_reflects_breaker_state(tmp_path: Path) -> None:
    client = _ScriptedEmbeddingClient(["auth"] * 3)
    provider = await _provider_with(client, tmp_path)

    assert provider.descriptor.metadata.get("vector_disabled") is False

    for i in range(3):
        await provider.notes().write(_draft(i))

    descriptor = provider.descriptor
    assert descriptor.metadata.get("vector_disabled") is True
    assert "auth" in descriptor.metadata.get("vector_disabled_reason", "")


# ── transient / quota never trip ────────────────────────────────────


async def test_transient_failures_do_not_trip(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="geny_executor.memory")
    client = _ScriptedEmbeddingClient(["transient"] * 5)
    provider = await _provider_with(client, tmp_path)

    for i in range(5):
        await provider.notes().write(_draft(i))

    # Retry-next-time stays the policy: every write attempted the call.
    assert client.calls == 5
    assert provider.vector_disabled is False

    # First occurrence warns; repeats drop to debug.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "transient" in warnings[0].message

    # The next success indexes normally — nothing was latched.
    await provider.notes().write(_draft(99))
    results = await provider.vector().search("body of note number 99", top_k=3)
    assert len(results) >= 1


async def test_quota_failures_do_not_trip(tmp_path: Path) -> None:
    client = _ScriptedEmbeddingClient(["quota"] * 5)
    provider = await _provider_with(client, tmp_path)

    for i in range(5):
        await provider.notes().write(_draft(i))

    assert client.calls == 5
    assert provider.vector_disabled is False


async def test_unknown_failures_do_not_trip_and_keep_traceback(
    tmp_path: Path, caplog
) -> None:
    """Unclassified failures keep the conservative path: no breaker
    motion, WARNING with traceback retained (it's the only diagnostic
    for a genuinely unexpected error)."""
    caplog.set_level(logging.DEBUG, logger="geny_executor.memory")
    client = _ScriptedEmbeddingClient(["unknown"] * 4)
    provider = await _provider_with(client, tmp_path)

    for i in range(4):
        await provider.notes().write(_draft(i))

    assert client.calls == 4
    assert provider.vector_disabled is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 4
    assert all(r.exc_info for r in warnings)


# ── consecutive means consecutive ───────────────────────────────────


async def test_success_resets_consecutive_auth_counter(tmp_path: Path) -> None:
    """Two stray 401s, a success, two more 401s — never three in a
    row, so the breaker must stay closed (key-rotation blips must not
    permanently degrade a session)."""
    client = _ScriptedEmbeddingClient(["auth", "auth", "ok", "auth", "auth", "ok"])
    provider = await _provider_with(client, tmp_path)

    for i in range(6):
        await provider.notes().write(_draft(i))

    assert client.calls == 6
    assert provider.vector_disabled is False
