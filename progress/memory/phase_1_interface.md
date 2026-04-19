# Phase 1 — Interface (memory initiative)

> **Started**: 2026-04-19
> **Closed**: 2026-04-19
> **Owner**: memory initiative
> **Target tag**: `v0.14.0`
> **Predecessor**: `progress/memory/phase_0_spec_freeze.md`
> **Gate**: G1 — `MemoryProvider` Protocol lands, `EphemeralMemoryProvider`
> conforms, Stage 2 and Stage 15 consume the provider, contract suite
> is green, C1 criterion has its runtime foundation (still skipped
> until the C-test itself flips — activation happens alongside Phase 2
> wiring, but the underlying capability is now implementable).

---

## Summary

Phase 1 delivers the **executor-native memory contract** — the
unified surface every future provider (file, SQL, composite, adapter)
must conform to. No Geny code is touched. The goal is a strong,
self-standing interface that can host the full set of memory
behaviours (layers, capabilities, retrieval composition, reflection,
snapshot/restore, promotion) as a single Protocol.

Scope in this phase:

1. `geny_executor.memory.provider` — `MemoryProvider` Protocol, 7
   layer handles (STM/LTM/Notes/Vector/Curated/Global/Index), and the
   supporting dataclasses/enums that formalise the 4-axis model
   (Layer × Capability × Backend × Scope) and typed events from the
   spec.
2. `geny_executor.memory.providers.ephemeral.EphemeralMemoryProvider`
   — the first concrete, dependency-free conformance. Doubles as the
   reference implementation and the default test fixture.
3. Stage 2 (Context) and Stage 15 (Memory) **rewiring** to accept an
   optional `provider` (and `hooks` for Stage 15) kwarg. Provider
   path is additive to the legacy strategy path so the 695-test
   legacy suite keeps passing.
4. `tests/contract/` — `MemoryProviderContract` mixin with 28
   reusable behavioural assertions, plus the ephemeral subclass.
   Every future provider reuses this exact mixin; divergence is a
   test failure, not a style preference.
5. Public API exposure via `geny_executor.memory.__init__` (39
   symbols) and a version bump to `0.14.0`.

Phase 1 is deliberately **additive**: the provider path runs *next to*
the legacy retriever/strategy/persistence triplet inside the stages.
Phase 3 will delete the legacy path once C7 (adapter parity) is
green.

## Changes

### `src/geny_executor/memory/provider.py` (new, 812 lines)

The unified contract. Key pieces, in declaration order:

- **Enums** — `Layer` (STM/LTM/NOTES/VECTOR/INDEX/CURATED/GLOBAL),
  `Capability` (READ/WRITE/SEARCH/LINK/PROMOTE/REINDEX/SNAPSHOT/
  REFLECT/SUMMARIZE), `Scope` (EPHEMERAL/SESSION/USER/TENANT/GLOBAL),
  `Importance` (CRITICAL/HIGH/MEDIUM/LOW with a `.boost` property:
  2.0/1.5/1.0/0.5, matching `MEMORY_SPEC.yaml → retrieval.importance_boosts`),
  `MemoryEvent` (9 typed event names from §3.5).
- **Domain dataclasses** — `Turn`, `ExecutionSummary`, `RecordReceipt`,
  `Insight`, `ReflectionContext`, `RetrievalQuery`, `RetrievalResult`,
  `Note`, `NoteMeta`, `NoteDraft`, `NotePatch`, `NoteRef`, `NoteGraph`,
  `MemorySnapshot`, `ReindexPlan`, `BackendInfo`, `EmbeddingDescriptor`,
  `CostEvent`, `CostModel`, `MemoryDescriptor` (self-describing —
  carries the `ConfigSchema` that `geny-executor-web` introspects),
  `MemoryHooks` (callbacks: `should_record_execution`, `should_reflect`,
  `should_auto_promote`).
- **7 handle Protocols**, all `@runtime_checkable`:
  - `STMHandle` — append/recent/search/truncate.
  - `LTMHandle` — append/read with `dated`/`topic` modes.
  - `NotesHandle` — write/list/update/delete/link/search with
    wikilink extraction and importance-weighted scoring.
  - `VectorHandle` — upsert/search/delete (optional).
  - `CuratedHandle` — session-scoped curation lane (optional).
  - `GlobalHandle` — cross-session global pool (optional).
  - `IndexHandle` — graph / tag counts / wikilink resolution.
- **`MemoryProvider` Protocol** — the single surface. Lifecycle
  (`initialize`/`close`), handle getters (`stm`/`ltm`/`notes`/
  `vector`/`curated`/`global_`/`index` — optional ones return
  `Optional[Handle]`), and cross-layer methods: `retrieve`,
  `record_turn`, `record_execution`, `reflect`, `snapshot`, `restore`,
  `promote`.

Design notes:

- `MemoryChunk` is imported under `TYPE_CHECKING` only to break a
  circular import through `stages.s02_context` (stage loads provider
  which loads MemoryChunk which loads stage…). Annotations are strings
  under `from __future__ import annotations`, so runtime is
  unaffected.
- Optional layers are gated with `Optional[Handle]` returns rather
  than separate provider subclasses. A provider declares *what it
  has* via `descriptor.layers` / `descriptor.capabilities`, and the
  contract suite cross-checks that layers declared in the descriptor
  resolve to non-None handles.

### `src/geny_executor/memory/providers/ephemeral.py` (new, 698 lines)

`EphemeralMemoryProvider` — zero-dependency, in-memory reference
conforming to `MemoryProvider`. Declares `Layer.STM`, `Layer.LTM`,
`Layer.NOTES`, `Layer.INDEX` in its descriptor; `vector()`,
`curated()`, `global_()` return `None` (honest capability gating).

Internal stores:

- `_STMStore` — recent-window list of `Turn`; keyword-scored `search`.
- `_LTMStore` — `main` list plus `dated[date]` and `topic[slug]`
  buckets, matching Geny's `LTMConfig.dated_file`/`topic_file` dual
  storage.
- `_NotesStore` — dict keyed by `NoteRef`; at write time extracts
  wikilinks via `r"\[\[([^\]]+)\]\]"` and stores them on
  `Note.links_out`; search score is
  `(1.0 + keyword_hits) * importance.boost + 0.3 * tag_overlap`.
- `_IndexCache` — materialises tag counts and the wikilink graph on
  demand, invalidated on any Notes write.

Snapshot/restore round-trips via a JSON payload with a SHA-256
checksum. `restore()` rejects a payload whose checksum does not match
the body — contract test `test_snapshot_round_trips` verifies.

### `src/geny_executor/memory/providers/__init__.py` (new)

Single-line public export: `EphemeralMemoryProvider`.

### `src/geny_executor/memory/__init__.py` (rewritten)

Phase 1+ contract symbols (32 names) are re-exported at the package
root. Legacy `GenyMemoryRetriever` / `GenyMemoryStrategy` /
`GenyPersistence` / `GenyPresets` are kept but marked in the module
docstring as **validation fixtures for Phase 3 (C7)**, not the
operating path. Total public surface: 39 symbols.

### `src/geny_executor/stages/s02_context/artifact/default/stage.py` (modified)

- New kwarg `provider: Optional[MemoryProvider] = None` on the stage
  constructor.
- When present, `provider.retrieve(RetrievalQuery(...))` runs in
  addition to the legacy retriever. The two chunk lists are merged
  and deduped by key before Stage 2 hands them to Stage 3.
- Emits `MemoryEvent.CONTEXT_BUILT.value` with the retrieval result
  payload so `geny-executor-web` (Phase 4) can display the 6-layer
  breakdown without reaching back into the provider.

### `src/geny_executor/stages/s15_memory/artifact/default/stage.py` (modified)

- New kwargs `provider: Optional[MemoryProvider] = None` and
  `hooks: Optional[MemoryHooks] = None`.
- New `_drive_provider(state)` method implementing the record-loop:
  1. Incrementally append any *new* `Turn`s to STM using
     `state.metadata["memory.last_recorded_idx"]` as the watermark.
  2. On terminal decision (`frozenset({"complete", "error",
     "escalate"})`), call `provider.record_execution(summary)`.
  3. Respect `hooks.should_reflect` / `hooks.should_record_execution`
     / `hooks.should_auto_promote` — defaults keep legacy behaviour.
- Legacy `ConversationPersistence` and `MemoryUpdateStrategy` slots
  remain; provider path is additive.

### `tests/contract/memory_provider_contract.py` (new, 348 lines)

`MemoryProviderContract` mixin with 28 behavioural assertions,
organised by handle:

| Section | Count |
|---|---|
| Descriptor + lifecycle (including handle-vs-descriptor agreement) | 3 |
| STM (append/recent, search, truncate) | 3 |
| LTM (main, dated, topic) | 3 |
| Notes (write, list/filter, patch, delete-return-bool, link, wikilink-edge, importance-floor search) | 7 |
| Index (tag counts, graph) | 2 |
| Cross-layer `retrieve` (layer breakdown, char-budget clamp) | 2 |
| Record (turn append, execution receipt) | 2 |
| Reflect | 1 |
| Snapshot round-trip (serialise + restore into fresh instance, checksum verified) | 1 |
| Promote (scope change, no-op when equal) | 2 |
| Optional-layer gating (declared ↔ handle presence) | 1 |
| Embedding compatibility (skips when no embedding configured) | 1 |

Subclasses override `provider` (and optionally `_fresh_from` for
snapshot round-trip if construction needs more than `()`).

### `tests/contract/test_memory_provider_ephemeral.py` (new, 23 lines)

`TestEphemeralProviderContract(MemoryProviderContract)` plugs in
`EphemeralMemoryProvider` via the `provider` fixture. No other code.
Phase 2 will add `test_memory_provider_file.py`, `_sql.py`,
`_composite.py` with identical bodies but different fixtures — the
assertion set stays constant.

### `pyproject.toml` / `src/geny_executor/__init__.py` (modified)

- Version bumped `0.13.5` → `0.14.0` per Phase 0's follow-up note:
  the bump is tied to shipping `geny_executor.memory.provider`.

## Tests

Full suite after Phase 1:

```
723 passed, 20 skipped in 4.19s
```

Breakdown of the delta vs. the pre-Phase-1 baseline (`695 passed`):

- `tests/contract/test_memory_provider_ephemeral.py` — **27 passed,
  1 skipped**. The one skip is
  `test_compatibility_check_when_embedding_present`, which correctly
  short-circuits because `EphemeralMemoryProvider.descriptor.embedding
  is None`. Phase 2 providers with embedding will flip it to an
  executed assertion.
- `tests/completeness/test_spec_loads.py` — green (loaded when PyYAML
  is available; skipped cleanly when it isn't, via
  `pytest.importorskip("yaml")` in the `spec` fixture — not at module
  top).
- `tests/completeness/test_c[1-7]_*.py` — **all still red** (skipped
  with phase attribution). Phase 2 flips C2/C3/C5/C6, Phase 3 flips
  C7, Phase 4 flips C4. C1's runtime foundation is now in place; the
  C1 test itself flips in Phase 2 once `FileMemoryProvider` lands so
  the end-to-end session→retrieve→record flow has a non-ephemeral
  harness.

No legacy test was modified. No legacy test regressed.

## Compatibility

- **Public API**: purely additive. 32 new names exported from
  `geny_executor.memory`; legacy 7 names retained at the same paths.
  Downstream importers continue to work.
- **Stage constructors**: both stages gained a `provider=`-style
  kwarg with a `None` default. Existing call sites that use positional
  strategies are unaffected; new code passes `provider=...` to opt in.
- **Behaviour**: when `provider is None`, both stages behave exactly
  as before Phase 1. Verified by the 695-test legacy suite running
  unchanged.
- **Geny**: not touched. There is no import from Geny anywhere in the
  Phase 1 change set, per the standing charter.
- **MEMORY_SPEC.yaml**: no modifications. The provider module is
  measured *against* the spec (importance boosts, event names,
  retrieval order, config field count) but does not update it. Any
  future spec drift requires a Phase 0-style amendment.

## Version bumps

- `geny-executor`: `0.13.5` → `0.14.0` (minor — new public contract).

## Follow-up

Phase 2 (`progress/memory/phase_2_native_providers.md`, to author):

- `FileMemoryProvider` — directory layout compatible with Geny's
  `LTMConfig` (STM JSONL, LTM markdown with dated/topic split, notes
  folders with frontmatter, FAISS `vectordb/`, `_index.json`). No
  Geny code imported — directory compatibility is a *format* target,
  not a runtime dependency.
- `EmbeddingClient` Protocol plus OpenAI / Voyage / Google / local
  backends. Drives the `VectorHandle` implementations and powers the
  C5 migration test.
- `SQLMemoryProvider` — SQLite + sqlite-vss primary; Postgres +
  pgvector secondary path behind the same Protocol.
- `CompositeMemoryProvider` — per-layer backend mix routing, so a
  user can keep STM in SQLite while running notes in Postgres.
- Concrete curated/global handles + `promote()` semantics end-to-end.
- `MemoryProviderFactory(descriptor_id, user_id, scope)` with the
  descriptor registry `geny-executor-web` will read (Phase 4).
- `GenyManagerAdapter` rebuilt strictly as a C7 test fixture under
  `tests/completeness/fixtures/adapter/` — not a runtime module.
- Flip C2, C3, C5, C6 from skip → real assertions. C1 activates here
  as well (needs a non-ephemeral provider for the full round-trip).
- Phase 1's contract mixin is reused verbatim across the new
  providers — any divergence surfaces as a test failure in
  `tests/contract/`.

Phase 2 gate (G2): C1–C3 and C5–C6 green against native providers,
no adapter dependency on the runtime path.
