# Phase 2 — Native Providers (memory initiative)

> **Started**: 2026-04-19
> **Owner**: memory initiative
> **Predecessor**: `progress/memory/phase_1_interface.md`
> **Gate**: G2 — the native provider family (file, vector, SQL,
> composite) conforms to `MemoryProvider`, ships behind
> `MemoryProviderFactory`, and clears C1·C2·C3·C5·C6 without any
> adapter dependency. C7 (adapter parity) is Phase 3 work and the C4
> REST surface is Phase 4.

---

## Why Phase 2 is split into five sub-PRs

Phase 1 landed as a single large PR because Phase 0 and Phase 1 share
one axis (contract surface). Phase 2 is not that shape. It braids four
independent but ordered additions — filesystem layout, embedding
pluggability, SQL storage, and layer-wise composition — each of which
is its own review surface. Cramming them together would give a
reviewer an 80-file diff with no meaningful entry point and make any
single regression hard to bisect.

Sub-PRs (ordered):

| Sub-PR | Branch | Target tag | Scope |
|---|---|---|---|
| **2a** | `feat/memory-phase-2a-file-provider` | `v0.15.0` | `FileMemoryProvider` (STM JSONL, LTM markdown, Notes YAML frontmatter, Index cache). |
| 2b | `feat/memory-phase-2b-embedding-clients` | `v0.16.0` | `EmbeddingClient` Protocol + OpenAI / Voyage / Google / local backends. |
| 2c | `feat/memory-phase-2c-sql-provider` | `v0.17.0` | `SQLMemoryProvider` (SQLite + sqlite-vss, Postgres + pgvector). |
| 2d | `feat/memory-phase-2d-composite-factory` | `v0.18.0` | `CompositeMemoryProvider` + `MemoryProviderFactory`. |
| 2e | `feat/memory-phase-2e-adapter-c-tests` | `v0.19.0` | Quarantined `GenyManagerAdapter` + activate C1·C2·C3·C5·C6. |

Each sub-PR targets `main` with its own `feat(vX.Y.Z):` tag so rollback
is surgical. Phase 2 closes when 2a–2e are merged and `pytest -m
completeness` turns green for C1·C2·C3·C5·C6 (C4 = Phase 4, C7 =
Phase 3).

---

## Sub-PR 2a — FileMemoryProvider (this PR)

**Target tag**: `v0.15.0`
**Branch**: `feat/memory-phase-2a-file-provider`

### Summary

First on-disk `MemoryProvider`. STM/LTM/Notes/Index layers are backed
by files laid out the way Geny's legacy `SessionMemoryManager` lays
them out, so a legacy reader can consume the output *without reading
any executor code* — and vice versa, the executor can pick up a
session a Geny installation already wrote. Vector/Curated/Global layers
return `None`; they wire in sub-PRs 2b / 2d.

Zero Geny imports. The format compatibility is enforced by
`tests/contract/test_memory_provider_file_layout.py`, which is a pure
format-lock suite independent of the behavioural contract suite.

### Changes

Subpackage `src/geny_executor/memory/providers/file/` — 11 modules,
each with one responsibility:

- **`layout.py`** — `DirectoryLayout` dataclass. Resolves every path
  (transcripts/session.jsonl, memory/MEMORY.md, memory/topics/*,
  memory/{daily,entities,projects,insights}/*, vectordb/index.faiss,
  memory/_index.json). `NOTE_CATEGORIES` = six subfolders (plus `root`
  for notes directly under memory/). `RESERVED_FILENAMES` guards
  MEMORY.md / _index.json / summary.md from being scanned as notes.
- **`frontmatter.py`** — hand-rolled YAML frontmatter parser (split /
  parse / dump). No PyYAML dependency. Handles inline lists, quoted
  strings, booleans/null/numbers. Deterministic dump order so
  round-trips don't produce spurious diffs.
- **`timezone.py`** — `resolve_timezone(name)` with precedence: arg →
  `GENY_TIMEZONE` env → local → UTC. Supports IANA names via
  `zoneinfo.ZoneInfo` plus numeric `±HH:MM` offsets.
- **`stm_store.py`** — `_JSONLSTMStore`. One JSONL record per turn
  (`{type, role, content, ts, metadata}`). 2000-line cap with
  `enforce_line_cap()` and atomic `truncate(keep_last=N)` via
  `.tmp.replace()`. Unknown `type` records (e.g. `tool_call`) are
  skipped on read but preserved on disk so the web mirror can render
  them directly.
- **`ltm_store.py`** — `_MarkdownLTMStore`. Main file (MEMORY.md) gets
  HTML-comment timestamps on every append. `write_dated(body, day)` →
  `memory/YYYY-MM-DD.md`. `write_topic(title, body)` → slugged file
  under `memory/topics/`. Search scoring blends keyword density with
  a 30-day recency half-life.
- **`notes_store.py`** — `_FilesystemNotesStore` (NotesHandle). Writes
  frontmatter + body to the correct category directory. Wikilink
  extraction (`[[target]]` / `[[target|alias]]`), bidirectional
  backlinks recomputed on every write. Search scoring:
  `(1 + keyword_hits) * note.importance.boost + 0.3 * tag_overlap`,
  with `importance_floor` respecting `Importance.boost`.
- **`index_store.py`** — `_FileIndexStore`. Derived cache at
  `memory/_index.json` with schema
  `{files, tag_map, link_graph, last_rebuilt, total_files,
  total_chars}`. `rebuild()` rescans notes; `snapshot()` materialises
  the cache to disk for the tarball.
- **`config.py`** — `file_provider_config_schema()` exposing all 21
  R-F config fields from `MEMORY_SPEC.yaml` (master enable, embedding
  provider/model/key, chunk size/overlap, retrieval top-k/threshold/
  max-inject, curated toggles, auto-curation schedule, Obsidian index
  toggle). This is what `geny-executor-web` will introspect to render
  the memory-settings form without any hardcoded field list.
- **`snapshot.py`** — `build_tarball(root) -> (bytes, sha256_hex)` and
  `restore_tarball(root, payload, checksum)`. Checksum mismatch raises
  `ValueError("snapshot checksum mismatch: …")`. Restore stages into
  `.restore-tmp`, verifies tarball safety (no absolute paths, no
  traversal), and swaps into place so a mid-restore crash cannot leave
  a partial tree. Extraction uses `filter="data"` to silence the
  Python 3.14 tarfile deprecation.
- **`provider.py`** — `FileMemoryProvider(MemoryProvider)`. Wires the
  stores, declares `Layer.{STM,LTM,NOTES,INDEX}` +
  `Capability.{READ,WRITE,SEARCH,LINK,SNAPSHOT}` in its descriptor.
  `record_execution()` writes dated LTM + insights note + rebuilds the
  index. `retrieve()` composes STM + LTM + Notes with a char budget
  (always keeps at least one chunk). `snapshot()` materialises the
  index first so the tarball is self-describing. `promote()` is a
  no-op rewrite of the ref's scope — real cross-scope motion is
  Composite (sub-PR 2d) territory.
- **`__init__.py`** — `from .provider import FileMemoryProvider`.

Also updated:

- `src/geny_executor/memory/providers/__init__.py` — re-exports
  `FileMemoryProvider`.
- `src/geny_executor/memory/__init__.py` — re-exports
  `FileMemoryProvider`; `__all__` now lists both ephemeral and file
  providers.
- `pyproject.toml` / `src/geny_executor/__init__.py` — bumped to
  `0.15.0`.

### Tests

Two new files under `tests/contract/`:

- **`test_memory_provider_file.py`** — single subclass
  `TestFileProviderContract(MemoryProviderContract)` with a `tmp_path`
  fixture. Every one of the 28 behavioural assertions in the contract
  mixin is reused verbatim. The only override is `_fresh_from`, which
  constructs a sibling `-restored` root for the snapshot round-trip
  test (since the file provider needs a *different* directory to
  restore into).
- **`test_memory_provider_file_layout.py`** — format-lock suite.
  Verifies: default path map (stm_jsonl, main_ltm, dated_ltm, topic
  slug placement, vectordb paths, index_json); `ensure()` creates the
  full tree; `is_reserved` matches MEMORY.md / _index.json but not
  normal notes; JSONL schema (one line per turn, `type=message`,
  ISO-8601 `ts`); unknown-type records skipped during read but
  preserved on disk; MEMORY.md append starts with an HTML timestamp
  comment; dated filename is `YYYY-MM-DD.md`; topic slug begins with
  the normalised title; frontmatter block contains title/tags/
  category/importance and parses back identically including wikilink
  targets; index JSON is written to disk with the five required keys;
  snapshot round-trip preserves notes; tampered checksum raises
  `ValueError, match="checksum"`; `retrieve()` composes LTM + Notes
  and respects `max_chars`.

Full suite (post-2a): **764 passed, 21 skipped** (skips are Phase 3
completeness tests and optional-layer tests still gated on the
embedding/SQL sub-PRs).

### Compatibility

Geny format, byte-for-byte. Specifically:

- `transcripts/session.jsonl` — same JSONL schema (including `type`
  field to distinguish message vs event records).
- `memory/MEMORY.md` — markdown with HTML-comment timestamps, same as
  `SessionMemoryManager._append_with_timestamp`.
- `memory/YYYY-MM-DD.md` — same dated file convention.
- `memory/topics/{slug}.md` — same slug rules (lowercase, hyphen
  replacement, Hangul passthrough).
- Notes frontmatter — keys `title`, `tags`, `category`, `importance`,
  `created`, `modified`, `links_to`; inline list form for small arrays
  so the file stays single-line-per-field friendly.
- `memory/_index.json` — derived cache, same five top-level keys.

Geny is not imported. The compatibility is format-level only, and it's
held in place by the format-lock test suite; changing a path or a key
in the provider without updating the lock is a test failure.

### Version bumps

`0.14.0 → 0.15.0`.

### Follow-up

- Sub-PR 2b — `EmbeddingClient` Protocol + FAISS vector store wired
  into `vectordb/`.
- Sub-PR 2c — `SQLMemoryProvider` mirroring this surface via SQLite /
  Postgres tables.
- Sub-PR 2d — `CompositeMemoryProvider` + `MemoryProviderFactory`
  (per-layer backend routing, scope promotion from SESSION → USER →
  TENANT → GLOBAL).
- Sub-PR 2e — quarantined `GenyManagerAdapter` fixture + activation of
  C1·C2·C3·C5·C6 completeness tests against the file provider.

---

## Sub-PR 2b — EmbeddingClient + Vector layer (closed 2026-04-19)

**Target tag**: `v0.16.0`
**Branch**: `feat/memory-phase-2b-embedding-clients`

### Summary

Adds the embedding pluggability axis and wires the Vector layer into
`FileMemoryProvider`. Four embedding backends conforming to a single
`EmbeddingClient` Protocol; the file provider now owns a
VectorHandle-conformant `_FileVectorStore` when an embedding client is
supplied at construction. Local (deterministic) backend has zero
dependencies and is used across the test suite. Remote backends are
optional installs.

### Changes

Subpackage `src/geny_executor/memory/embedding/` — 6 modules:

- **`client.py`** — `EmbeddingClient` Protocol (`embed`, `close`,
  `descriptor`) + `EmbeddingError`. Every backend satisfies this
  Protocol, and `FileMemoryProvider` / `_FileVectorStore` only ever
  speaks through it.
- **`local.py`** — `LocalHashEmbeddingClient`. SHA-256 hashing trick
  producing a deterministic L2-normalised vector of configurable
  dimension (default 384, bounded to [32, 4096]). Zero deps. Not
  semantic — but same-token texts land closer than no-token-overlap
  texts, which is enough for testability.
- **`openai.py`** — `OpenAIEmbeddingClient`. Wraps the `openai` SDK's
  `AsyncOpenAI.embeddings.create`. Reference dims baked in for
  `text-embedding-3-small/large/ada-002`. SDK optional; import only
  happens on first `embed` call so the module loads without the extra.
- **`voyage.py`** — `VoyageEmbeddingClient`. POSTs directly to
  `https://api.voyageai.com/v1/embeddings` via httpx (transitive dep
  through anthropic). Supports a `transport=` injection for tests
  so the happy path can be exercised without network.
- **`google.py`** — `GoogleEmbeddingClient`. Uses `google-genai`'s
  async surface `client.aio.models.embed_content`.
- **`registry.py`** — `create_embedding_client(provider, …)` factory
  dispatching on provider name. Unknown names raise `ValueError`;
  missing optional SDKs surface the original `ImportError` with the
  correct install instructions.

Vector store — `src/geny_executor/memory/providers/file/vector_store.py`:

- **`_FileVectorStore`** — VectorHandle-conformant store. Vectors
  packed as little-endian float32 into `vectordb/index.bin`; metadata
  mirror in `vectordb/metadata.json`. Pure-Python cosine similarity
  (no numpy dep). Single lock serialises all writes. Reindexing same
  filename replaces the row. Dimension validation on every insert
  (raises `ValueError("vector dimension mismatch: …")`) — this is the
  invariant C5 relies on. `reindex()` rebuilds every row from source
  notes and returns a `ReindexPlan(layer=VECTOR, …)` the UI can
  surface.

Provider wiring — `src/geny_executor/memory/providers/file/provider.py`:

- Constructor now accepts `embedding_client: Optional[EmbeddingClient]`.
  If supplied, `vector()` returns the `_FileVectorStore`; otherwise
  `None` (back-compat with sub-PR 2a).
- `record_execution()` now indexes the written note into the vector
  store when one is configured, so the vector row is born with the
  note — no separate pass needed.
- `retrieve()` adds a Vector layer arm (guarded on
  `Layer.VECTOR in query.layers` and vector presence).
- `descriptor` now surfaces `Layer.VECTOR` + `Capability.REINDEX` when
  wired, plus a `BackendInfo` entry carrying provider/model/dimension
  metadata for the web console to render.
- `snapshot().layers` now includes `Layer.VECTOR` when present. The
  on-disk tarball already captured the `vectordb/` tree; the only
  change is the declared layer list.
- `restore()` rebuilds `_vector` against the restored layout so the
  handle keeps working after a swap.

### Tests

- **`tests/contract/test_embedding_clients.py`** — 19 tests covering
  descriptor fields, embed order/determinism/normalisation for local;
  SDK-stubbed OpenAI happy path + error path + helpful ImportError on
  missing SDK; transport-stubbed Voyage happy/error; registry
  dispatch for every provider plus unknown-provider guard.
- **`tests/contract/test_memory_provider_file_vector.py`** — 12 tests
  across five areas: wiring (vector is non-None when configured, None
  otherwise, descriptor surfaces VECTOR layer), index+search
  (deterministic hits, row replacement, removal, dimension-mismatch
  rejection), auto-index on `record_execution`, `retrieve()` includes
  vector source when declared, `reindex()` rebuilds from source and
  returns the right plan shape, snapshot round-trip preserves the
  vector payload.

Full suite: **795 passed, 21 skipped** (31 new tests since 2a).

### Compatibility

- `vectordb/index.faiss` path is reserved in `DirectoryLayout` but
  not written — the pure-Python store uses `index.bin` alongside so
  a future FAISS-based provider can live beside it without collision.
- `vectordb/metadata.json` is self-describing (dimension, model,
  metric) so a replacement store can validate compatibility before
  reading.
- Adding an embedding client is strictly additive — existing sessions
  without one continue working unchanged. C5 dimension-change
  handling lights up now; the reindex flow is surfaced via the
  `ReindexPlan` dataclass from Phase 1 without any schema changes.

### Version bumps

`0.15.0` → `0.16.0`.

### Follow-up

- Sub-PR 2c — `SQLMemoryProvider` mirrors the same surface via
  SQLite + sqlite-vss / Postgres + pgvector.
- Sub-PR 2d — `CompositeMemoryProvider` fans the Vector layer out
  per-scope (SESSION / USER / TENANT / GLOBAL).
- Sub-PR 2e — C5 completeness test lights up (dimension swap →
  reindex plan → apply → `memory.reindexed` event).
