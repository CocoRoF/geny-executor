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

---

## Sub-PR 2c — SQLMemoryProvider (closed 2026-04-19)

**Target tag**: `v0.17.0`
**Branch**: `feat/memory-phase-2c-sql-provider`

### Summary

Second on-disk `MemoryProvider`, this time with SQL semantics. Same
seven layer handles, same descriptor surface, same behavioural
contract — different storage shape. STM/LTM/Notes/Vector/Index live as
SQLite tables in a single `*.db` file; the file provider's
markdown/JSONL surface remains unchanged. Zero new core dependencies:
the provider is built on stdlib `sqlite3` wrapped in an asyncio lock
so the rest of the runtime can `await` it.

The dialect choice (SQLite today, Postgres + pgvector later) flows
through `_SQLiteConnection`. Stores never reach for the cursor
directly, which keeps a future Postgres swap to a per-connection-class
change instead of a per-store rewrite.

### Changes

Subpackage `src/geny_executor/memory/providers/sql/` — 11 modules:

- **`schema.py`** — `SCHEMA_VERSION = "1"` + idempotent
  `CREATE TABLE IF NOT EXISTS` for the canonical seven tables
  (`stm_turns`, `ltm_documents`, `notes`, `note_tags`, `note_links`,
  `vector_rows`, `provider_meta`). FK cascades from `note_tags` and
  the `source` side of `note_links` to `notes(filename)`; `target`
  is intentionally not an FK so a wikilink can point at a not-yet-
  written note (matches the file provider's eager-link behaviour).
  `SQLITE_TABLES` is the lock used by the schema test.
- **`connection.py`** — `_SQLiteConnection` async wrapper around
  stdlib `sqlite3`. One `asyncio.Lock` serialises every
  cursor.execute / executemany / fetchone / fetchall;
  `check_same_thread=False` is explicit because the lock is the
  single source of mutual exclusion. `transaction()` is an async
  context manager using `BEGIN IMMEDIATE` so two concurrent writers
  never silently overlap. `truncate_all()` is what `restore_snapshot`
  rides on.
- **`stm_store.py`** — `_SQLSTMStore`. One row per turn in
  `stm_turns` with `(role, content, content_kind, type, ts,
  metadata_json)`. `content_kind` discriminates `"string"` vs
  `"json"` so structured tool-call content survives a round trip
  byte-for-byte. `truncate(keep_last=N)` uses `OFFSET` to find the
  cutoff `id` in a single statement.
- **`ltm_store.py`** — `_SQLLTMStore`. Three logical kinds (main,
  dated, topic) on one table with `UNIQUE(kind, ref_name)` so an
  append to MEMORY.md is an UPSERT, a dated write picks the day's
  row, and a topic write keys off slug. Body composition (HTML
  timestamp comments on main, evergreen + dated render in
  `read_main`) matches the file provider character-for-character so
  the cross-provider contract test passes.
- **`notes_store.py`** — `_SQLNotesStore`. Notes row has the same
  fields the file provider's frontmatter carries; tags are
  normalised into `note_tags`; wikilinks are parsed at write time
  into `note_links` with `origin='wikilink'`, and explicit
  `link()` / `unlink()` writes `origin='explicit'`. `read()`
  reconstructs the full Note (tags + links_to + backlinks) in two
  follow-up queries. `delete()` also drops `note_links WHERE
  target = ?` because target isn't an FK. `search()` reuses the
  file provider's scoring formula
  `(1 + keyword_hits) * importance.boost + 0.3 * tag_overlap`.
- **`vector_store.py`** — `_SQLVectorStore`. Vectors live in
  `vector_rows` as packed little-endian float32 BLOBs alongside
  provenance metadata. `index()` / `index_batch()` use SQLite
  UPSERT (`ON CONFLICT(filename) DO UPDATE`) so a re-index is one
  statement. Cosine similarity is pure Python — same calculation
  the file provider uses — so no native extension is needed for
  this PR. `_validate_dim` raises `ValueError("vector dimension
  mismatch: …")` on every insert; this is the invariant C5
  depends on. `reindex()` clears + rebuilds rows from the notes
  store via `notes_text_lookup`, returning a `ReindexPlan`
  carrying the descriptor + rebuilt-row count.
- **`index_store.py`** — `_SQLIndexStore`. The DB is canonical, so
  `rebuild()` is a no-op; `snapshot()` materialises a derived view
  (`{files, tag_map, link_graph, last_rebuilt, total_files,
  total_chars}`) shaped identically to the file provider so
  external readers don't branch on backend.
- **`snapshot.py`** — `build_snapshot(conn) -> (bytes, sha256_hex)`
  dumps every row of every owned table to a JSON document with the
  shape `{format, version, generated_at, tables: {name: rows}}`.
  BLOBs are base64-encoded on the wire (`{"__b64__": "…"}`). The
  checksum is SHA-256 of the JSON bytes. `restore_snapshot` validates
  the checksum, calls `truncate_all()` inside a transaction, and
  re-inserts rows with the BLOB decoder mirror — so a half-restore
  cannot leave a hybrid state.
- **`config.py`** — `sql_provider_config_schema()` mirrors the file
  provider's 21 R-F fields plus a `dsn` field, so the same
  `geny-executor-web` form renders both providers without a code
  fork.
- **`provider.py`** — `SQLMemoryProvider(MemoryProvider)`. Wires
  the stores; declares `Layer.{STM,LTM,NOTES,INDEX}` +
  `Capability.{READ,WRITE,SEARCH,LINK,SNAPSHOT}`; lights up
  `Layer.VECTOR` + `Capability.REINDEX` when an `EmbeddingClient`
  is supplied. `record_execution()` mirrors the file provider
  (writes LTM dated + insights note + indexes the vector row when
  wired). `retrieve()` composes STM + LTM + Notes + Vector with the
  same char budget. `promote()` updates `notes.scope` directly so
  subsequent reads agree.
- **`__init__.py`** — `from .provider import SQLMemoryProvider`.

Also updated:

- `src/geny_executor/memory/providers/__init__.py` — re-exports
  `SQLMemoryProvider`.
- `src/geny_executor/memory/__init__.py` — re-exports
  `SQLMemoryProvider`; `__all__` now lists ephemeral, file, and SQL
  providers.
- `pyproject.toml` / `src/geny_executor/__init__.py` — bumped to
  `0.17.0`.

### Tests

Three new files under `tests/contract/`:

- **`test_memory_provider_sql.py`** — single subclass
  `TestSQLProviderContract(MemoryProviderContract)` with a
  `tmp_path` fixture. The 28-assertion behavioural mixin runs
  verbatim against the SQL backend; the only override is
  `_fresh_from`, which builds a sibling `*-restored.db` so the
  snapshot round-trip restores into an independent file.
- **`test_memory_provider_sql_schema.py`** — format-lock suite for
  the SQL backend. Opens a sibling sync `sqlite3` connection and
  asserts: every table in `SQLITE_TABLES` is created on
  `initialize()`; the `notes` column set matches the contract
  (`filename, title, body, importance, category, scope, backend,
  frontmatter_json, created_at, updated_at`); `vector_rows` carries
  `vector_blob`, `dimension`, and `filename`; `stm_turns` writes
  string and json content with the right `content_kind`
  discriminator and ISO-8601 `ts`; `note_links` distinguishes
  `origin='wikilink'` from `origin='explicit'`; `note_tags`
  deduplicates and normalises; the snapshot payload is JSON with a
  `tables` key and includes every owned table; tampered checksum
  raises `ValueError, match="checksum"`; `retrieve()` composes
  Notes + LTM; `promote()` persists the new scope to the row.
- **`test_memory_provider_sql_vector.py`** — 9 tests covering
  vector wiring (handle present/absent + descriptor surfaces VECTOR
  layer + REINDEX capability), index+search (deterministic hits +
  row replacement on re-index of same filename), dimension-mismatch
  rejection, auto-index on `record_execution()`, retrieve includes
  vector source when declared, `reindex()` returns the right plan
  shape, and snapshot round-trip preserves the vector rows
  including their BLOB payload.

Full suite (post-2c): **843 passed, 22 skipped** (48 new tests since
2b — 28 from the contract mixin reuse, 12 from schema lock, 8 from
vector parity).

### Compatibility

- **Same descriptor surface as the file provider.** `Layer` set,
  `Capability` set, `EmbeddingDescriptor` shape, scope semantics,
  and `BackendInfo` schema are identical — only the `backend`
  string differs (`"sqlite"` vs `"filesystem"`) and the per-layer
  metadata describes tables instead of paths.
- **Cross-provider contract parity.** The 28-test
  `MemoryProviderContract` mixin runs unchanged against both
  providers, which is the proof that user code (and the eventual
  Composite) doesn't have to branch on backend.
- **Snapshot format is provider-tagged.** `MemorySnapshot.provider`
  is `"sql"` here vs `"file"` in 2a; restore explicitly rejects
  cross-provider payloads — there is no silent format coercion.
  Cross-provider migration is a Composite-level concern (sub-PR 2d).
- **Embedding pluggability.** Same `EmbeddingClient` Protocol from
  2b. The SQL provider can be wired with any of the four backends
  (local / OpenAI / Voyage / Google) without store changes.
- **Zero net new core dependencies.** Stdlib `sqlite3` only. The
  pgvector arm is a follow-up sub-PR and would slot in by replacing
  `_SQLiteConnection` + `_SQLVectorStore` while leaving the public
  surface untouched.

### Version bumps

`0.16.0` → `0.17.0`.

### Follow-up

- Sub-PR 2d — `CompositeMemoryProvider` + `MemoryProviderFactory`
  (per-layer backend routing, scope promotion SESSION → USER →
  TENANT → GLOBAL, cross-provider migration helpers).
- Sub-PR 2e — quarantined `GenyManagerAdapter` fixture + activation
  of C1·C2·C3·C5·C6 against both file and SQL providers.
- Postgres + pgvector backend — separate follow-up PR. The dialect
  abstraction keeps stores portable; the swap point is
  `_SQLiteConnection` plus the `vector_store` UPSERT statement.
