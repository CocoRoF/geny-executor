# Changelog

All notable changes to `geny-executor` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.24.0] — 2026-04-20

Additive release on top of 0.23.0. Introduces `Pipeline.attach_runtime(...)`,
a single explicit injection point for the session-scoped runtime objects
(memory retriever, memory strategy, conversation persistence) that cannot
be encoded in an `EnvironmentManifest`. Manifests express declarative
shape — stages, artifacts, strategy choices, configs — but not the
per-session objects a host needs to wire in after construction. Before
this release, hosts reached into stage internals to set those; now they
call one helper.

No breaking changes. Pipelines that never call `attach_runtime` behave
identically to 0.23.0 — stages still carry whatever retriever / strategy /
persistence was supplied at construction (or their defaults:
`NullRetriever`, `AppendOnlyStrategy`, `NullPersistence`). The 0.22.x-style
`GenyPresets.worker_adaptive(...)` / `GenyPresets.vtuber(...)` builders
remain available and unchanged; `attach_runtime` is an additional path for
manifest-first hosts.

### Added

- **`Pipeline.attach_runtime(*, memory_retriever=None, memory_strategy=None,
  memory_persistence=None)`** in `geny_executor.core.pipeline`. Walks the
  registered stages and replaces the relevant slot strategies:
  - `memory_retriever` → Stage 2 (Context), slot `retriever`.
  - `memory_strategy` → Stage 15 (Memory), slot `strategy`.
  - `memory_persistence` → Stage 15 (Memory), slot `persistence`.
  Kwargs are keyword-only. Omitted kwargs leave the corresponding slot
  untouched. Missing stages are silently skipped — a pipeline without a
  Memory stage simply has nowhere to attach memory runtime.
- **`Pipeline._has_started`** flag, flipped by `_init_state` on the first
  `run()` / `run_stream()` invocation. `attach_runtime` raises
  `RuntimeError` after this flip — prior stage state has already captured
  references to the pre-attach slot values, so swapping them would yield a
  mixed-runtime pipeline whose behavior is hard to reason about. Build a
  fresh pipeline and attach before running.

### Why

Plan/02 of the 20260420_3 Geny cycle moves session creation from
hardcoded `GenyPresets.*` branches to `Pipeline.from_manifest_async(...)`.
Manifests are declarative, so they cannot carry runtime objects
(`SessionMemoryManager`, `llm_reflect` callback, `CuratedKnowledgeManager`).
`attach_runtime` provides the missing post-manifest wiring step without
forcing hosts to reach into `_slots["retriever"].strategy` directly.

See `Geny/dev_docs/20260420_3/plan/02_default_env_per_role.md` for the
full cutover context.

### Tests

`tests/unit/test_pipeline_attach_runtime.py` (new, 8 tests):

- Replaces Context.retriever slot identity.
- Replaces Memory.strategy + Memory.persistence slots.
- Accepts all three kwargs together.
- Idempotent before first run — last call wins per kwarg.
- Omitting a kwarg preserves the prior value (partial attach).
- Missing target stage is a silent no-op.
- Raises `RuntimeError` after `_init_state` flips `_has_started`.
- Calling with no kwargs is a valid no-op.

Full suite: 1023 passed, 18 skipped. Ruff + format clean.

## [0.23.0] — 2026-04-20

Additive release on top of 0.22.1. Extends the Stage 10 tool event
vocabulary with per-call events so downstream log consumers can
render the input, outcome, and latency of individual tool calls.
Prior to 0.23.0 only summary events (`tool.execute_start` /
`tool.execute_complete`) were emitted, forcing hosts like Geny to
either read pipeline-internal state or re-parse the Anthropic
response — both brittle. The 0.23.0 contract is event-level and
stable.

No breaking changes. Existing summary events are preserved
byte-for-byte; consumers that listen only to `tool.execute_*` see
no behavior change. The new `on_event` kwarg on
`ToolExecutor.execute_all` is keyword-only and optional — default
`None` matches 0.22.1 semantics exactly. Third-party executors
implementing `ToolExecutor` continue to work without modification
(they simply don't emit the new events, which was their existing
reality).

### Added

- **`tool.call_start`** event, fired by the default Stage 10
  executors (`SequentialExecutor`, `ParallelExecutor`) immediately
  before each individual dispatch. Payload:
  `{tool_use_id, name, input}` — the full Anthropic-supplied call
  id, tool name, and input dict. Paired with `tool.call_complete`
  via `tool_use_id`.
- **`tool.call_complete`** event, fired immediately after each
  dispatch. Payload: `{tool_use_id, name, is_error, duration_ms}`.
  Does not carry the output payload — full results remain on the
  message bus (state) to keep the event stream bounded.
- **`on_event` keyword-only kwarg** on
  `ToolExecutor.execute_all(...)` (interface + both default
  implementations). Shape: `Callable[[str, dict], None]`. The
  default `ToolStage` wires it to `state.add_event`, preserving
  the existing event-listener path (`state._event_listener`).
- **`ToolEventCallback` type alias** in
  `geny_executor.stages.s10_tool.interface`, exported alongside
  `ToolExecutor` / `ToolRouter`.

### Why

Host-side log UIs (e.g., Geny's `tool_detail_formatter`) need the
per-call input dict to render a call-by-call detail pane. The
0.22.1 summary events omit this, and the pipeline-internal
`pending_tool_calls` field is not a stable event contract. This
release upgrades the contract so hosts can stop reaching into
pipeline state. See
`Geny/dev_docs/20260420_3/plan/01_immediate_fixes.md` (PR II) for
the design rationale and the full event-vocabulary audit.

### Tests

`tests/unit/test_tool_call_events.py` (new, 6 tests):

- Sequential executor emits `call_start` / `call_complete` per call,
  in order, carrying the correct payload.
- `is_error=True` propagates into `call_complete`.
- `on_event=None` (omitted) is a no-op — matches 0.22.1.
- Parallel executor emits paired `call_start` / `call_complete`
  events keyed by `tool_use_id`; inter-pair ordering is not
  asserted (parallelism).
- `ToolStage` nests per-call events *inside*
  `tool.execute_start` / `tool.execute_complete`, preserving the
  outer bracket contract.

Full suite: 1015 passed, 18 skipped.

## [0.22.1] — 2026-04-20

CI hygiene patch on top of 0.22.0. No runtime behavior change — same
public API, same import surface, identical test outcomes (1003 passed,
5 skipped).

### Fixed

- `ruff check` now passes on `main`: dropped two unused imports that
  slipped through the 0.22.0 PRs (`ToolError` in
  `tools/mcp/adapter.py`, `MCPServerConfig` in
  `tests/unit/test_adhoc_providers.py`). (#27)
- `ruff format --check` now passes on `main`: eleven files that the
  0.22.0 PRs touched diverged from the project's default ruff
  formatter; applied `ruff format` so CI stays green. (#28)

## [0.22.0] — 2026-04-20

Tool / MCP integration hardening release. Bundles four breaking
changes discovered during the Geny ↔ executor cutover (see
`Geny/dev_docs/20260420_2/plan/` for the full context). The release
is intentionally packaged as one breaking bump so downstream Geny
can pin `geny-executor>=0.22.0,<0.23.0` and cut over in a single
PR rather than chasing four micro-upgrades.

### Added

- **`ToolError` / `ToolFailure` / `ToolErrorCode`** in
  `geny_executor.tools.errors`. Structured error model replacing ad-hoc
  string returns. Every host-side error now surfaces a stable payload
  `{error: {code, message, details}}` which the Anthropic tool_result
  bridge renders with a leading `ERROR <code>: <message>` header line.
  Codes: `UNKNOWN_TOOL`, `INVALID_INPUT`, `TOOL_CRASHED`,
  `ACCESS_DENIED`, `TRANSPORT`. (#22)
- **`validate_input(schema, payload)`** — jsonschema helper used by the
  default router and available for tool implementations. Converts
  jsonschema failures into `ToolFailure(code=INVALID_INPUT)`. (#22)
- **`MCPConnectionError(server_name, phase, cause)`** in
  `geny_executor.tools.mcp.errors` — a single structured exception for
  every phase of MCP server start-up (`connect`, `initialize`,
  `list_tools`, `sdk_missing`). (#24)
- **`Pipeline.from_manifest_async`** — async sibling of
  `from_manifest` that assembles stages, opens MCP servers with
  fail-fast semantics, registers adapters, and attaches
  `pipeline.mcp_manager` / `pipeline.tool_registry`. (#24)
- **`MCPManager.add_server(config, *, registry=None)`** /
  **`MCPManager.remove_server(name, *, registry=None)`** — runtime
  hot-swap of MCP servers that also keeps the registry in sync. (#24)
- **`AdhocToolProvider` Protocol** in
  `geny_executor.tools.providers` — runtime-checkable Protocol with
  `list_names()` / `get(name)` that lets hosts supply tools not
  expressible as `AdhocToolDefinition`. (#25)
- **`ToolsSnapshot.external: List[str]`** — manifest-level whitelist
  naming which provider-backed tools are active in a given
  environment. Legacy manifests (without the field) continue to load
  unchanged. (#25)
- **`Pipeline.from_manifest(..., adhoc_providers=(), tool_registry=None)`**
  and the matching async signature — walks `manifest.tools.external`,
  registers the first claiming provider per name into the supplied
  (or fresh) registry, attaches it to the pipeline. (#25)

### Changed (breaking)

- **Every MCP tool is now always namespaced `mcp__{server}__{tool}`**
  (previously the bare tool name). The prefix is mandatory; there is
  no opt-out. Host-side tool registries, logs, and downstream
  display code that matched on bare MCP tool names need to be
  updated. (#23)
- **MCP lifecycle is fail-fast.** Previously an MCP server that
  failed its `initialize` or `list_tools` step could persist in a
  "connected-but-no-op" state. v0.22.0 raises `MCPConnectionError`
  from `MCPManager.connect_all` at session-start time and rolls
  back every transiently-connected server before the exception
  propagates. Manifests that reference a broken MCP server will no
  longer load — the failure is now eager, not lazy. (#24)
- **`MCPServerConnection.call_tool`** return type expanded from
  `str` to `str | list[dict]`. Single-text-block responses still
  return `str`; multi-block and non-text responses return
  `list[dict]` preserving block `type`. `MCPToolAdapter.execute`
  passes both through to `ToolResult.content` unchanged. Direct
  callers of `call_tool` may now need an `isinstance` branch. (#24)
- **`ToolRegistry.register`** now emits a warning when a different
  tool instance is re-registered under an existing name. The
  previous silent overwrite hid double-registration bugs. (#23)
- **Default `RegistryRouter`** emits structured `ToolError` payloads
  for unknown tool, invalid input, tool crash, and access-denied
  flows. Callers that parsed the previous plain-string error
  content must switch to the structured shape. (#22)

### Dependencies

- Adds `jsonschema>=4.0` as a runtime dependency. (#22)

### Migration notes

- **MCP tool names**: any prompt, mapping, or log-scrape that
  referred to `read_file` now needs to reference
  `mcp__filesystem__read_file` (or the appropriate server prefix).
- **MCP manifests**: any environment that previously got away with
  a half-broken MCP server definition will now fail loudly at
  session start. Clean stale `mcp_servers` entries before deploy.
- **Tool error parsing**: host code that did
  `if result.content.startswith("Error:")` should switch to
  checking `result.is_error` and reading the structured
  `content["error"]["code"]`.
- **Unified tool surface (opt-in)**: hosts using the new
  `AdhocToolProvider` hook can point every environment — env_id
  or non-env_id — at a single `Pipeline.from_manifest_async(...)`
  call and drop any bespoke `ToolRegistry` plumbing. See the
  companion `Geny/dev_docs/20260420_2/plan/01_unified_tool_surface.md`.

### PRs in this release

- #22 — structured `ToolError` + jsonschema input validation.
- #23 — mandatory `mcp__{server}__{tool}` namespace.
- #24 — MCP fail-fast lifecycle + `Pipeline.from_manifest_async`.
- #25 — `AdhocToolProvider` Protocol + `tools.external` field.
