# Changelog

All notable changes to `geny-executor` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.32.2] — 2026-04-24

Patch release — removes a ruff F401 (unused import) that blocked
0.32.1 CI from passing lint. No runtime behaviour changes.

### Fixed

- **`src/geny_executor/permission/matrix.py`** — removed an unused
  top-level `PermissionBehavior` import. The symbol is only reached
  at runtime via `rule.behavior` (already imported transitively), so
  dropping the top-level name does not affect behaviour.

### Notes

Both 0.32.0 and 0.32.1 tags exist on the repo with no published
wheels — this release is the first green-CI version. Phase 1 scope
is unchanged from the 0.32.0 design.

## [0.32.1] — 2026-04-24

Patch release — fixes Python 3.13 CI failure that blocked 0.32.0 from
publishing. No runtime behaviour changes; the source tree is otherwise
identical to the 0.32.0 target.

### Fixed

- **`tests/unit/test_phase6_history.py`** — two `ExecutionReplayer`
  tests (`test_replay_basic`, `test_replay_empty_raises`) called
  `asyncio.get_event_loop().run_until_complete(...)`. Python 3.13
  removed the implicit-event-loop fallback for this call and raises
  `RuntimeError: There is no current event loop` in the main thread
  when no loop is running, failing the CI runner. Replaced with
  `asyncio.run(...)` which works on 3.11+ identically.
- **`tests/unit/test_stage10_partition_executor.py`** — same issue in
  the new PartitionExecutor tests added by 0.32.0: 10 test methods
  built an explicit `new_event_loop()` + `run_until_complete` +
  `close` triplet, and the `_TimedTool` fixture called
  `asyncio.get_event_loop().time()` inside its async `execute()`.
  Switched to `asyncio.run(...)` at the entry point and
  `time.monotonic()` for wall-clock timing. More concise and
  Python-3.13-safe.

### Notes

The 0.32.0 git tag exists on the repo but no wheel was published to
PyPI — this patch release carries the same Phase 1 foundation
functionality (PRs #49–#52) under a fresh version so the first
published release is green on all supported Python versions.

## [0.32.0] — 2026-04-24

**Executor uplift Phase 1 — Foundation.** First release of a multi-phase
cycle toward 1.0 (see `Geny/executor_uplift/` in the Geny repo for the
full design, 12-part detailed plan, and migration roadmap). This
release lays down four primitive layers that subsequent releases build
on: extended Tool ABC metadata, permission rule matrix, subprocess hook
event taxonomy, and capability-aware Stage 10 orchestration. Every
change is additive — existing pipelines behave identically until they
opt in to the new surfaces.

### Added — Tool ABC metadata (PR #49)

- **`ToolCapabilities(frozen)`** — `concurrency_safe` · `read_only` ·
  `destructive` · `idempotent` · `network_egress` · `interrupt` ·
  `max_result_chars`. Fail-closed defaults. Runtime traits consumed by
  Stage 10 orchestrator, Permission matrix, and the upcoming Tool
  Review stage (Phase 9).
- **`PermissionDecision(frozen)`** — `behavior` (allow/deny/ask) +
  optional `updated_input` + `reason`.
- **`ToolContext`** new optional fields: `permission_mode`,
  `state_view`, `event_emit`, `parent_tool_use_id`, `extras`.
- **`ToolResult`** new optional fields: `display_text` (preferred by
  `to_api_format`), `persist_full`, `state_mutations`, `artifacts`,
  `new_messages`, `mcp_meta`.
- **`Tool`** ABC optional overrides with defaults: `output_schema`,
  `validate_input`, `capabilities(input)`,
  `check_permissions(input, ctx)`, `prepare_permission_matcher(input)`,
  `on_enter / on_exit / on_error` lifecycle hooks, `user_facing_name`,
  `activity_description`, `is_enabled`, plus `aliases`, `is_mcp`,
  `mcp_info` class attributes.
- **`build_tool()`** factory — construct a Tool instance without
  subclassing. Clears `__abstractmethods__` after property injection.

### Added — Permission rule matrix (PR #50)

New `geny_executor.permission` package.

- **`PermissionBehavior`** (`ALLOW / DENY / ASK`),
  **`PermissionMode`** (`DEFAULT / PLAN / AUTO / BYPASS`),
  **`PermissionSource`** + `SOURCE_PRIORITY` (CLI > LOCAL > PROJECT >
  USER > PRESET_DEFAULT).
- **`PermissionRule(frozen)`** — `tool_name` (`"*"` wildcard) + optional
  `pattern` + `behavior` + `source` + `reason`.
- **`evaluate_permission()`** — single entry point. Resolution order:
  (1) BYPASS short-circuit, (2) walk rules in source-priority order
  first-match-wins, (3) PLAN-mode destructive escalation to ASK,
  (4) optional fallback to the tool's own `check_permissions`,
  (5) default ALLOW. `_ToolLike` Protocol avoids circular imports.
- **`parse_permission_rules()` / `load_permission_rules()` /
  `load_hierarchical_rules()`** — YAML or JSON file loader with
  graceful PyYAML fallback to JSON.

### Added — Hook taxonomy + SharedKeys namespace (PR #51)

New `geny_executor.hooks` package and `geny_executor.core.shared_keys`
module.

- **`HookEvent`** enum — 16 kinds (SESSION_START/END,
  PIPELINE_START/END, STAGE_ENTER/EXIT, USER_PROMPT_SUBMIT,
  PRE/POST_TOOL_USE, POST_TOOL_FAILURE, PERMISSION_REQUEST/DENIED,
  LOOP_ITERATION_END, CWD_CHANGED, MCP_SERVER_STATE, NOTIFICATION).
- **`HookEventPayload`** — stable top-level schema; event-specific
  fields in `details` bag for forward compat.
- **`HookOutcome(frozen)`** — `continue_` / `suppress_output` /
  `decision` / `stop_reason` / `modified_input` /
  `hook_specific_output`. `passthrough` / `block` / `approve` /
  `from_response` helpers. `combine()` merges multiple outcomes with
  "most restrictive wins" semantics.
- **`SharedKeys`** — canonical string constants for well-known
  `state.shared` entries across three namespaces: `executor.*` (incl.
  pre-declared keys for Phase-9 stages), `memory.*`, `geny.*`.
- **`SharedKeys.plugin_key(namespace, key)`** — builder that returns
  `"plugin.{namespace}.{key}"` with identifier validation.

Hook **runner** (subprocess dispatch with timeout + stdout parsing)
lands in Phase 5 — this release ships only the taxonomy so dependent
checkpoints can import the types.

### Added — Stage 10 PartitionExecutor (PR #52)

First consumer of the Tool ABC metadata.

- **`PartitionExecutor`** registered as a third implementation in
  Stage 10's `executor` slot alongside `SequentialExecutor` and
  `ParallelExecutor`. Inspects each pending tool call's
  `Tool.capabilities(input).concurrency_safe` to run safe tools in a
  bounded parallel batch (`max_concurrency` default 10) and unsafe
  tools serially after. Result list preserves input order.
- **`PartitionExecutor.bind_registry`** — late-bind pattern mirroring
  `RegistryRouter`. `ToolStage.execute` now binds the registry into
  both the router and the executor when each exposes this method.
- **Fail-closed** — missing registry, unknown tool name, or
  `capabilities()` raising all degrade to unsafe (serial).

Opt-in: existing pipelines still default to `SequentialExecutor`.
Swap via `slot.swap("partition")` or
`PipelineMutator.swap_strategy(stage_order=10, slot_name="executor",
impl_name="partition")`.

### Compatibility

- **All additions are additive.** Existing Tool subclasses implementing
  only the 4 required members continue to work without modification.
- **No manifest / preset migration required.** Existing manifests load
  and run exactly as on 0.31.x.
- **Regression tests green.** 511 pre-existing unit tests continue to
  pass; this release adds 95 new tests (36 + 21 + 25 + 13) for a
  total of **606 passing + 189 skipped**.

### Cycle pointer

This release is Phase 1 of 10 in the executor uplift cycle.
Subsequent milestones: 0.33.0 (Orchestration) → 0.34.0 (Built-in tool
catalog) → 0.35.0 (Skills) → 0.36.0 (Hooks runner) → 0.37.0 (MCP
uplift) → 0.38.x (Stage enhancements) → 0.39.0 (MCP advanced) →
**1.0.0 (21-stage re-composition + v2→v3 manifest migration)**. See
`executor_uplift/11_migration_roadmap.md` and
`executor_uplift/12_detailed_plan.md` in the Geny repository for the
full plan.

## [0.30.0] — 2026-04-22

Minor release adding a single plugin-oriented primitive: the
`session_runtime` attach slot. Hosts can now thread session-scoped
non-stage objects (creature state, persona providers, emitter chains)
through the pipeline via a typed attribute carrier rather than
abusing `state.shared` as a stringly-typed bag — important for
third-party plugin coexistence where key-namespacing is otherwise
the host's problem.

Pure additive — every existing host and test passes unchanged. The
new slot defaults to `None`; behavior is only reachable when a host
opts in by passing `session_runtime=` to `attach_runtime`.

### Added

- **`Pipeline.attach_runtime(session_runtime=...)`** — seventh kwarg
  alongside `memory_retriever`, `memory_strategy`,
  `memory_persistence`, `system_builder`, `tool_context`,
  `llm_client`. Post-run re-attach refused (same discipline as the
  other kwargs).
- **`PipelineState.session_runtime: Optional[Any]`** — field on the
  run state, propagated from the attached value via `_init_state`.
  Explicit caller-supplied state wins over the attached default
  (matches `llm_client` semantics).

### Intentionally not added

- **No Protocol / ABC.** The executor does not inspect or constrain
  the attached object's shape — it is `Any`. Docstring includes a
  non-binding compatibility guideline (`getattr(..., "foo", None)`;
  missing attrs treated as opt-out) so competing plugins sharing a
  pipeline have a coordination hint without executor-enforced policy.
- **No automatic lifecycle hooks.** Host is responsible for any
  per-turn mutation or persistence; the slot is a plain reference.

### Host upgrade note

Existing hosts require no change. Hosts wanting to migrate
stringly-typed `state.shared["foo"]` bags onto a typed carrier can do
so incrementally — the two paths coexist.

## [0.29.0] — 2026-04-21

Minor release bundling cycle `20260421_4`: stage state interface,
unified LLM client package, and per-stage model routing for memory
stages. Five interlocking additive changes, one public interface
deletion. No silent behaviour change for pre-cycle pipelines — the
new paths are reachable only when a host opts in by setting a stage
override or attaching an `llm_client`.

### Added

- **`PipelineState.shared: Dict[str, Any]`** — pipeline-lifetime
  global scratchpad, cleared per run. Separate from
  `state.metadata` so stages that want a "global context" slot
  don't have to fight for dict keys.
- **`Stage.local_state(state) -> Dict[str, Any]`** — ergonomic
  per-stage scratchpad convention returning
  `state.metadata.setdefault(self.name, {})`. Two stages can now
  keep their own bookkeeping without collisions.
- **`Stage.resolve_model_config(state) -> ModelConfig`** — upgrades
  the prior `resolve_model` helper from "string model name" to the
  full `ModelConfig` bundle (model + sampling + thinking settings).
  Reads `self._model_override` first; otherwise builds from state
  defaults. `resolve_model` kept as a thin alias for back-compat.
- **`geny_executor.llm_client`** — new top-level package with
  `BaseClient` + `ClientCapabilities`, per-vendor
  `AnthropicClient` / `OpenAIClient` / `GoogleClient` / `VLLMClient`,
  and a provider-name `ClientRegistry`. Each client speaks the
  canonical `APIRequest` / `APIResponse` shape and silently drops
  unsupported fields, emitting `llm_client.feature_unsupported`
  events instead of raising.
- **`state.llm_client`** — optional `BaseClient` slot populated via
  `Pipeline.attach_runtime(llm_client=…)`. Any stage reaches for it
  when it needs an LLM; s06_api, s02 compaction, and s15
  reflection all consume it in this release.
- **`PipelineMutator.set_stage_model(order, cfg)`** — public entry
  point for installing per-stage `ModelConfig` overrides from host
  code. Raises `MutationError` (not `LookupError`) when the stage
  order is absent, matching the rest of the mutator's error
  surface.
- **`LLMSummaryCompactor`** (s02_context) — real summarizer that
  replaces the prior placeholder stub. Reads the resolved
  `ModelConfig` via a closure bound at stage init so per-run
  overrides take effect, and calls `state.llm_client.create_message`
  with `purpose="s02.summarize"`. Falls back to the static
  placeholder path when no override or client is present, preserving
  the pre-release no-cost guarantee.
- **`ReflectionResolver`** (s15_memory) — native reflection path for
  `GenyMemoryStrategy`. Dataclass carrying three closures
  (`resolve_cfg`, `has_override`, `client_getter`) that the strategy
  consults at reflect time instead of invoking a pre-baked
  `llm_reflect` callback. When both are provided, the legacy
  callback wins — hosts migrate by dropping the callback, not by
  toggling a flag. Calls through `state.llm_client` with
  `purpose="s15.reflect"`.

### Changed

- **s06_api (APIStage)** migrated onto the unified client. The per-
  vendor `APIProvider` artifact system
  (`stages/s06_api/artifact/{default,openai,google}/providers.py`)
  is **deleted**. `APIStage` now resolves a client via
  `state.llm_client` → stage-local `ClientRegistry.get(provider)`
  fallback → error, and calls `client.create_message(...)` directly.
  The stage's `provider: str` config field (new) replaces the
  `APIProvider` strategy slot.
- **`LLMSummaryCompactor` / `ReflectionResolver`** use closures
  bound to the owning stage handle so model/client resolution
  happens at call time, not pipeline-build time. Host code that
  installs overrides after `from_manifest_async` sees them honoured
  on the very next request.
- **`APIRequest` / `APIResponse` / `ContentBlock`** canonical types
  move from `stages.s06_api.types` into the top-level
  `geny_executor.llm_client.types` module. The old module re-exports
  from the new location; imports keep working without change.

### Removed

- `stages/s06_api/artifact/default/providers.py`
- `stages/s06_api/artifact/openai/providers.py`
- `stages/s06_api/artifact/google/providers.py`
- The `APIProvider` strategy slot on `APIStage`. Manifest-v2
  migration: artifacts named `"anthropic"` / `"openai"` / `"google"`
  on s06_api keep working via a migration shim that maps them to
  provider names consumed by the new `provider: str` config field.

### Upgrade notes

- Hosts that previously constructed `AnthropicProvider` /
  `OpenAIProvider` / `GoogleProvider` directly must switch to
  `ClientRegistry.get(provider)(api_key=…, base_url=…)` and inject
  via `attach_runtime(llm_client=…)`. The geny host does this in
  cycle-4 PR-6 (Geny `16690d7`).
- Pipelines that relied on the per-stage model override going
  ignored (pre-0.29.0 behaviour outside s06_api) will, if a host
  starts calling `set_stage_model(2, …)` or `set_stage_model(15, …)`,
  pick up a real LLM call on those stages. The override-absent
  branch still dials zero LLMs — the new work is gated by the
  host explicitly installing a `ModelConfig`.
- No breaking changes to public imports that did not live under
  `stages/s06_api/artifact/`. Hosts importing `Pipeline`,
  `PipelineMutator`, `ModelConfig`, `GenyMemoryStrategy` etc.
  continue unchanged.

### Cycle references

- Plan: `dev_docs/20260421_4/plan/01_pipeline_state_shared_and_local.md`
  → `plan/06_geny_memory_model_routing.md` (Geny side)
- Analysis: `dev_docs/20260421_4/analysis/02_memory_llm_inventory.md`
  (site-by-site justification)
- Progress: `progress/pr1_pipeline_state_shared_and_local.md`
  → `progress/pr5_memory_stages_use_model_override.md`

## [0.28.0] — 2026-04-21

Minor release. `GenyMemoryRetriever` gains a new L0 "recent turns"
layer that injects the tail of the short-term-memory transcript before
any semantic/keyword matching runs. The goal is to restore
conversational continuity on trigger-style turns — idle reflection,
sub-worker auto-reports, and inter-agent DMs — whose query text has no
lexical overlap with the prior dialogue and would otherwise miss the
last few turns entirely.

The new constructor argument `recent_turns: int = 6` controls the tail
size; pass `0` to disable. Layer budget is capped at 40% of
`max_inject_chars` so downstream layers (session summary, MEMORY.md,
vector, keyword, backlink, curated) still fit. Entries are injected
verbatim as `[<role>] <content>` lines, where `<role>` is read from
each STM entry's `metadata["role"]` (falling back to `"user"`), so new
roles such as `internal_trigger` and `assistant_dm` — added by Geny's
agent_session in the same cycle — flow through unmodified.

Duck-typed: if the injected memory manager exposes no
`short_term.get_recent(n)`, the layer quietly skips and the remaining
layers behave exactly as in 0.27.x. No breaking changes.

## [0.27.0] — 2026-04-21

Minor release. `Pipeline.from_manifest` / `from_manifest_async` now
auto-register the framework's shipped tool classes when the manifest
declares them via `tools.built_in`. The field was previously read-only
annotation; it is now a live dispatch list.

Accepted values for `manifest.tools.built_in`:

* `["*"]` — registers every class in
  `geny_executor.tools.built_in.BUILT_IN_TOOL_CLASSES` (Read, Write,
  Edit, Bash, Glob, Grep).
* `["Write", "Read"]` — registers only the named classes.
* `[]` or missing — no framework tools attached (preserves 0.26.x
  behaviour).

Built-ins register before external providers, so an external
`AdhocToolProvider` declaring an equally-named tool shadows the
built-in — host code can replace any framework default with a
hardened variant by shipping a same-named provider entry.

No breaking changes. Pipelines whose manifests carry `built_in: []`
(the value Geny's `default_manifest` wrote prior to 0.27.0) behave
identically to 0.26.x.

### Added

- **`BUILT_IN_TOOL_CLASSES`** — new public mapping in
  `geny_executor.tools.built_in` from registry name (`"Write"`) to
  tool class (`WriteTool`). Extensible: adding a new file-system or
  search tool to the framework now means dropping a module under
  `tools/built_in/` and one entry in the map.
- **`_register_built_in_tools`** — pipeline-internal helper that
  consumes `manifest.tools.built_in` and populates the registry via
  the map. Runs before `_register_external_tools` so external
  providers can still override.

### Changed

- `manifest.tools.built_in` graduates from annotation-only to active
  dispatch. Manifests authored against 0.26.x continue to work — an
  empty or missing field is a no-op.

## [0.26.0] — 2026-04-20

Additive release on top of 0.25.0. Extends
`Pipeline.attach_runtime(...)` with two new kwargs — `system_builder`
and `tool_context` — so manifest-built pipelines can be fully wired
for session-scoped behavior without reaching into stage internals.
Before this release the host had to mutate `SystemStage._slots["builder"].strategy`
and `ToolStage._context` by hand after `from_manifest_async`; now
one call does it all.

No breaking changes. Pipelines that don't pass the new kwargs behave
identically to 0.25.0. The existing `memory_retriever` /
`memory_strategy` / `memory_persistence` kwargs are untouched.

### Added

- **`attach_runtime(system_builder=...)`** — swaps Stage 3 (System)
  slot `builder` with the supplied `PromptBuilder`. Hosts that
  compose multi-block builders at session build time (e.g.
  `ComposablePromptBuilder([PersonaBlock(...), DateTimeBlock(),
  MemoryContextBlock()])`) can now attach them instead of baking
  them into a manifest (manifests can only serialize a static
  prompt string — block composition is runtime behavior).
- **`attach_runtime(tool_context=...)`** — overwrites Stage 10
  (Tool) `_context` with the supplied `ToolContext`. The attached
  context supplies host-level session fields (`working_dir`,
  `storage_path`, `env_vars`, `allowed_paths`, `metadata`). Note
  that `session_id` is still overwritten inside Stage 10's
  `execute` from the pipeline's per-run state — the attached
  context carries values that persist across runs.
- **Helper:** `Pipeline._set_tool_stage_context(...)` — internal
  helper for the `tool_context` kwarg. `ToolContext` is not a
  pluggable strategy slot (it is a data carrier), so it gets its
  own narrow setter rather than piggy-backing on
  `_set_stage_slot_strategy`.

### Why

Geny's manifest-first cutover
(`Geny/dev_docs/20260420_3/plan/02_default_env_per_role.md` → PR 17)
needs every session to flow through
`from_manifest_async → attach_runtime → run`. Two things blocked a
clean PR 17:

1. **Composable system prompt.** Geny builds a
   `ComposablePromptBuilder` per session that weaves `PersonaBlock`
   (role-specific system prompt) + `DateTimeBlock` (current-time
   injection) + `MemoryContextBlock` (active memory). A manifest's
   `system.prompt` string cannot encode block composition. Before
   this release, Geny reached into the stage's slot to swap the
   builder by hand.
2. **Session-scoped ToolContext.** Stage 10 builds per-call
   `ToolContext` from `self._context.working_dir` /
   `.storage_path`. Those paths live under a session's scratch
   directory, which is allocated at session-creation time and is
   not expressible in a static manifest.

Both are classic "runtime state that cannot live in a manifest" —
the same category `attach_runtime` was introduced for in v0.24.0.
Extending the existing helper keeps the host's wiring flow flat:
"build from manifest, attach runtime, run."

### Tests

`tests/unit/test_pipeline_attach_runtime.py` — 6 new tests
(14 total, all passing):

- `test_attach_runtime_replaces_system_builder` — passing
  `system_builder=<builder>` swaps Stage 3 slot `builder`; the
  `SystemStage._builder` property reflects the new strategy.
- `test_attach_runtime_replaces_tool_context` — passing
  `tool_context=<ctx>` overwrites `ToolStage._context` with the
  supplied instance; `working_dir` / `storage_path` / `metadata`
  survive.
- `test_attach_runtime_system_builder_missing_stage_noop` — a
  pipeline without a SystemStage silently ignores
  `system_builder`.
- `test_attach_runtime_tool_context_missing_stage_noop` — a
  pipeline without a ToolStage silently ignores `tool_context`.
- `test_attach_runtime_all_five_kwargs_together` — one call
  attaching all five (three memory + system_builder +
  tool_context) wires every target stage correctly.
- `test_attach_runtime_after_run_raises_for_v26_kwargs` — the
  post-run guard applies to the new kwargs too; each raises
  `RuntimeError` if the pipeline has already started.

Full suite: 1035 passed, 18 skipped.

## [0.25.0] — 2026-04-20

Additive release on top of 0.24.0. Makes the adaptive
`binary_classify` evaluation strategy resolvable from
`EnvironmentManifest` without import-time plumbing. Previously
manifest-restore silently fell back to `signal_based` because
`binary_classify` lived only in the `adaptive` artifact and was not
registered in the default `EvaluateStage`'s slot registry.

No breaking changes. Pipelines that don't reference
`binary_classify` from a manifest are byte-identical to 0.24.0.
The `adaptive` artifact remains strategy-only and its Python-level
import path (`from geny_executor.stages.s12_evaluate.artifact.adaptive.strategy import BinaryClassifyEvaluation`)
is unchanged — the 0.25.0 change is purely additive inside the
default stage's strategy slot.

### Added

- **`binary_classify`** entry in the default Stage 12
  (`EvaluateStage`) strategy slot registry — `StrategySlot.registry`
  now includes `{"signal_based", "criteria_based",
  "agent_evaluation", "binary_classify"}`. Manifests with
  `artifact="default"` and `strategies={"strategy":
  "binary_classify"}` now restore to a real
  `BinaryClassifyEvaluation` instance instead of silently falling
  back to `SignalBasedEvaluation`.
- **`BinaryClassifyEvaluation.configure(config: dict)`** — applies
  `easy_max_turns` and `not_easy_max_turns` from the manifest's
  `strategy_configs`. Unknown keys are ignored so newer manifests
  don't break older strategies.

### Why

Geny's manifest-first cutover
(`Geny/dev_docs/20260420_3/plan/02_default_env_per_role.md` →
`build_default_manifest.stages`) needs to serialize the
`worker_adaptive` preset faithfully. That preset pipes a
`BinaryClassifyEvaluation` into Stage 12 via the builder's
`.with_evaluate(strategy=...)` kwarg. A manifest-built pipeline
with `strategies.strategy = "binary_classify"` must produce the
same runtime behavior — otherwise the adaptive preset loses its
identity the moment it passes through an `EnvironmentManifest`.

### Tests

`tests/unit/test_binary_classify_manifest.py` (new, 6 tests):

- Manifest with `binary_classify` resolves to a real
  `BinaryClassifyEvaluation` (not `SignalBasedEvaluation`).
- `strategy_configs` flow through — `easy_max_turns` and
  `not_easy_max_turns` land on `strategy._config`.
- Absent `strategy_configs` preserves `BinaryClassifyConfig()`
  defaults.
- `configure(...)` ignores unknown keys.
- `configure({})` is a no-op on a pre-configured strategy.
- The default registry still exposes the three pre-existing
  strategies (regression guard against accidental replacement).

Full suite: 1029 passed, 18 skipped. Ruff + format clean.

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
