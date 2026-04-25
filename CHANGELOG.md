# Changelog

All notable changes to `geny-executor` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 1.1.0

### Added — Task lifecycle output streaming (PR-A.1.1)

- ``TaskRecord.output_path`` — optional pointer to externally
  persisted output bytes (file path / blob URI). Defaults to
  ``None`` for backward compat.
- ``TaskFilter`` — query object combining status / kind /
  ``created_after`` / ``limit``. Used by ``TaskRegistry.list_filtered``.
- ``TaskRegistry.list_filtered(filter)`` — default impl on top of
  ``list_all``. Persistent backends (Postgres / Redis) override
  to push the filter into the query layer.
- ``TaskRegistry.append_output(task_id, chunk)`` /
  ``read_output(task_id, offset, limit)`` /
  ``stream_output(task_id)`` — output streaming surface. Defaults
  to no-op / empty bytes / immediate-return so existing backends
  remain compatible without changes.
- ``InMemoryRegistry`` — implements the streaming surface with
  per-task ``bytearray`` buffers + ``asyncio.Event`` so consumers
  wake on each ``append_output`` rather than polling. ``remove`` /
  terminal status transitions wake waiters so they drain and exit
  cleanly.

20 new unit tests in ``tests/unit/test_s13_task_registry_output.py``.

### Added — FileBackedRegistry (PR-A.1.2)

- ``FileBackedRegistry(root: Path)`` — durable single-process task
  registry. Mutations append to ``root/registry.jsonl``; tombstones
  for ``remove`` so reload doesn't resurrect deleted tasks. Output
  bytes per task in ``root/outputs/<task_id>.bin`` (path-traversal
  safe). Corrupt / partial JSONL lines logged + skipped on load.
- Exported from ``geny_executor.stages.s13_task_registry``.

17 new tests in ``tests/unit/test_s13_file_backed_registry.py``.

### Added — BackgroundTaskRunner + executors (PR-A.1.3)

- ``geny_executor.runtime`` — new framework-runtime layer that
  lives outside the synchronous pipeline path. Service code (FastAPI
  lifespan / CLI bootstrap / SDK bootstrap) instantiates it at
  startup and tears it down at shutdown.
- ``BackgroundTaskExecutor`` ABC — one executor per task ``kind``.
  Yields output bytes; raises on failure.
- ``LocalBashExecutor`` — runs ``payload['command']`` via shell;
  streams stdout (+stderr merged) up to a configurable
  ``max_output_bytes`` cap.
- ``LocalAgentExecutor`` — dispatches to a
  :class:`SubagentTypeOrchestrator` via ``run_subagent`` /
  ``spawn``; serializes the result (str / bytes / json) for
  consumers reading via ``stream_output``.
- ``BackgroundTaskRunner`` — owns ``asyncio.Task`` futures;
  ``submit / stop / shutdown / start``. ``start`` sweeps stale
  RUNNING records (crash recovery). Concurrency limited by
  ``max_concurrent`` semaphore. Idempotent re-submit, idempotent
  shutdown.

20 new unit tests in ``tests/unit/test_runtime_task_runner.py``.

### Added — AgentTool built-in (PR-A.1.4)

- ``AgentTool`` (registered as ``"Agent"``) — LLM-callable tool that
  spawns a sub-agent via a host-supplied
  :class:`SubagentTypeOrchestrator`. The orchestrator is read from
  ``ToolContext.extras["agent_orchestrator"]`` (host wires at startup).
- Recursion guarded by ``extras["agent_depth"]`` /
  ``extras["agent_max_depth"]`` (default 3) so AgentTool calling
  AgentTool can't run away.
- All error paths return structured ``{"error": {"code": ..., "message": ...}}``
  payloads so the LLM can introspect and recover instead of seeing
  free-form exception strings.
- Added to ``BUILT_IN_TOOL_CLASSES`` and a new ``"agent"`` feature
  group.

16 new unit tests in ``tests/unit/test_agent_tool.py``.

### Added — 6 task lifecycle tools (PR-A.1.5)

LLM-callable wrappers around BackgroundTaskRunner + TaskRegistry:

- ``TaskCreate`` — submit a new background task; returns task_id +
  current status.
- ``TaskGet`` — fetch one record by id.
- ``TaskList`` — list with optional ``status`` / ``kind`` / ``limit``
  filter; ordered by created_at desc.
- ``TaskUpdate`` — mutate ``payload`` only. Status transitions are
  intentionally NOT user-mutable so a misbehaving LLM can't mark a
  still-running task as DONE.
- ``TaskOutput`` — read accumulated bytes (offset + limit). Capped
  at 1 MiB per call so the response budget can't be blown.
- ``TaskStop`` — cooperative cancel via runner.

Wiring contract: hosts inject ``task_registry`` + ``task_runner``
into ``ToolContext.extras`` at startup. Read-only tools (Get / List /
Update / Output) work without ``task_runner`` so a host can read
state from a backend populated by a different process.

22 new unit tests in ``tests/unit/test_task_tools.py``.

## [1.0.0] — 2026-04-25

**First stable release.** Closes the multi-month executor uplift
roadmap. PyPI classifier moves from ``Development Status :: 4 -
Beta`` to ``Development Status :: 5 - Production/Stable``.

This release bundles the deferred Sub-phase 9c follow-ups (the
read half of HITL + crash recovery) plus the formal stability
declaration. There are **no breaking changes** vs 0.46.x — every
0.46-pinned host can pin ``geny-executor[web]>=1.0.0,<2.0.0`` and
upgrade with no code changes.

### Added — S9c.1 Pipeline.resume API for HITL (PR #120)

- ``Pipeline._pending_hitl: Dict[str, Future[HITLDecision]]`` —
  internal token-keyed registry the resume requester populates
  and the resume API resolves.
- ``Pipeline.list_pending_hitl()`` — token list of unresolved
  requests.
- ``Pipeline.resume(token, decision)`` — resolves the pending
  Future. Accepts :class:`HITLDecision` or strings
  (``"approve"`` / ``"reject"`` / ``"cancel"``). Raises
  ``KeyError`` on unknown token, ``RuntimeError`` when already
  resolved, ``ValueError`` on unknown decision string.
- ``Pipeline.cancel_pending_hitl(token) -> bool`` — convenience
  for "session terminated, drop in-flight approvals" cleanup.
- ``PipelineResumeRequester(pipeline)`` — :class:`Requester`
  that registers a Future on ``pipeline._pending_hitl`` under
  the request's token and awaits it. Cleans up the registration
  in a ``finally`` block so cancellation never leaks entries.
  Added to ``HITLStage``'s slot registry as
  ``"pipeline_resume"``.

### Added — S9c.2 Checkpoint restoration helpers (PR #121)

- ``CheckpointNotFound`` LookupError — distinguishable from
  backend errors which propagate.
- ``state_from_payload(payload) -> PipelineState`` — inverse of
  ``PersistStage._build_payload``. Tolerates missing keys,
  ignores unknown extras, rebuilds :class:`TokenUsage`.
- ``state_from_record(record)`` — convenience wrapper.
- ``async restore_state_from_checkpoint(persister, checkpoint_id)``
  — reads via the persister and rebuilds. Raises
  ``CheckpointNotFound`` when the persister returns ``None``.
- Runtime fields (``llm_client`` / ``session_runtime``) are
  intentionally **not** restored — hosts rebind them on the run
  that uses the restored state.

### Stability commitment

The library now ships under semver 1.0:

* **Breaking changes** require a major version bump (2.0).
* **Additive features** ship in minor (1.x.0); they preserve the
  default behaviour of every 1.0-era pipeline.
* **Bug fixes** ship in patch (1.0.x).
* The 21-stage layout, the strategy-slot interfaces, the
  :class:`Pipeline` / :class:`PipelineState` / :class:`PipelineConfig`
  class surfaces, the :class:`MCPManager` API, the manifest v3
  schema, and the slot-registry conventions are all considered
  stable. Internals prefixed with ``_`` remain freely
  changeable.

### Roadmap completion summary

The executor uplift shipped over six minor releases (0.42 → 0.46)
and one stability marker (1.0.0):

* **Phase 7** (12 sprints) — every stage gained at least one new
  strategy slot or class-level extension surface.
* **Phase 8** (4 sprints) — credential store, OAuth 2.0 flow,
  ``mcp://`` URI scheme, prompts→Skills bridge.
* **Phase 9 Sub-phase 9a** (5 sprints) — 16-stage → 21-stage
  layout, manifest v2→v3 migration, preset regen.
* **Phase 9 Sub-phase 9b** (5 sprints) — real strategy slots for
  the five new stages (tool_review / task_registry / hitl /
  summarize / persist).
* **Phase 9 Sub-phase 9c** (2 sprints) — Pipeline.resume +
  checkpoint restoration helpers.

Phase 10 (Observability — frontend dashboard) remains optional
and does not block 1.0.

---

## [0.46.0] — 2026-04-25

**Closes Phase 9 Sub-phase 9b — every former scaffold has real
behaviour now.** Five sprints (S9b.1 → S9b.5) bundled into one
minor release. All five previously-scaffold stages
(`tool_review`, `task_registry`, `hitl`, `summarize`, `persist`)
now have full strategy-slot implementations. Defaults preserve
pre-0.46.0 behaviour (no-op / always-approve / no-summary /
no-persist), so existing pipelines continue to run identically.

### Added — S9b.1 Stage 11 Tool Review (PR #114)

- ``ToolReviewFlag`` frozen dataclass + ``Reviewer`` Strategy ABC.
- Five default reviewers: ``SchemaReviewer`` (per-tool required
  fields), ``SensitivePatternReviewer`` (api key / AWS / private
  key / bearer regex), ``DestructiveResultReviewer`` (mutating
  tool whitelist), ``NetworkAuditReviewer`` (host allowlist),
  ``SizeReviewer`` (warn / error byte bands).
- ``ToolReviewStage`` exposes a ``reviewers`` ``SlotChain``
  (default order: schema → sensitive → destructive → network →
  size). Per-reviewer failure isolation; flag list lives at
  ``state.shared['tool_review_flags']``; reset every execute().
- Helpers: ``collect_flags``, ``has_error_flag``, ``reset_flags``,
  ``append_flags`` + ``SEVERITY_*`` constants. Events:
  ``tool_review.flag``, ``tool_review.reviewer_error``,
  ``tool_review.completed``.

### Added — S9b.2 Stage 13 Task Registry (PR #115)

- ``TaskStatus`` enum (pending/running/done/failed/cancelled) +
  ``TaskRecord`` mutable dataclass with ``mark()`` and
  ``is_terminal``.
- ``TaskRegistry`` Strategy ABC + ``InMemoryRegistry``
  (process-lifetime). ``TaskPolicy`` Strategy ABC + three
  defaults: ``FireAndForgetPolicy`` (default), ``EagerWaitPolicy
  (executor=...)``, ``TimedWaitPolicy(executor=...,
  timeout_seconds=30)``.
- ``TaskRegistryStage`` exposes ``registry`` + ``policy`` slots.
  Drains ``state.shared[PENDING_TASKS_KEY]``, coerces dicts,
  registers, runs the policy (try/except so a bad policy can't
  wedge the loop). Publishes ``state.shared[TASKS_BY_STATUS_KEY]``
  group-by-status snapshot. Events: ``task.registered``,
  ``task.done`` / ``task.failed`` / ``task.timeout``,
  ``task_registry.invalid_payload`` /
  ``task_registry.policy_error`` / ``task_registry.synced``.

### Added — S9b.3 Stage 15 HITL (PR #116)

- ``HITLRequest`` frozen dataclass (auto-generated 16-byte
  URL-safe token) + ``HITLDecision`` enum (approve/reject/cancel)
  + ``HITLEntry`` audit record + coercion helpers.
- ``Requester`` Strategy ABC + two defaults: ``NullRequester``
  (always approves — safe default) and ``CallbackRequester``
  (delegates to host async callable; ``configure()`` supports
  late wiring).
- ``TimeoutPolicy`` Strategy ABC + three defaults:
  ``IndefiniteTimeout``, ``AutoApproveTimeout``,
  ``AutoRejectTimeout``. Validation up front and on configure.
- ``HITLStage`` exposes ``requester`` + ``timeout`` slots.
  Bypass when ``state.shared['hitl_request']`` empty. Bounded
  wait via ``asyncio.wait_for`` when ``timeout_seconds`` set;
  on timeout the policy decides the verdict. Requester
  exceptions emit ``hitl.requester_error`` and return cancel.
  Reject → ``loop_decision="complete"`` + ``HITL_REJECTED``;
  cancel → ``escalate`` + ``HITL_CANCELLED``. Audit log at
  ``state.shared['hitl_history']``; latest verdict at
  ``state.shared['hitl_last_decision']``.
- ``Pipeline.resume`` API for cross-request resumption is
  intentionally deferred — the current Requester abstraction
  already covers in-process WebSocket-style HITL.

### Added — S9b.4 Stage 19 Summarize (PR #117)

- ``SummaryRecord`` dataclass (turn_id / abstract / key_facts /
  entities / tags / importance / created_at). Re-uses
  ``memory.provider.Importance``.
- ``Summarizer`` Strategy ABC + two defaults: ``NoSummarizer``
  (default — returns None / no-op) and ``RuleBasedSummarizer``
  (sentence-split + capitalised-token extraction; configurable
  caps + extra_tags; handles bare-string and block-shaped
  assistant messages).
- ``ImportanceScorer`` Strategy ABC + two defaults:
  ``FixedImportance(grade=MEDIUM)`` (default) and
  ``HeuristicImportance`` (high keywords → HIGH, escalation to
  CRITICAL on tool-review error; low keywords → LOW; many facts
  / entities → HIGH).
- ``SummarizeStage`` exposes ``summarizer`` + ``importance``
  slots. Bypass for default NoSummarizer. Per-component try/
  except. Publishes ``state.shared['turn_summary']`` +
  ``state.shared['summary_history']``. Optional forward to
  ``state.session_runtime.memory_provider.record_summary``
  when present (failures isolated). Events: ``summary.skipped``,
  ``summary.written``, ``summary.summarizer_error``,
  ``summary.importance_error``,
  ``summary.provider_recorded`` / ``summary.provider_error``.

### Added — S9b.5 Stage 20 Persist (PR #118)

- ``CheckpointRecord`` dataclass (auto-generated ``ckpt_*`` id /
  session_id / iteration / created_at / payload).
- ``Persister`` Strategy ABC + two defaults: ``NoPersister``
  (default no-op) and ``FilePersister(base_dir)`` (atomic
  JSON-file writes via tempfile + ``os.replace`` + ``fsync``
  running in ``asyncio.to_thread``; implements ``read`` +
  ``list_checkpoints``).
- ``FrequencyPolicy`` Strategy ABC + three defaults:
  ``EveryTurnFrequency`` (default), ``EveryNTurnsFrequency
  (n=5)``, ``OnSignificantFrequency`` (significant when an
  event in ``significant_events`` fired this turn, or
  tool-review error, or high-importance summary, or
  ``state.completion_signal`` set).
- ``PersistStage`` exposes ``persister`` + ``frequency`` slots.
  ``should_bypass`` for default NoPersister. Frequency check
  first; payload covers non-runtime state only (live
  ``llm_client`` / ``session_runtime`` excluded). Persister
  exceptions emit ``checkpoint.persister_error``. Successful
  writes update ``state.shared['last_checkpoint']`` +
  ``state.shared['checkpoint_history']``. Events:
  ``checkpoint.skipped``, ``checkpoint.written``,
  ``checkpoint.persister_error``.
- ``Pipeline.resume_from_checkpoint`` is intentionally deferred
  — this release ships the *write* half so hosts can start
  collecting checkpoints; the read/restore API lands in a
  follow-up.

### Compatibility

Additive only. Default slot strategies for every promoted stage
preserve the exact pre-0.46.0 behaviour:

* tool_review: empty pending tool calls → ``should_bypass`` True.
* task_registry: empty queue → publishes empty status view, no
  side effects.
* hitl: empty request key → ``should_bypass`` True.
* summarize: ``NoSummarizer`` → ``should_bypass`` True.
* persist: ``NoPersister`` → ``should_bypass`` True.

### Phase 9 summary

Two sub-phases, ten sprints. Sub-phase 9a (S9a.1–S9a.5) widened
the canonical pipeline from 16 to 21 slots and migrated manifests
+ presets. Sub-phase 9b (S9b.1–S9b.5) replaced each scaffold's
pass-through body with a real strategy-slot implementation.
``Pipeline.resume`` / ``resume_from_checkpoint`` for cross-
request HITL and crash-recovery remain on the follow-up backlog
— Sub-phase 9b ships the in-process write half of both.

---

## [0.45.0] — 2026-04-25

**Closes Phase 9 Sub-phase 9a (21-stage scaffolding) of the
executor uplift roadmap.** Largest single structural change in
the uplift: the canonical pipeline grew from 16 to 21 slots.
Sub-phase 9a is **no-op behaviour-wise** — five new slots are
pass-through / bypass scaffolds that Sub-phase 9b will fill with
real implementations. Existing pipelines continue to run
identically; new infrastructure makes 9b a one-PR-per-stage
exercise.

### Stage layout (new)

| Order | Module | Body | Source |
|---|---|---|---|
|  1 | s01_input | Input | unchanged |
|  2 | s02_context | Context | unchanged |
|  3 | s03_system | System | unchanged |
|  4 | s04_guard | Guard | unchanged |
|  5 | s05_cache | Cache | unchanged |
|  6 | s06_api | API | unchanged |
|  7 | s07_token | Token | unchanged |
|  8 | s08_think | Think | unchanged |
|  9 | s09_parse | Parse | unchanged |
| 10 | s10_tool | Tool | unchanged |
| **11** | **s11_tool_review** | **Tool Review (pass-through)** | **NEW (S9a.2)** |
| 12 | s12_agent | Agent | renamed from s11_agent |
| **13** | **s13_task_registry** | **Task Registry (pass-through)** | **NEW (S9a.2)** |
| 14 | s14_evaluate | Evaluate | renamed from s12_evaluate |
| **15** | **s15_hitl** | **HITL (always-bypass)** | **NEW (S9a.2)** |
| 16 | s16_loop | Loop | renamed from s13_loop |
| 17 | s17_emit | Emit | renamed from s14_emit |
| 18 | s18_memory | Memory | renamed from s15_memory |
| **19** | **s19_summarize** | **Summarize (no-op)** | **NEW (S9a.2)** |
| **20** | **s20_persist** | **Persist (NoPersist)** | **NEW (S9a.2)** |
| 21 | s21_yield | Yield | renamed from s16_yield |

### Added — S9a.1 Stage rename (PR #108)

- ``git mv`` for the six existing stages whose orders moved.
  All 110 import references in ``src/`` and ``tests/`` updated
  via grep + sed. ``order`` properties left at the legacy values
  in this PR (they move in S9a.3).

### Added — S9a.2 Scaffolding stages (PR #109)

- Five new directories with pass-through / bypass implementations:
  ``s11_tool_review``, ``s13_task_registry``, ``s15_hitl``,
  ``s19_summarize``, ``s20_persist``. Each ships ``__init__`` /
  ``artifact/__init__`` / ``artifact/default/__init__`` /
  ``artifact/default/stage.py`` and exposes a ``Stage`` alias for
  ``create_stage``.

### Added — S9a.3 Pipeline wiring (PR #110)

- ``STAGE_MODULES`` re-keyed from 16 → 21 entries; ``STAGE_ALIASES``
  gains five new short names.
- Per-stage ``order`` properties bumped to match the new slot
  (Agent 11 → 12, Evaluate 12 → 14, Loop 13 → 16, Emit 14 → 17,
  Memory 15 → 18, Yield 16 → 21).
- ``Pipeline.LOOP_END`` 13 → 16, ``FINALIZE_START`` 14 → 17,
  ``FINALIZE_END`` 16 → 21; ``_DEFAULT_STAGE_NAMES`` extended.
- ``Pipeline.describe()`` and ``PipelineMutator.snapshot()`` walk
  ``STAGE_MODULES`` instead of hard-coded ``range(1, 17)`` so future
  renumberings don't need a code edit.

### Added — S9a.4 Manifest v2 → v3 auto-migration (PR #111)

- ``MANIFEST_VERSION`` bumped ``"2.0"`` → ``"3.0"``.
- ``EnvironmentManifest.from_dict`` chains v1 → v2 → v3 in one
  call. The v2 → v3 step pads the stages list out to the new
  21-slot layout — any of the five new orders missing from the
  payload are inserted as inactive default pass-through entries.
  Existing entries are preserved byte-for-byte; the migration is
  idempotent (existing entries at the new orders are not
  overwritten).

### Added — S9a.5 Preset regen (PR #112)

- Five new ``PipelineBuilder`` opt-in methods:
  ``with_tool_review`` / ``with_task_registry`` / ``with_hitl`` /
  ``with_summarize`` / ``with_persist``.
- ``PipelinePresets.agent`` / ``.geny_vtuber`` and
  ``GenyPresets.worker_easy`` / ``.worker_adaptive`` /
  ``.worker_full`` / ``.vtuber`` updated to call the new methods
  so introspection and manifest export show all 21 slots
  populated. ``minimal`` / ``chat`` / ``evaluator`` intentionally
  unchanged.

### Compatibility

- Existing pipelines continue to run identically — the five new
  stages are pass-through / bypass; ``_try_run_stage`` silently
  skips unregistered slots.
- Manifests load forward (v1 / v2 → v3) automatically.
- Hosts that pin ``geny-executor[web]>=0.45.0,<0.46.0`` and
  rebuild from manifest will see ``len(introspect_all()) == 21``
  and ``Pipeline.describe()`` returning 21 entries.

### Phase 9 Sub-phase 9a summary

Five sprints, one release. The pipeline architecture is now ready
for Sub-phase 9b — each new stage gets a dedicated PR replacing
its scaffold body with real behaviour (Tool Review chain, Task
Registry, HITL gate with ``Pipeline.resume`` API, Summarize LTM
indexer, Persist session checkpoint).

---

## [0.44.0] — 2026-04-25

**Closes Phase 8 (MCP Advanced) of the executor uplift roadmap.**
Bundles four sprints (S8.1 → S8.4) into one minor release. All
new surfaces are independently opt-in; existing MCP integrations
see no behaviour change.

### Added — S8.1 Credential store (PR #103)

- ``geny_executor.tools.mcp.credentials`` module:
    * ``CredentialStore`` Protocol — get / set / delete / keys.
    * ``MemoryCredentialStore`` — process-lifetime dict.
    * ``FileCredentialStore`` — JSON-file persistence with
      ``mode=0600`` atomic writes (tempfile + ``os.replace`` +
      ``fsync``). Tolerates missing/empty files; rejects corrupt
      JSON / non-object payloads with descriptive ``ValueError``;
      creates parent directories on first set.
    * ``mcp_credential_key(server_name)`` — canonical
      ``mcp:<name>`` prefix helper.

### Added — S8.2 OAuth 2.0 authorization-code flow (PR #104)

- ``geny_executor.tools.mcp.oauth`` module:
    * ``OAuthAuthConfig`` frozen dataclass + required-field
      validation.
    * ``OAuthToken`` (access/refresh/expires_at/scope/raw) with
      JSON round-trip + ``is_expired(leeway_seconds=30)`` and a
      ``from_token_response`` normaliser (``expires_in`` → epoch
      ``expires_at``).
    * ``OAuthError`` single error type.
    * ``build_authorize_url`` — composes URLs with state + scope
      + extra params (tolerates pre-existing query strings).
    * ``find_free_port`` helper.
    * ``OAuthFlow`` end-to-end orchestrator: 32-byte URL-safe
      state for CSRF; stdlib ``HTTPServer`` bound to ``127.0.0.1``
      by default; ``consent_handler`` callback for the URL;
      injectable ``http_post`` (default ``httpx``); persists JSON
      blob under ``mcp:<server_name>`` via the credential store.
      Threads cleanly shut down in the ``finally`` block.
      ``load_cached_token`` returns ``None`` on corrupt cache;
      ``revoke_cached_token`` removes it.

### Added — S8.3 mcp:// URI scheme + manager resource API (PR #105)

- ``geny_executor.tools.mcp.uri`` module:
    * ``mcp://<server>[/<resource_id>]`` grammar; server name
      regex ``[A-Za-z0-9_.-]+``; opaque ``resource_id`` passed
      back to the MCP SDK verbatim.
    * ``parse_mcp_uri`` / ``build_mcp_uri`` / ``is_mcp_uri`` /
      ``MCPURIError`` / ``MCP_URI_SCHEME``.
- ``MCPManager`` API:
    * ``read_mcp_resource(uri)`` — parses, routes, returns
      ``None`` for unknown / disconnected; invalid URI raises
      ``MCPURIError``.
    * ``list_all_resources()`` — aggregates across connected
      servers; adds ``server`` and ``mcp_uri`` keys per entry.

### Added — S8.4 MCP prompts → Skills bridge (PR #106)

- Per-connection (``MCPServerConnection``):
    * ``list_prompts()`` — returns
      ``[{name, description, arguments: [{name, description, required}]}]``.
    * ``get_prompt(name, arguments)`` — returns
      ``[{role, content}]`` message list. Both failure-isolated
      like the resource API (returns empty/None with WARN log).
- Manager (``MCPManager``):
    * ``list_all_prompts()`` — aggregates across connected
      servers; adds ``server`` key.
    * ``get_mcp_prompt(server, name, arguments)`` — routes;
      ``None`` for unknown / disconnected.
- ``geny_executor.skills.mcp_bridge`` module:
    * ``mcp_skill_id(server, prompt)`` →
      ``"mcp__<server>__<prompt>"``.
    * ``mcp_prompts_to_skills(manager)`` → ``List[Skill]`` with
      ``extras = {server, prompt_name, arguments, source="mcp"}``.
      Per-server failure isolation. Body is a short placeholder;
      hosts wanting prompt-as-tool routing subclass
      ``SkillTool`` and look up the call target via
      ``metadata.extras``.
    * ``MCP_SKILL_ID_PREFIX`` / ``MCP_SKILL_SOURCE_TAG``
      constants re-exported.

### Compatibility

Additive only. Existing per-connection ``list_resources`` /
``read_resource`` / tool-discovery surfaces and the FSM (S6.x
shipped earlier) are unchanged. Hosts that don't construct an
``OAuthFlow`` or call any of the new manager helpers see zero
functional change.

### Phase 8 summary

Four sprints in one release: a pluggable credential store +
full OAuth 2.0 authorization-code flow + ``mcp://`` URI scheme +
prompts→Skills bridge. Phase 9 (the 21-stage reconstruction —
the largest structural change in the uplift) follows.

---

## [0.43.0] — 2026-04-25

**Closes Phase 7 of the executor uplift roadmap.** Bundles the final
two sprints (S7.11 + S7.12) into one minor release. Both are
independently opt-in; without consuming the new surfaces, behaviour
is identical to 0.42.x.

### Added — S7.11 Stage 14 Emit (PR #100)

- ``Emitter`` ABC gains two optional class-level scheduling hints:
  ``requires: Tuple[str, ...]`` (names of emitters that must
  succeed first) and ``timeout_seconds: Optional[float]`` (per-emit
  wall-clock budget). Both default to "no constraint" so existing
  emitters keep working unchanged.
- ``OrderedEmitterChain`` (new class alongside the unchanged
  legacy ``EmitterChain``) honours those hints:
    * Topological order via Kahn's algorithm. Cycles fall back to
      declared order with an ``emit.cycle_detected`` event.
      Unknown deps emit ``emit.unknown_dependency`` and are
      dropped from the dep set so a typo cannot wedge the whole
      chain.
    * Dep-failure skip — dependents whose required emitters did
      not ``emitted=True`` are skipped with metadata
      ``{"skipped": "dep_failed", "deps": [...]}`` and an
      ``emit.skipped_dep_failed`` event.
    * Timeout-based backpressure — per-emitter consecutive-timeout
      counter. Once it reaches ``backpressure_threshold`` (default
      3), the emitter is skipped (metadata.skipped="backpressure")
      with an ``emit.skipped_backpressure`` event until success or
      :meth:`reset_backpressure`. Non-timeout exceptions don't
      count toward backpressure (correctness bugs ≠ latency).
- ``EmitResult`` gains ``emitter_name: str = ""`` for clean
  result→producer pairing. Legacy chain leaves it blank;
  ``OrderedEmitterChain`` populates it on every result.

### Added — S7.12 Stage 16 Yield (PR #101)

- ``MultiFormatFormatter(formats=…, include_thinking=False)`` —
  produces text + structured + markdown payloads in one pass.
  ``state.final_output`` becomes a dict keyed by the requested
  format names; consumers pick whichever they need without
  re-running the pipeline.
- ``include_thinking`` toggle folds the most recent thinking turn
  from ``state.thinking_history`` into the markdown output (off
  by default — matches existing privacy posture).
- Public helpers ``build_structured(state)`` (same shape as
  ``StructuredFormatter``) and ``build_markdown(state,
  include_thinking=False)`` (`# Result` / optional `## Thinking`
  / optional `## Status` / metadata footer) for hosts that want
  the payloads without going through a formatter.
- ``YieldStage``'s formatter slot registry now exposes
  ``"multi_format"``.

### Compatibility

Additive only. No default slot strategy or chain class changes —
existing pipelines see zero functional change. ``EmitterChain``
and the legacy formatters (``Default`` / ``Structured`` /
``Streaming``) are unchanged; the new ``OrderedEmitterChain`` and
``MultiFormatFormatter`` are alternatives, not replacements.

### Phase 7 summary

Twelve sprints across nine stages, shipped over six minor releases
(0.38–0.43). Every stage now ships at least one new strategy slot
or class-level extension surface, all opt-in, all backward-
compatible. Phase 8 (MCP Advanced) and Phase 9 (21-stage
reconstruction) are next.

---

## [0.42.0] — 2026-04-25

Phase 7 sprint batch — three more stage enhancements bundled into
one minor release. Each is independently opt-in; without consuming
the new surfaces, behaviour is identical to 0.41.x.

### Added — S7.8 Stage 6 API (PR #96)

- ``ModelRouter`` Strategy ABC in
  ``geny_executor.stages.s06_api.interface`` — single
  ``route(cfg, state) -> Optional[ModelConfig]`` method.
- ``PassthroughRouter`` (default, no-op) and ``AdaptiveModelRouter``
  ship as built-in artifact registry entries. Adaptive picks
  Opus / Sonnet / Haiku tiers from lightweight heuristics:
  ``thinking_enabled`` → heavy, character-count thresholds →
  heavy/light, tools-on-state → balanced. Tier model names and
  thresholds are constructor-tunable.
- ``APIStage`` gains a third strategy slot ``router``. Slot lookup
  exposes ``"passthrough"`` / ``"adaptive"``.
- ``APIStage.execute()`` runs the slot via a new
  ``_route_model(state)`` helper that emits ``api.model_routed``
  on actual swaps and ``api.router.error`` if the router raises
  (call is never blocked). State is not mutated — the override
  applies only to the call.

### Added — S7.9 Stage 15 Memory (PR #97)

- ``geny_executor.stages.s15_memory.insight`` module with the
  ``record_insight()`` / ``coerce_insight()`` /
  ``drain_pending_insights()`` helpers and the
  ``PENDING_INSIGHTS_KEY`` / ``INSIGHTS_KEY`` ``state.metadata``
  contract. Re-uses the existing
  ``geny_executor.memory.provider.Insight`` + ``Importance`` types
  as the canonical record shape — no parallel hierarchy.
- ``StructuredReflectiveStrategy`` registered as
  ``"structured_reflective"`` in ``MemoryStage``'s strategy slot.
  Drains pending insights, appends to
  ``state.metadata[INSIGHTS_KEY]``, emits ``memory.insight_recorded``
  per record + ``memory.structured_reflection_done`` summary +
  ``memory.insight_invalid`` on coercion failure (queue is always
  cleared so a bad payload cannot wedge subsequent runs). Clears
  the legacy ``needs_reflection`` flag once it processes the queue.

### Added — S7.10 Stage 8 Think (PR #98)

- ``ThinkingBudgetPlanner`` Strategy ABC in
  ``geny_executor.stages.s08_think.interface`` — single
  ``plan(state) -> int`` method.
- ``StaticThinkingBudget`` (default, fixed-value) and
  ``AdaptiveThinkingBudget`` (heuristic-based: base +
  ``tools_bonus`` + ``reflection_bonus`` + size-step bonus per
  ``size_step_chars``, clamped to ``[min_budget, max_budget]``).
- ``apply_thinking_budget(state, planner)`` helper writes the
  planned value back onto ``state.thinking_budget_tokens`` and
  emits ``think.budget_applied {planner, from, to}``.
- ``ThinkStage`` gains a ``budget_planner`` slot (registry:
  ``"static"`` / ``"adaptive"``) and an
  ``apply_planned_budget(state)`` method that hosts call from a
  pre-Stage-6 hook. ``execute()`` itself does **not** auto-invoke
  the planner — Stage 8 only runs after the API response is in hand.
- ``make_planner(adaptive_budget, min_budget, max_budget,
  base_budget)`` factory matches the ``ConfigSchema``-style flags
  from the design doc.

### Compatibility

Additive only. The default slot strategies (``PassthroughRouter``,
``AppendOnlyStrategy``, ``StaticThinkingBudget``) all preserve the
exact pre-0.42.0 behaviour; existing pipelines see zero functional
change.

---

## [0.41.0] — 2026-04-24

Phase 7 sprint batch — three more stage enhancements bundled into
one minor release. Each is independently opt-in; without consuming
the new surfaces, behaviour is identical to 0.40.x.

### Added — S7.5 Stage 11 Agent (PR #92)

- ``geny_executor.stages.s11_agent.subagent_type`` subpackage:
    * ``SubagentTypeDescriptor`` — frozen dataclass: ``agent_type``,
      ``factory`` (sync or async, zero-arg), ``description``,
      ``allowed_tools``, ``model_override``, ``extras``.
    * ``SubagentTypeRegistry`` — id→descriptor map mirroring
      ``ToolRegistry`` (register / unregister / get / list_types /
      contains / len).
    * ``SubagentTypeOrchestrator`` — :class:`AgentOrchestrator`
      subclass that walks ``state.delegate_requests`` against the
      registry, dispatches each, surfaces descriptor metadata on
      every ``sub_result``. Failure-isolated.
- ``AgentStage`` registry now exposes ``"subagent_type"``.

### Added — S7.6 Stage 12 Evaluate (PR #93)

- ``EvaluationChain([ev1, ev2, ...])`` — sequential evaluator
  composition. Runs evaluators in declared order; first
  ``decision != "continue"`` wins (short-circuit). Empty chain →
  benign ``complete`` no-op. Failure-isolated.
- ``EvaluateStage`` registry now exposes ``"evaluation_chain"``.

### Added — S7.7 Stage 13 Loop (PR #94)

- ``BudgetDimension`` ABC + five built-in dimensions:
  ``IterationBudget``, ``CostBudget``, ``TokenBudget``,
  ``WallClockBudget``, ``ToolCallBudget``.
- ``MultiDimensionalBudgetController([dims...])`` — replaces the
  fixed-two-dimension ``BudgetAwareLoopController`` with a
  pluggable registry. First exceeded dimension wins;
  ``last_exceeded_dimension`` exposed for observability.
- ``LoopStage`` registry now exposes ``"multi_dim_budget"``.

### Compatibility

Additive only. ``DelegateOrchestrator`` /
``BudgetAwareLoopController`` and the existing single-evaluator
strategy slot all keep working. Hosts opt into the new surfaces by
constructing them and swapping into the relevant Stage's strategy
slot.

Full unit suite: 1317 passed, 1 skipped.

## [0.40.0] — 2026-04-24

Phase 7 sprint batch — three stage enhancements bundled into one
minor release. Each is independently opt-in; without consuming the
new surfaces, dispatch + prompt assembly + parsing all behave
identically to 0.39.x.

### Added — S7.1 Stage 3 System (PR #88)

- ``geny_executor.stages.s03_system.persona`` subpackage:
    * ``PersonaResolution`` (frozen dataclass) — single-turn snapshot
      of ``persona_blocks`` + ``system_tail`` + ``cache_key``.
    * ``PersonaProvider`` — ``@runtime_checkable`` Protocol; sync
      ``resolve(state, *, session_meta) → PersonaResolution``.
    * ``DynamicPersonaPromptBuilder`` — calls the provider on every
      build and composes through the inner
      ``ComposablePromptBuilder``. Holds no persona state itself, so
      provider mutations are visible on the next turn without
      rebuilding the pipeline.
- ``SystemStage`` strategy registry now includes ``"dynamic_persona"``
  alongside ``"static"`` / ``"composable"``. Hosts attach the
  builder via ``Pipeline.attach_runtime(system_builder=...)``.

### Added — S7.2 Stage 2 Context (PR #89)

- ``MCPServerConnection.list_resources()`` + ``read_resource(uri)``
  — async wrappers around the SDK's resource API. Fail-open with
  WARNING logs on transport / protocol failures.
- ``geny_executor.stages.s02_context.MCPResourceRetriever`` —
  ``MemoryRetriever`` subclass that lists / filters / reads MCP
  resources (the second MCP primitive after tools) and wraps each
  match as a ``MemoryChunk(source="mcp_resource")``. Global
  ``max_resources`` cap (default 5) shared across all servers;
  per-server / per-URI failures isolated.

### Added — S7.3 Stage 9 Parse (PR #90)

- ``ParsedResponse.structured_output_error: Optional[str]`` — new
  field that disambiguates the three structured-output outcomes:
  ``None`` (clean / absent), ``"JSON parse failed: ..."`` (text
  wasn't JSON), or ``"schema mismatch at <path>: ..."`` (JSON
  parsed but didn't match the bound schema).
- ``StructuredOutputParser(schema=...)`` — validates the schema at
  construction time (bad schema → ``ValueError``) and the parsed
  payload at parse time. Validation failure clears
  ``structured_output`` to ``None`` so downstream stages don't see
  partially-trusted data.

### Compatibility

Additive only:

* Hosts that don't construct ``DynamicPersonaPromptBuilder`` get
  the same ``StaticPromptBuilder`` / ``ComposablePromptBuilder``
  default they had at 0.39.x.
* Hosts that don't attach an ``MCPResourceRetriever`` see no
  Stage 2 behaviour change.
* ``StructuredOutputParser`` without a schema preserves the legacy
  best-effort parse — only the new ``structured_output_error``
  field carries extra disambiguation.

Full unit suite: 1247 passed, 1 skipped.

## [0.39.0] — 2026-04-24

Phase 7 Sprint S7.4 — Permission matrix lands in dispatch. The
``PermissionRule`` + ``evaluate_permission`` substrate has been part
of the codebase since 0.32.0 (Phase 1) but no consumer fired it.
Stage 10's ``RegistryRouter`` now consults the matrix on every tool
call before any subprocess hooks run, so a DENY decision short-
circuits the entire pipeline.

### Added (PR #86)

- ``ToolContext.permission_rules`` — new optional list field.
- ``Pipeline.attach_runtime(permission_rules=..., permission_mode=...)``
  — both kwargs, independently updatable.
- ``ToolStage.execute`` propagates rules + mode into the per-call
  ``ToolContext``.
- ``RegistryRouter._dispatch_with_lifecycle`` calls
  ``evaluate_permission`` between input validation and
  ``PRE_TOOL_USE`` hook firing. ``DENY`` returns ``ACCESS_DENIED``;
  ``ASK`` is treated as ``DENY`` for safety until the Phase 9 HITL
  stage lands. ``ALLOW`` proceeds (re-validating
  ``decision.updated_input`` if the matrix rewrote it). ``BYPASS``
  mode short-circuits even ``DENY`` rules (developer escape hatch).

### Compatibility

Without ``permission_rules`` attached, dispatch is byte-identical to
0.38.x. Mode coercion (``str`` → ``PermissionMode``) is forgiving:
unknown values fall back to ``DEFAULT`` rather than raising.

Full unit suite: 1183 passed, 1 skipped.

## [0.38.0] — 2026-04-24

Phase 6 — MCP uplift. Replaces the per-server boolean
``is_connected`` with a five-state finite-state machine, adds an
admin disable / enable lifecycle, maps MCP tool annotations onto
``ToolCapabilities`` so PartitionExecutor can fan read-only MCP
tools out in parallel, and lets hosts swap a live ``MCPManager``
into a built pipeline via ``attach_runtime``.

### Added — connection FSM (PR #83)

- ``MCPConnectionState`` (``geny_executor.tools.mcp.state``) — five
  states: ``PENDING`` / ``CONNECTED`` / ``FAILED`` / ``NEEDS_AUTH``
  / ``DISABLED``.
- ``MCPServerConnection.state`` + ``last_error`` properties.
  ``is_connected`` is now derived (``state == CONNECTED``).
- Auth-shaped failures classified into ``NEEDS_AUTH`` so admin UIs
  can prompt for credentials instead of retrying blindly.
- ``MCPManager.disable_server(name)`` + ``enable_server(name)`` —
  admin lifecycle that retains config across the toggle. Distinct
  from ``disconnect`` (which evicts).
- ``list_server_status()`` includes ``state`` + ``last_error``;
  ``connected`` boolean retained for back-compat.

### Added — annotation → ToolCapabilities mapping (PR #84)

- ``MCPToolAdapter.capabilities(input)`` reads MCP annotations and
  returns a populated ``ToolCapabilities``. Mapping:
  ``readOnlyHint=True`` → ``read_only`` + ``concurrency_safe``;
  ``destructiveHint=True`` → ``destructive`` (overrides
  ``concurrency_safe``); ``idempotentHint=True`` → ``idempotent``;
  ``openWorldHint=True`` → ``network_egress``.
- ``manager._serialise_mcp_tool`` captures ``annotations`` from each
  SDK tool object (object-attr OR dict form supported).

### Added — pipeline integration (PR #84)

- ``Pipeline.attach_runtime(mcp_manager=...)`` — kwarg accepts a
  pre-built ``MCPManager``. Replaces any manifest-built manager and
  re-seeds the pipeline's ``tool_registry`` from the manager's
  CONNECTED servers. Skips DISABLED / FAILED / NEEDS_AUTH; never
  clobbers existing entries with the same prefixed name.

### Compatibility

Without using any of the new surfaces, dispatch is byte-identical to
0.37.x — all changes are additive. Hosts that hand-set
``conn._connected = True`` in tests need to migrate to
``conn._state = MCPConnectionState.CONNECTED`` (the new field is the
backing for ``is_connected``).

Full unit suite: 1171 passed, 1 skipped.

## [0.37.0] — 2026-04-24

Phase 5 — subprocess hooks land. The Phase 1 hook event taxonomy
(``HookEvent`` / ``HookEventPayload`` / ``HookOutcome``) was always
the half of the contract sitting in core; this release adds the
runtime + Stage 10 wiring that actually fires user-configured hook
scripts around tool dispatch.

### Added — hook runner (PR #80)

- ``geny_executor.hooks.runner.HookRunner`` — spawns subprocess
  hooks via ``asyncio.create_subprocess_exec`` (never
  ``shell=True``), serialises ``HookEventPayload`` to stdin as
  JSON, parses stdout into a ``HookOutcome``. Multiple matching
  hooks combine via ``HookOutcome.combine`` (most-restrictive
  wins) and short-circuit once blocked.
- ``geny_executor.hooks.config`` — ``HookConfigEntry`` /
  ``HookConfig`` / ``parse_hook_config`` / ``load_hooks_config``.
  YAML loader with location-suffixed validation errors and
  forward-compat skip for unknown event names.
- Two-switch opt-in: both ``HookConfig.enabled = True`` AND
  ``GENY_ALLOW_HOOKS=1`` env required to invoke any subprocess.
- Per-entry ``timeout_ms`` (default 5000ms) enforced via
  ``asyncio.wait_for`` — overruns kill the process and fail-open
  passthrough so a slow hook never blocks the agent.
- Every failure mode (command not found, non-zero exit, non-JSON
  stdout, permission denied, generic spawn error) → fail-open
  passthrough + WARNING log. Pipeline never dies on a broken hook.
- Optional JSONL audit log (``audit_log_path``) + per-invocation
  async callback (``HookRunner.set_audit_callback``).

### Added — Stage 10 wiring (PR #81)

- ``ToolContext.hook_runner`` field — typed ``Any`` to keep
  ``tools/base.py`` import-cycle-free.
- ``Pipeline.attach_runtime(hook_runner=...)`` — hosts construct
  one ``HookRunner`` (per session typically) and attach it before
  the first run. Threaded through the Tool stage's context to the
  per-call ctx Stage 10 builds.
- ``RegistryRouter._dispatch_with_lifecycle`` now fires
  ``PRE_TOOL_USE`` before ``execute``, honouring ``blocked``
  (returns ``ACCESS_DENIED`` short-circuit) and ``modified_input``
  (re-validated against the tool's input schema, then used as the
  payload). On the way out it fires ``POST_TOOL_USE`` for clean
  results and ``POST_TOOL_FAILURE`` for both soft errors
  (``is_error=True``) and unexpected exceptions — unified
  observation channel for hooks that audit failures.

### Compatibility

Without a ``hook_runner`` bound, dispatch is byte-identical to
0.36.x. With a runner attached but neither switch flipped (``enabled``
or env), the runner short-circuits to passthrough — nothing actually
spawns. So even an accidentally-attached runner is safe.

Full unit suite: 1122 passed, 1 skipped.

## [0.36.1] — 2026-04-24

Hotfix patch. The lifecycle-hook dispatcher shipped in 0.33.0 (PR #61)
called ``tool.on_enter(...)`` / ``on_exit(...)`` / ``on_error(...)``
directly — fine for every proper ``Tool`` ABC subclass (which inherits
no-op defaults) but it crashed for host-supplied adapters that
implement the structural Tool interface without inheriting from the
ABC. Geny's ``_GenyToolAdapter`` is the canonical example: it exposes
``name`` / ``description`` / ``input_schema`` / ``execute`` but has
never declared lifecycle methods.

Observed error in the field:

    '_GenyToolAdapter' object has no attribute 'on_enter'

### Fixed

- ``stages/s10_tool/artifact/default/routers.py`` — ``_fire_hook`` now
  looks up lifecycle methods via ``getattr`` with a safe fallback.
  Hooks that are absent, ``None``, or otherwise non-callable are
  silently skipped; synchronous hook bodies are detected and awaited
  only when the return value is awaitable. Callers (``RegistryRouter.
  _dispatch_with_lifecycle``) pass the tool + hook name + args instead
  of materialising the coroutine at the call site, so an attribute
  miss can't escape the router's try/except boundary.

### Tests

Five new regression tests in ``test_tool_lifecycle_hooks.py`` covering
duck-typed tools without hook attrs (happy path, ``ToolFailure``
exception path, unexpected ``Exception`` path), a non-callable
``on_enter`` attribute, and a synchronous ``on_exit`` hook. Full unit
suite: 1075 passed, 1 skipped.

### Compatibility

Zero API surface change. Any tool previously working continues to
work. Host adapters that lacked lifecycle methods but were crashing
on 0.33.x–0.36.0 now run cleanly.

## [0.36.0] — 2026-04-24

Phase 4 Weeks 7-8 — Skills system ships in inline-execution form.

### Added — Skills foundation (PR #76)

- New `geny_executor.skills` subpackage:
  - `Skill` / `SkillMetadata` / `SkillContext` dataclasses.
  - `parse_frontmatter(text) → (dict, body)` — stdlib + pyyaml
    `safe_load`. Handles missing delimiters, non-dict top-level
    values, and invalid YAML with explicit "no frontmatter"
    semantics so malformed skills surface at the loader layer.
  - `parse_skill_file(path)` / `load_skills_dir(root)` — one-SKILL.md
    and bulk loaders. Bulk load returns `SkillLoadReport(loaded,
    errors)`; `strict=True` re-raises the first error.
  - `SkillRegistry` — flat id→Skill map, duplicate rejected with
    `ValueError`, explicit `unregister` for override semantics.
- New core dependency: **pyyaml>=6.0**.

### Added — SkillTool integration (PR #77)

- `SkillTool(skill)` — exposes one Skill as a callable Tool. Tool
  name = skill id; description = skill description + `[skill, mode]`
  tag. Uniform `{args: object}` input schema across every skill.
- `SkillToolProvider(registry, name=...)` — subclass of the Phase 3
  `ToolProvider` Protocol. Plug into
  `Pipeline.from_manifest_async(tool_providers=[...])` to expose
  every registered skill as a tool.
- Inline execution mode: the tool returns the rendered skill body
  with a compact header (skill name, version, allowed_tools,
  model_override). The LLM reads the body as instructions and
  executes the steps using its existing tool roster.
- Fork execution mode stubbed: skills marked `execution_mode: fork`
  fail fast with a clean "not yet available in this release" error,
  pending the Phase 7 AgentTool runtime.
- `{placeholder}` template interpolation over `invoke_args` with a
  safe-fallback dict — unknown placeholders and malformed format
  specs pass through unchanged.

### Notes

Full unit suite: 1070 passed, 1 skipped. Additive — existing hosts
don't need to consume the Skills subsystem unless they want to.

## [0.35.0] — 2026-04-24

Phase 3 Week 7 release — closes Phase 3 with the ``ToolProvider``
Protocol, the architectural cornerstone for pluggable tool sources.

### Added

- **`ToolProvider` ABC** (`geny_executor.tools.provider`) —
  self-contained, lifecycle-aware tool bundles. Where
  `AdhocToolProvider` is name-keyed lookup, `ToolProvider` is a full
  feature pack: the provider owns its name, its tool roster, and
  optional ``startup`` / ``shutdown`` hooks.
- **`BuiltInToolProvider(features=..., names=...)`** — first concrete
  provider, wraps the executor's built-in catalogue via
  `get_builtin_tools`. Hosts can opt into the whole catalogue or a
  feature-gated subset.
- **`register_providers` / `shutdown_providers`** — the registration
  helpers. Duplicate provider names raise; tool name collisions
  within the registry log + skip (first provider wins); startup
  failures unwind every previously started provider before re-raising.
- **`Pipeline.from_manifest_async(tool_providers=[...])`** — new
  kwarg accepts the provider list. Registration happens after
  manifest-declared built-ins + adhoc providers, before MCP adapter
  discovery, so manifest authority wins on conflicts. MCP bring-up
  failure now also unwinds any started providers (atomic).
- **`pipeline.tool_providers`** property + **`pipeline.shutdown_tool_providers()`**
  for host-driven teardown.

### Why this matters

Hosts that bundle their own tools (Geny's creature / feed / knowledge
suite, third-party plugins, MCP facades) no longer need to enumerate
tool names in every manifest. They ship a single `XToolProvider`
class, the host imports and configures it, the pipeline does the rest.
This is the "geny-executor first" principle made concrete at the
plugin boundary.

### Notes

Full unit suite: 1008 passed, 2 skipped. Purely additive — existing
`from_manifest_async` callers that don't pass `tool_providers=` see
no behaviour change.

## [0.34.0] — 2026-04-24

Phase 3 release — built-in tool catalog expands from 6 → 13 tools and
the Phase 1 `state_mutations` contract finally lands in state. Scope
is deliberately additive: hosts upgrading from 0.33.x that don't
consume any of the new tools see no behaviour change.

### Added — built-in tool catalog (now 13 tools)

- **`WebFetch`** (PR #65) — HTTP(S) fetcher with stdlib HTML → text
  extraction. `concurrency_safe=True` + `read_only=True` +
  `network_egress=True`. Body cap (1 MiB default), text cap (80 000
  chars default), 5-hop redirect limit, 30 s default timeout. Scheme
  allowlist rejects `file://` / `ftp://` / data URIs.
- **`WebSearch`** (PR #66) — DuckDuckGo text search via the new
  `[web]` optional extra (`ddgs>=9.11`). Missing dep → clean
  "pip install 'geny-executor[web]'" hint; never crashes at import.
  Hard cap 30 results, region + safesearch forwarded.
- **`TodoWrite`** (PR #68) — Claude Code-style task list updates.
  Full-list rewrite semantics, stable IDs derived from position +
  content, Markdown checklist rendering. Introduces the `workflow`
  feature family.
- **`NotebookEdit`** (PR #70) — `.ipynb` cell editing (replace /
  insert / delete) via stdlib JSON. Atomic writes (temp file →
  fsync → os.replace), `save=false` dry-run mode, code-cell outputs
  cleared on replace.
- **`ToolSearch`** (PR #71) — keyword discovery over the live tool
  catalogue. Reads `state_view.tools` (set by `ToolStage`), falls
  back to `BUILT_IN_TOOL_CLASSES`. Ranked matches (exact name > name
  substring > description > schema). Introduces the `meta` feature
  family.
- **`EnterPlanMode` / `ExitPlanMode`** (PR #72) — toggle the public
  `executor.plan_mode` flag on `state.shared` via the state_mutations
  contract. Stage 4 Guard can consult the flag to block destructive
  tools during planning.

### Added — selection + typing

- **`BUILT_IN_TOOL_FEATURES`** + **`get_builtin_tools(features=...,
  names=...)`** (PR #67) — programmatic feature-gated selection API
  complementing the declarative `manifest.tools.built_in` path. Every
  built-in tool belongs to exactly one feature family
  (`filesystem` / `shell` / `web` / `workflow` / `meta`), enforced by
  a structural test.

### Added — capability flags on existing built-ins (PR #64)

- `Read` / `Grep` / `Glob` now advertise `concurrency_safe=True` +
  `read_only=True` + `idempotent=True`. Under `PartitionExecutor` /
  `StreamingToolExecutor` these fan out in parallel instead of
  serialising.
- `Write` / `Edit` / `Bash` keep the fail-closed default (unsafe) —
  they mutate state or run arbitrary commands.

### Added — state_mutations wiring (PR #69)

- `ToolResult.state_mutations` — the dict of proposed updates to
  `state.shared` that tools return — now actually flows into state
  across all four Stage 10 executors.
- New `ToolContext.state_apply` callback (set by `ToolStage` from a
  closure over `state.shared`) + `state_view` handle (for read-only
  introspection, wired for `ToolSearch`).
- Namespace allowlist: `executor.` / `memory.` / `geny.` /
  `plugin.<ns>.` only; unknown prefixes logged and dropped. Skipped
  on `is_error=True` results so failing tools don't leak half-written
  state.

### Dependencies

- `httpx>=0.27` declared as a core dependency (was already transitive
  via `anthropic`; now explicit because `WebFetch` imports it).
- New `[web]` optional extra pulls `ddgs>=9.11` for `WebSearch`.
- Added to `[dev]` too so the full test suite runs without the extra.

### Notes

Full unit suite: 990 passed, 2 skipped (up from 844 at 0.33.0).

Carried over from the 0.33.x line without change. No call-site
migrations required. Hosts on 0.33.x that set
`ToolContext(storage_path=...)` (as Geny does) automatically get the
persistence + state_mutations behaviour for any tool that returns them.

## [0.33.0] — 2026-04-24

Phase 2 Orchestration release — completes Week 4 checkpoints on top of
the 0.32.x Phase 1 foundation. Stage 10 (Tool) gains streaming
execution, automatic result persistence, lifecycle hooks around every
tool dispatch, and a stage-level concurrency budget knob.

### Added

- **StreamingToolExecutor** (PR #59, `stages/s10_tool/streaming.py`) —
  online variant of `PartitionExecutor`. Exposes an `add()` / `drain()`
  interface so hosts integrating with streaming LLM responses can kick
  off concurrency-safe tools as `tool_use` blocks arrive, then collect
  results in receive order on drain. Unsafe calls raise a chain barrier
  the moment they queue so subsequent safe calls wait. 14 new unit
  tests cover ordering, bounded parallelism, fail-closed metadata
  lookup, event emission, and the safe/unsafe/safe interleave pattern.

- **Tool result persistence** (PR #60, `stages/s10_tool/persistence.py`)
  — `maybe_persist_large_result` inspects each `ToolResult.content`
  against the tool's resolved `ToolCapabilities.max_result_chars`. When
  exceeded, writes a JSON envelope to
  `{storage_path}/tool-results/{tool_use_id}.json` and returns a new
  `ToolResult` with a short `display_text` + the path in `persist_full`.
  Wired into all four Stage 10 executors. Fail-open: missing
  `storage_path` / `OSError` → original payload returned with a warning
  log. 16 new tests including integration through each executor.

- **Tool lifecycle hook wiring** (PR #61,
  `stages/s10_tool/artifact/default/routers.py`) — `RegistryRouter` now
  fires `on_enter` → execute → `on_exit` (or `on_error` on raise). A
  `ToolResult` with `is_error=True` is still a normal return, so
  `on_exit` fires and observes the flag. All hook failures are logged
  and swallowed so a misbehaving hook never masks a successful tool
  call or blocks the next lifecycle event. 9 new tests.

- **Stage-level `max_concurrency` knob** (PR #62,
  `stages/s10_tool/artifact/default/stage.py`) — `ToolStage(max_concurrency=N)`
  ctor arg + ConfigSchema integer field (min 1, max 64). `update_config`
  propagates the value onto the active executor; re-applied on every
  `execute()` call so swapped-in executors inherit the budget instead
  of reverting to their class default. 12 new tests.

### Changed

- `ToolContext.storage_path` is now used by Stage 10 executors for
  tool-result persistence. When absent, behaviour is identical to
  0.32.x (inline full payload, warn on oversize).

### Notes

Full unit suite: 844 passed, 2 skipped. Functionally additive — hosts
upgrading from 0.32.x without consuming any of the new surfaces see
no behaviour change. Phase 3 (built-in tool catalog) begins in the
next minor.

## [0.32.3] — 2026-04-24

Patch release — applies `ruff format` to bring Phase 1 additions onto
the repo's canonical style. No semantic changes. `ruff check` +
`ruff format --check` now both pass on CI. This is the first Phase 1
release that is green across all three supported Python versions
(3.11 / 3.12 / 3.13) AND both lint jobs.

### Fixed

- 9 files reformatted (PR #57): whitespace / wrapping / trailing
  comma adjustments only. Affected the Phase 1 uplift additions
  (`tools/base.py`, `permission/*`, `hooks/events.py`, Stage 10
  executors) plus three pre-existing files that had drifted from the
  canonical format prior to this release.

### Notes

Consolidates 0.32.0 → 0.32.3 into a single publishable line. 0.32.0
/ 0.32.1 / 0.32.2 tags exist but were never published to PyPI due to
progressively discovered CI issues. Functionally identical to 0.32.0
for anyone consuming the library.

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
