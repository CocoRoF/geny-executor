# Event catalogue

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate: python scripts/gen_event_docs.py -->

> Generated from `geny_executor.events.catalog` on 2026-06-15.
> Catalogue version: **3** · events: **109**

Every event name the engine emits, value == wire string. The enum
is a *names registry*, not a rename — consumers matching raw strings
keep working. Contract rules (see the module docstring for the full
statement):

- **Append-only**: new events arrive in minor releases; renaming or
  removing a member is a major-version change.
- **Payloads may gain fields** in minor releases; existing fields keep
  their meaning. Field docs below are descriptive, not strict schemas.
- `…?` marks fields present only in some emissions of the event.

Consume via `pipeline.on(event_type, handler)`, `pipeline.run_stream(...)`,
or the multi-subscriber tap `pipeline.events(replay_from=...)` (2.2.0).
`llm_client.*` events travel through the client's `event_sink` callback.

## Pipeline lifecycle

### `pipeline.start`

Enum member: `EventTypes.PIPELINE_START`

| Field | Description |
|---|---|
| `input` | str — the user turn, truncated to Pipeline.EVENT_DATA_TRUNCATE chars |

### `pipeline.complete`

Enum member: `EventTypes.PIPELINE_COMPLETE`

| Field | Description |
|---|---|
| `iterations` | int — loop iterations this turn |
| `result` | str? — full final text (run_stream only; never truncated) |
| `total_cost_usd` | float? — this turn's cost (run_stream only) |

### `pipeline.error`

Enum member: `EventTypes.PIPELINE_ERROR`

| Field | Description |
|---|---|
| `error` | str — message (legacy field, always present) |
| `code` | str — stable ExecutorErrorCode value ('exec.*'), 'exec.unknown' fallback |
| `exception_type` | str — fully qualified exception class name |
| `total_cost_usd` | float? — this turn's cost (run_stream only) |

## Stage lifecycle

### `stage.enter`

Enum member: `EventTypes.STAGE_ENTER`

_No payload fields — identity is carried on the event envelope_
_(type / stage / iteration / seq / run_id / session_id)._

### `stage.exit`

Enum member: `EventTypes.STAGE_EXIT`

_No payload fields — identity is carried on the event envelope_
_(type / stage / iteration / seq / run_id / session_id)._

### `stage.bypass`

Enum member: `EventTypes.STAGE_BYPASS`

_No payload fields — identity is carried on the event envelope_
_(type / stage / iteration / seq / run_id / session_id)._

### `stage.error`

Enum member: `EventTypes.STAGE_ERROR`

| Field | Description |
|---|---|
| `error` | str — message |
| `code` | str — stable ExecutorErrorCode value |
| `exception_type` | str — fully qualified exception class name |

## Run-start configuration announcements

### `config.override_applied`

Enum member: `EventTypes.CONFIG_OVERRIDE_APPLIED`

| Field | Description |
|---|---|
| `field` | str — ModelOverrides field name |
| `value` | Any — the override value applied for this run |
| `source` | str — 'per_run' |

### `runtime.llm_client_override`

Enum member: `EventTypes.RUNTIME_LLM_CLIENT_OVERRIDE`

| Field | Description |
|---|---|
| `manifest_provider` | str — Stage 6 provider the manifest declared |
| `client_provider` | str — provider of the attached client that overrides it |

## Loop control (Stage 16)

### `loop.force_complete`

Enum member: `EventTypes.LOOP_FORCE_COMPLETE`

| Field | Description |
|---|---|
| `reason` | str — 'max_iterations' \| 'cost_budget' |
| `iteration` | int? — iteration count (max_iterations only) |
| `total_cost_usd` | float? — turn cost (cost_budget only) |
| `budget_usd` | float? — the configured budget (cost_budget only) |

### `loop.continue`

Enum member: `EventTypes.LOOP_CONTINUE`

| Field | Description |
|---|---|
| `iteration` | int |
| `signal` | str\|None — completion signal if any |
| `pending_tools` | int — queued tool calls |
| `has_tool_results` | bool |
| `upstream_decision` | str — loop_decision before the controller ran |

### `loop.complete`

Enum member: `EventTypes.LOOP_COMPLETE`

| Field | Description |
|---|---|
| `iteration` | int |
| `signal` | str\|None |
| `pending_tools` | int |
| `has_tool_results` | bool |
| `upstream_decision` | str |

### `loop.error`

Enum member: `EventTypes.LOOP_ERROR`

| Field | Description |
|---|---|
| `iteration` | int |
| `signal` | str\|None |
| `pending_tools` | int |
| `has_tool_results` | bool |
| `upstream_decision` | str |

### `loop.escalate`

Enum member: `EventTypes.LOOP_ESCALATE`

| Field | Description |
|---|---|
| `iteration` | int |
| `signal` | str\|None |
| `pending_tools` | int |
| `has_tool_results` | bool |
| `upstream_decision` | str |

## Stage 1 — Input

### `input.normalized`

Enum member: `EventTypes.INPUT_NORMALIZED`

| Field | Description |
|---|---|
| `text_length` | int — normalized text length |

## Stage 2 — Context

### `context.built`

Enum member: `EventTypes.CONTEXT_BUILT`

| Field | Description |
|---|---|
| `message_count` | int |
| `memory_refs` | int? — count of attached memory refs (stage form) |
| `estimated_tokens` | int? |
| `chunks` | int? — RetrievalResult.to_event form (provider-driven path) |

### `context.compacted`

Enum member: `EventTypes.CONTEXT_COMPACTED`

| Field | Description |
|---|---|
| `strategy` | str — compactor name/class |
| `trigger` | str? — 'proactive' (Stage 2) \| 'guard' (Stage 4) |
| `messages_before` | int? |
| `messages_after` | int? |
| `saved_tokens_estimate` | int? |

### `context.compaction_failed`

Enum member: `EventTypes.CONTEXT_COMPACTION_FAILED`

| Field | Description |
|---|---|
| `compactor` | str |
| `trigger` | str — 'proactive' \| 'guard' |
| `error` | str |

### `context.compaction_record_failed`

Enum member: `EventTypes.CONTEXT_COMPACTION_RECORD_FAILED`

| Field | Description |
|---|---|
| `compactor` | str |
| `error` | str |

## Stage 18 — Memory (+ Stage 2 compaction)

### `memory.compaction.summarized`

Enum member: `EventTypes.MEMORY_COMPACTION_SUMMARIZED`

| Field | Description |
|---|---|
| `model` | str |
| `provider` | str |
| `old_count` | int — messages before compaction |
| `summary_chars` | int |

### `memory.compaction.llm_failed`

Enum member: `EventTypes.MEMORY_COMPACTION_LLM_FAILED`

| Field | Description |
|---|---|
| `error` | str |
| `compactor` | str |

## Stage 3 — System

### `system.built`

Enum member: `EventTypes.SYSTEM_BUILT`

| Field | Description |
|---|---|
| `prompt_type` | str — 'content_blocks' \| 'string' |
| `prompt_length` | int — characters |

## Stage 4 — Guard

### `guard.check`

Enum member: `EventTypes.GUARD_CHECK`

| Field | Description |
|---|---|
| `passed` | bool |
| `guard_name` | str — guard (or comma-joined chain) name |
| `message` | str |
| `violations` | list? — [{guard_name, message, action}] (chain form) |

### `guard.warn`

Enum member: `EventTypes.GUARD_WARN`

| Field | Description |
|---|---|
| `message` | str |

### `guard.compacting`

Enum member: `EventTypes.GUARD_COMPACTING`

| Field | Description |
|---|---|
| `guard_name` | str — guard that signalled compaction (token_budget) |
| `reason` | str — the guard message |

## Stage 5 — Cache

### `cache.applied`

Enum member: `EventTypes.CACHE_APPLIED`

| Field | Description |
|---|---|
| `strategy` | str — cache strategy class name |
| `system_is_blocks` | bool |
| `cache_key` | str |

## Stage 6 — API (incl. streaming chunk forwarding)

### `api.request`

Enum member: `EventTypes.API_REQUEST`

| Field | Description |
|---|---|
| `model` | str |
| `provider` | str |
| `message_count` | int |
| `has_tools` | bool |
| `has_thinking` | bool |
| `stream` | bool |

### `api.response`

Enum member: `EventTypes.API_RESPONSE`

| Field | Description |
|---|---|
| `stop_reason` | str |
| `text_length` | int |
| `tool_calls` | int |
| `input_tokens` | int |
| `output_tokens` | int |

### `api.retry`

Enum member: `EventTypes.API_RETRY`

| Field | Description |
|---|---|
| `attempt` | int — 1-based attempt that just failed |
| `category` | str — ErrorCategory value |
| `code` | str? — ExecutorErrorCode value (non-stream path) |
| `delay` | float — backoff seconds before next attempt |
| `stream` | bool? — True on the streaming retry path |

### `api.error`

Enum member: `EventTypes.API_ERROR`

| Field | Description |
|---|---|
| `code` | str — stable ExecutorErrorCode value (e.g. 'exec.cli.auth_failed') |
| `category` | str — ErrorCategory value the retry machinery classified |
| `provider` | str — client provider name |
| `cli_version` | str? — CLI version when a CLI-backed client knows it |
| `message` | str — human-readable error text |

### `api.router.error`

Enum member: `EventTypes.API_ROUTER_ERROR`

| Field | Description |
|---|---|
| `router` | str |
| `error` | str |

### `api.model_routed`

Enum member: `EventTypes.API_MODEL_ROUTED`

| Field | Description |
|---|---|
| `router` | str |
| `from` | str — baseline model |
| `to` | str — routed model |

### `api.timeout_unsupported`

Enum member: `EventTypes.API_TIMEOUT_UNSUPPORTED`

| Field | Description |
|---|---|
| `provider` | str |
| `timeout_ms` | int — the configured-but-undeliverable timeout |

### `text.delta`

Enum member: `EventTypes.TEXT_DELTA`

| Field | Description |
|---|---|
| `text` | str — one streamed text chunk |

### `thinking.delta`

Enum member: `EventTypes.THINKING_DELTA`

| Field | Description |
|---|---|
| `text` | str — one streamed extended-thinking chunk |

### `api.tool_use`

Enum member: `EventTypes.API_TOOL_USE`

| Field | Description |
|---|---|
| `id` | str\|None — tool_use block id |
| `name` | str\|None — tool name |
| `input` | dict — tool input (may be partial until input_json_delta completes) |
| `source` | str — 'cli' (executed inside a CLI backend) \| 'api' (Stage 10 will dispatch) \| 'internal' (the Stage 6 internal loop is about to dispatch it) |

### `api.cli_tool_call`

Enum member: `EventTypes.API_CLI_TOOL_CALL`

| Field | Description |
|---|---|
| `id` | str\|None |
| `name` | str\|None |
| `input` | dict |
| `source` | str — always 'cli'; companion to api.tool_use for narrow subscriptions |

### `api.input_json_delta`

Enum member: `EventTypes.API_INPUT_JSON_DELTA`

| Field | Description |
|---|---|
| `delta` | str — partial JSON fragment of the pending tool input |

### `api.content_block_stop`

Enum member: `EventTypes.API_CONTENT_BLOCK_STOP`

_No payload fields — identity is carried on the event envelope_
_(type / stage / iteration / seq / run_id / session_id)._

### `api.tool_result`

Enum member: `EventTypes.API_TOOL_RESULT`

| Field | Description |
|---|---|
| `tool_use_id` | str — id of the tool_use this result answers |
| `content` | Any — tool result content as the backend reported it |
| `is_error` | bool |
| `source` | str — 'cli' \| 'api' \| 'internal' (Stage 6 internal loop dispatched it) |

### `api.internal_loop_capped`

Enum member: `EventTypes.API_INTERNAL_LOOP_CAPPED`

| Field | Description |
|---|---|
| `turns` | int — inner tool turns the loop completed before stopping |
| `reason` | str — 'max_inner_turns' \| 'cost_budget' |

## Stage 7 — Token

### `token.tracked`

Enum member: `EventTypes.TOKEN_TRACKED`

| Field | Description |
|---|---|
| `input_tokens` | int |
| `output_tokens` | int |
| `cache_write` | int |
| `cache_read` | int |
| `cost_usd` | float\|None — this call's cost |
| `total_cost_usd` | float — turn accumulator after this call |

## Stage 8 — Think

### `think.processed`

Enum member: `EventTypes.THINK_PROCESSED`

| Field | Description |
|---|---|
| `thinking_block_count` | int |
| `total_thinking_tokens` | int |

### `think.budget_applied`

Enum member: `EventTypes.THINK_BUDGET_APPLIED`

| Field | Description |
|---|---|
| `planner` | str |
| `from` | int — previous thinking budget |
| `to` | int — newly applied budget |

## Stage 9 — Parse

### `parse.complete`

Enum member: `EventTypes.PARSE_COMPLETE`

| Field | Description |
|---|---|
| `text_length` | int |
| `tool_calls` | int |
| `signal` | str\|None |
| `stop_reason` | str |

## Stage 10 — Tool

### `tool.execute_start`

Enum member: `EventTypes.TOOL_EXECUTE_START`

| Field | Description |
|---|---|
| `count` | int |
| `tools` | list[str] — tool names about to dispatch |

### `tool.execute_complete`

Enum member: `EventTypes.TOOL_EXECUTE_COMPLETE`

| Field | Description |
|---|---|
| `count` | int |
| `errors` | int — results flagged is_error |

### `tool.call_start`

Enum member: `EventTypes.TOOL_CALL_START`

| Field | Description |
|---|---|
| `tool_use_id` | str |
| `name` | str |
| `input` | dict |

### `tool.call_complete`

Enum member: `EventTypes.TOOL_CALL_COMPLETE`

| Field | Description |
|---|---|
| `tool_use_id` | str |
| `name` | str |
| `is_error` | bool |
| `duration_ms` | int |

## Stage 11 — Tool review

### `tool_review.flag`

Enum member: `EventTypes.TOOL_REVIEW_FLAG`

| Field | Description |
|---|---|
| `reviewer` | str — ReviewFlag.to_dict() |
| `severity` | str |
| `message` | str |

### `tool_review.completed`

Enum member: `EventTypes.TOOL_REVIEW_COMPLETED`

| Field | Description |
|---|---|
| `reviewers` | list[str] |
| `flags` | int |
| `tool_calls` | int |
| `tool_results` | int |

### `tool_review.reviewer_error`

Enum member: `EventTypes.TOOL_REVIEW_REVIEWER_ERROR`

| Field | Description |
|---|---|
| `reviewer` | str |
| `error` | str |

## Stage 12 — Agent

### `agent.orchestrate_start`

Enum member: `EventTypes.AGENT_ORCHESTRATE_START`

| Field | Description |
|---|---|
| `orchestrator` | str |
| `delegate_count` | int |

### `agent.orchestrate_complete`

Enum member: `EventTypes.AGENT_ORCHESTRATE_COMPLETE`

| Field | Description |
|---|---|
| `delegated` | bool |
| `sub_result_count` | int |

### `agent.delegations_capped`

Enum member: `EventTypes.AGENT_DELEGATIONS_CAPPED`

| Field | Description |
|---|---|
| `requested` | int — delegate requests queued this turn |
| `cap` | int — the max_delegations limit that truncated them |

## Stage 13 — Task registry

### `task.registered`

Enum member: `EventTypes.TASK_REGISTERED`

| Field | Description |
|---|---|
| `task_id` | str |
| `kind` | str |
| `status` | str — TaskStatus value |

### `task.done`

Enum member: `EventTypes.TASK_DONE`

| Field | Description |
|---|---|
| `task_id` | str |
| `kind` | str |

### `task.failed`

Enum member: `EventTypes.TASK_FAILED`

| Field | Description |
|---|---|
| `task_id` | str |
| `kind` | str |
| `error` | str |

### `task.timeout`

Enum member: `EventTypes.TASK_TIMEOUT`

| Field | Description |
|---|---|
| `task_id` | str |
| `kind` | str |
| `timeout_seconds` | float |

### `task_registry.synced`

Enum member: `EventTypes.TASK_REGISTRY_SYNCED`

| Field | Description |
|---|---|
| `new` | int |
| `by_status` | dict[str, int] |
| `total` | int |

### `task_registry.invalid_payload`

Enum member: `EventTypes.TASK_REGISTRY_INVALID_PAYLOAD`

| Field | Description |
|---|---|
| `payload_repr` | str — repr of the rejected payload, truncated |

### `task_registry.policy_error`

Enum member: `EventTypes.TASK_REGISTRY_POLICY_ERROR`

| Field | Description |
|---|---|
| `policy` | str |
| `error` | str |

## Stage 14 — Evaluate

### `evaluate.start`

Enum member: `EventTypes.EVALUATE_START`

| Field | Description |
|---|---|
| `strategy` | str |

### `evaluate.complete`

Enum member: `EventTypes.EVALUATE_COMPLETE`

| Field | Description |
|---|---|
| `passed` | bool |
| `score` | float\|None |
| `decision` | str |
| `loop_decision` | str |
| `feedback` | str — truncated to 200 chars |

## Stage 15 — HITL

### `hitl.request`

Enum member: `EventTypes.HITL_REQUEST`

| Field | Description |
|---|---|
| `token` | str — HITLRequest.to_dict(); resolve via Pipeline.resume(token, ...) |
| `reason` | str |
| `severity` | str |
| `payload` | dict |

### `hitl.decision`

Enum member: `EventTypes.HITL_DECISION`

| Field | Description |
|---|---|
| `token` | str |
| `decision` | str — approve \| reject \| cancel |
| `via` | str? — resolution channel (Stage 10 permission path) |

### `hitl.no_decision`

Enum member: `EventTypes.HITL_NO_DECISION`

| Field | Description |
|---|---|
| `token` | str |
| `verdict` | str — timeout-policy verdict applied |

### `hitl.timeout`

Enum member: `EventTypes.HITL_TIMEOUT`

| Field | Description |
|---|---|
| `token` | str |
| `timeout_seconds` | float |
| `verdict` | str |

### `hitl.requester_error`

Enum member: `EventTypes.HITL_REQUESTER_ERROR`

| Field | Description |
|---|---|
| `requester` | str |
| `error` | str |

## Stage 17 — Emit

### `emit.start`

Enum member: `EventTypes.EMIT_START`

| Field | Description |
|---|---|
| `emitter_count` | int |
| `channels` | list[str] |

### `emit.complete`

Enum member: `EventTypes.EMIT_COMPLETE`

| Field | Description |
|---|---|
| `channels_emitted` | list[str] |
| `all_emitted` | bool |

### `emit.timeout`

Enum member: `EventTypes.EMIT_TIMEOUT`

| Field | Description |
|---|---|
| `emitter` | str |
| `timeout_seconds` | float\|None |
| `consecutive_timeouts` | int |

### `emit.skipped_backpressure`

Enum member: `EventTypes.EMIT_SKIPPED_BACKPRESSURE`

| Field | Description |
|---|---|
| `emitter` | str |
| `consecutive_timeouts` | int |

### `emit.skipped_dep_failed`

Enum member: `EventTypes.EMIT_SKIPPED_DEP_FAILED`

| Field | Description |
|---|---|
| `emitter` | str |
| `deps` | list[str] — failed dependencies |

### `emit.cycle_detected`

Enum member: `EventTypes.EMIT_CYCLE_DETECTED`

| Field | Description |
|---|---|
| `ordered_count` | int |
| `total` | int |
| `emitters` | list[str] |

### `emit.unknown_dependency`

Enum member: `EventTypes.EMIT_UNKNOWN_DEPENDENCY`

| Field | Description |
|---|---|
| `emitter` | str |
| `dependency` | str |

## Stage 18 — Memory (+ Stage 2 compaction)

### `memory.updated`

Enum member: `EventTypes.MEMORY_UPDATED`

| Field | Description |
|---|---|
| `strategy` | str |

### `memory.persisted`

Enum member: `EventTypes.MEMORY_PERSISTED`

| Field | Description |
|---|---|
| `session_id` | str |
| `message_count` | int |
| `persistence` | str |

### `memory.turn_recorded`

Enum member: `EventTypes.MEMORY_TURN_RECORDED`

| Field | Description |
|---|---|
| `role` | str |
| `bytes` | int |

### `memory.execution_recorded`

Enum member: `EventTypes.MEMORY_EXECUTION_RECORDED`

| Field | Description |
|---|---|
| `receipt` | ExecutionReceipt.to_event() fields (see memory/provider.py) |

### `memory.insight`

Enum member: `EventTypes.MEMORY_INSIGHT`

| Field | Description |
|---|---|
| `insight` | Insight.to_event() fields (see memory/provider.py) |

### `memory.promoted`

Enum member: `EventTypes.MEMORY_PROMOTED`

| Field | Description |
|---|---|
| `ref` | dict — promoted MemoryRef |
| `from_scope` | str |
| `to_scope` | str |

### `memory.reindexed`

Enum member: `EventTypes.MEMORY_REINDEXED`

| Field | Description |
|---|---|
| `spec` | MemoryEvent spec slot — reserved; no engine emitter today |

### `memory.cost`

Enum member: `EventTypes.MEMORY_COST`

| Field | Description |
|---|---|
| `spec` | MemoryEvent spec slot — reserved; no engine emitter today |

### `memory.snapshot`

Enum member: `EventTypes.MEMORY_SNAPSHOT`

| Field | Description |
|---|---|
| `spec` | MemoryEvent spec slot — reserved; no engine emitter today |

### `memory.insight_recorded`

Enum member: `EventTypes.MEMORY_INSIGHT_RECORDED`

| Field | Description |
|---|---|
| `insight` | Insight.to_event() fields |

### `memory.insight_invalid`

Enum member: `EventTypes.MEMORY_INSIGHT_INVALID`

| Field | Description |
|---|---|
| `error` | str |
| `iteration` | int |

### `memory.reflection_queued`

Enum member: `EventTypes.MEMORY_REFLECTION_QUEUED`

| Field | Description |
|---|---|
| `message_count` | int |
| `iteration` | int |

### `memory.structured_reflection_done`

Enum member: `EventTypes.MEMORY_STRUCTURED_REFLECTION_DONE`

| Field | Description |
|---|---|
| `recorded` | int |
| `total` | int |
| `iteration` | int |

### `memory.provider_recorded`

Enum member: `EventTypes.MEMORY_PROVIDER_RECORDED`

| Field | Description |
|---|---|
| `count` | int — turns recorded into the provider |
| `total_messages` | int |

### `memory.retrieve_breakdown`

Enum member: `EventTypes.MEMORY_RETRIEVE_BREAKDOWN`

| Field | Description |
|---|---|
| `query_preview` | str — truncated to 120 chars |
| `layers` | dict — per-layer hit counts |
| `total_chars` | int |
| `chunk_count` | int |
| `slim_mode` | bool |

### `memory.retrieved_empty`

Enum member: `EventTypes.MEMORY_RETRIEVED_EMPTY`

| Field | Description |
|---|---|
| `query_preview` | str |
| `reason` | str |
| `session_id` | str |

## Stage 19 — Summarize

### `summary.written`

Enum member: `EventTypes.SUMMARY_WRITTEN`

| Field | Description |
|---|---|
| `record` | SummaryRecord.to_dict() fields (turn_id, importance, ...) |

### `summary.skipped`

Enum member: `EventTypes.SUMMARY_SKIPPED`

| Field | Description |
|---|---|
| `summarizer` | str |
| `reason` | str |

### `summary.session_closed`

Enum member: `EventTypes.SUMMARY_SESSION_CLOSED`

| Field | Description |
|---|---|
| `chars` | int |
| `turns` | int |
| `decision` | str |

### `summary.session_close_error`

Enum member: `EventTypes.SUMMARY_SESSION_CLOSE_ERROR`

| Field | Description |
|---|---|
| `error` | str |

### `summary.importance_error`

Enum member: `EventTypes.SUMMARY_IMPORTANCE_ERROR`

| Field | Description |
|---|---|
| `importance` | str |
| `error` | str |

### `summary.provider_recorded`

Enum member: `EventTypes.SUMMARY_PROVIDER_RECORDED`

| Field | Description |
|---|---|
| `turn_id` | str |
| `importance` | str |

### `summary.provider_error`

Enum member: `EventTypes.SUMMARY_PROVIDER_ERROR`

| Field | Description |
|---|---|
| `error` | str |

### `summary.summarizer_error`

Enum member: `EventTypes.SUMMARY_SUMMARIZER_ERROR`

| Field | Description |
|---|---|
| `summarizer` | str |
| `error` | str |

## Stage 20 — Persist

### `checkpoint.written`

Enum member: `EventTypes.CHECKPOINT_WRITTEN`

| Field | Description |
|---|---|
| `checkpoint_id` | str |
| `session_id` | str |
| `iteration` | int |
| `persister` | str |

### `checkpoint.skipped`

Enum member: `EventTypes.CHECKPOINT_SKIPPED`

| Field | Description |
|---|---|
| `frequency` | str |
| `iteration` | int |

### `checkpoint.persister_error`

Enum member: `EventTypes.CHECKPOINT_PERSISTER_ERROR`

| Field | Description |
|---|---|
| `persister` | str |
| `error` | str |

## Stage 21 — Yield

### `yield.complete`

Enum member: `EventTypes.YIELD_COMPLETE`

| Field | Description |
|---|---|
| `text_length` | int |
| `iterations` | int |
| `total_cost_usd` | float |

### `yield.summary`

Enum member: `EventTypes.YIELD_SUMMARY`

| Field | Description |
|---|---|
| `text_length` | int |
| `iterations` | int |
| `total_cost_usd` | float |

## llm_client event_sink channel (boundary telemetry)

### `llm_client.feature_unsupported`

Enum member: `EventTypes.LLM_CLIENT_FEATURE_UNSUPPORTED`

| Field | Description |
|---|---|
| `provider` | str |
| `field` | str — request field the client cannot honour |

### `llm_client.parameter_dropped`

Enum member: `EventTypes.LLM_CLIENT_PARAMETER_DROPPED`

| Field | Description |
|---|---|
| `provider` | str |
| `field` | str |
| `value` | Any — the discarded value |

### `llm_client.drift_healed`

Enum member: `EventTypes.LLM_CLIENT_DRIFT_HEALED`

| Field | Description |
|---|---|
| `provider` | str |
| `model` | str |
| `field` | str — request field rebuilt after a vendor 400 |
| `message` | str — the vendor error that named the problem |

### `llm_client.unknown_wire_shape`

Enum member: `EventTypes.LLM_CLIENT_UNKNOWN_WIRE_SHAPE`

| Field | Description |
|---|---|
| `provider` | str |
| `unknown_type` | str\|None — first unrecognised line type |
| `count` | int — unknown + malformed lines |
| `unknown_line_count` | int |
| `malformed_line_count` | int |
| `cli_version` | str |

---

Companion docs: [error_codes.md](error_codes.md) for the `code` values
carried by error events; [architecture.md](architecture.md) for where
each stage sits in the 21-stage layout.
