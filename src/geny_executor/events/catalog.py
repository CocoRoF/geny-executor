"""EventTypes — the published, versioned catalogue of engine event names.

Why this exists (2.2.0, audit 2026-06-09 §3.2): the event stream was the
de-facto host contract — Geny and GAPT both built their UIs on it — but
the taxonomy lived as untyped string literals scattered across ~40 call
sites. GAPT *guessed* names and shipped a 100%-text-loss bug and a
$0-cost bug off two wrong guesses; Geny maintained two divergent
600-line mapping switches. This module is the single authoritative
enumeration of every event name the engine emits, plus a field-level
payload description per event (:data:`PAYLOADS`).

Contract rules
--------------
* **Values are the wire strings.** ``EventTypes.TEXT_DELTA == "text.delta"``
  — the enum is a *names registry*, not a rename. Existing consumers
  matching raw strings keep working forever.
* **Append-only.** New events may be added in minor releases; renaming
  or removing a member is a major-version change.
* **Payloads may gain fields** in minor releases; existing fields keep
  their meaning. :data:`PAYLOADS` documents the fields each event
  carries today (descriptions, not strict schemas — events are
  observability, not RPC).
* Completeness is enforced by a grep/AST-driven test
  (``tests/unit/test_event_catalog.py``): every string literal passed
  to ``state.add_event`` / ``Pipeline._emit`` in ``src/`` must appear
  here, so a new emit site cannot ship uncatalogued.

What is deliberately NOT here
-----------------------------
* :class:`~geny_executor.hooks.events.HookEvent` — the hook taxonomy is
  a separate, already-typed channel (blocking semantics, its own
  ``FIRED_EVENTS`` honesty test). Catalogue events are observational.
* Memory-plane STM journal entries (``MemoryProvider.append_event``)
  — provider-internal persistence, not the pipeline stream.

The ``llm_client.*`` family is included even though it travels through
the client's ``event_sink`` callback rather than ``state.add_event`` —
hosts that wire a sink receive exactly these names, and they were the
other half of the unpublished contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

#: Bumped when the catalogue gains members (append-only). Hosts can pin
#: a minimum version to assert the names they consume exist.
#: v2: + ``agent.delegations_capped`` (Stage 12 max_delegations wiring).
#: v3: + ``api.internal_loop_capped`` (2.3.0 Stage 6 internal agentic
#: loop hit its turn/budget cap and degraded to the pipeline path).
EVENT_CATALOG_VERSION = 3


class EventTypes(str, Enum):
    """Every event name the engine emits, value == wire string.

    ``str``-derived so members compare equal to the raw strings already
    flowing through :class:`~geny_executor.events.types.PipelineEvent`
    and ``state.events`` dicts::

        if event.type == EventTypes.TEXT_DELTA:
            ...
    """

    # ── Pipeline lifecycle (EventBus, emitted by Pipeline) ──
    PIPELINE_START = "pipeline.start"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_ERROR = "pipeline.error"
    STAGE_ENTER = "stage.enter"
    STAGE_EXIT = "stage.exit"
    STAGE_BYPASS = "stage.bypass"
    STAGE_ERROR = "stage.error"

    # ── Run-start announcements (deferred via _pending_runtime_events) ──
    CONFIG_OVERRIDE_APPLIED = "config.override_applied"
    RUNTIME_LLM_CLIENT_OVERRIDE = "runtime.llm_client_override"

    # ── Loop control ──
    LOOP_FORCE_COMPLETE = "loop.force_complete"
    # Stage 16 emits ``loop.{decision}`` — the four canonical verdicts.
    # A custom LoopController returning a fifth string would emit an
    # uncatalogued name; the completeness test allowlists that f-string
    # site, and custom controllers own their custom names.
    LOOP_CONTINUE = "loop.continue"
    LOOP_COMPLETE = "loop.complete"
    LOOP_ERROR = "loop.error"
    LOOP_ESCALATE = "loop.escalate"

    # ── Stage 1: Input ──
    INPUT_NORMALIZED = "input.normalized"

    # ── Stage 2: Context ──
    CONTEXT_BUILT = "context.built"
    CONTEXT_COMPACTED = "context.compacted"
    CONTEXT_COMPACTION_FAILED = "context.compaction_failed"
    CONTEXT_COMPACTION_RECORD_FAILED = "context.compaction_record_failed"
    MEMORY_COMPACTION_SUMMARIZED = "memory.compaction.summarized"
    MEMORY_COMPACTION_LLM_FAILED = "memory.compaction.llm_failed"

    # ── Stage 3: System ──
    SYSTEM_BUILT = "system.built"

    # ── Stage 4: Guard ──
    GUARD_CHECK = "guard.check"
    GUARD_WARN = "guard.warn"
    GUARD_COMPACTING = "guard.compacting"

    # ── Stage 5: Cache ──
    CACHE_APPLIED = "cache.applied"

    # ── Stage 6: API ──
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_RETRY = "api.retry"
    API_ERROR = "api.error"
    API_ROUTER_ERROR = "api.router.error"
    API_MODEL_ROUTED = "api.model_routed"
    API_TIMEOUT_UNSUPPORTED = "api.timeout_unsupported"
    # Streaming chunk forwarding (2.2.0, audit §3.2 / Tier 1-1: the
    # GAPT/Geny monkey-patch killer — pre-2.2.0 only text deltas were
    # forwarded; tool_use / thinking chunks died inside Stage 6).
    TEXT_DELTA = "text.delta"
    THINKING_DELTA = "thinking.delta"
    API_TOOL_USE = "api.tool_use"
    API_CLI_TOOL_CALL = "api.cli_tool_call"
    API_INPUT_JSON_DELTA = "api.input_json_delta"
    API_CONTENT_BLOCK_STOP = "api.content_block_stop"
    API_TOOL_RESULT = "api.tool_result"
    # 2.3.0: the Stage 6 internal agentic loop (tool_loop="internal")
    # stopped resolving tool calls because it hit its turn or budget
    # cap; the last response is returned with its tool_use blocks
    # intact so Stage 9/10 pick them up — graceful degradation to the
    # pipeline path, announced rather than silent.
    API_INTERNAL_LOOP_CAPPED = "api.internal_loop_capped"

    # ── Stage 7: Token ──
    TOKEN_TRACKED = "token.tracked"

    # ── Stage 8: Think ──
    THINK_PROCESSED = "think.processed"
    THINK_BUDGET_APPLIED = "think.budget_applied"

    # ── Stage 9: Parse ──
    PARSE_COMPLETE = "parse.complete"

    # ── Stage 10: Tool ──
    TOOL_EXECUTE_START = "tool.execute_start"
    TOOL_EXECUTE_COMPLETE = "tool.execute_complete"
    # Per-call timing pair emitted by the executor strategies around each
    # individual dispatch (Stage 10 batches AND Stage 6 internal-loop
    # dispatches — both run through the same executors). Catalogued in
    # 2.3.0: they had always been emitted through the on_event callback,
    # which the AST completeness test cannot see — the indirect-emission
    # blind spot the catalogue's own docstring warns about.
    TOOL_CALL_START = "tool.call_start"
    TOOL_CALL_COMPLETE = "tool.call_complete"

    # ── Stage 11: Tool review ──
    TOOL_REVIEW_FLAG = "tool_review.flag"
    TOOL_REVIEW_COMPLETED = "tool_review.completed"
    TOOL_REVIEW_REVIEWER_ERROR = "tool_review.reviewer_error"

    # ── Stage 12: Agent ──
    AGENT_ORCHESTRATE_START = "agent.orchestrate_start"
    AGENT_ORCHESTRATE_COMPLETE = "agent.orchestrate_complete"
    AGENT_DELEGATIONS_CAPPED = "agent.delegations_capped"

    # ── Stage 13: Task registry ──
    TASK_REGISTERED = "task.registered"
    TASK_DONE = "task.done"
    TASK_FAILED = "task.failed"
    TASK_TIMEOUT = "task.timeout"
    TASK_REGISTRY_SYNCED = "task_registry.synced"
    TASK_REGISTRY_INVALID_PAYLOAD = "task_registry.invalid_payload"
    TASK_REGISTRY_POLICY_ERROR = "task_registry.policy_error"

    # ── Stage 14: Evaluate ──
    EVALUATE_START = "evaluate.start"
    EVALUATE_COMPLETE = "evaluate.complete"

    # ── Stage 15: HITL (also fired from Stage 10's permission ASK path) ──
    HITL_REQUEST = "hitl.request"
    HITL_DECISION = "hitl.decision"
    HITL_NO_DECISION = "hitl.no_decision"
    HITL_TIMEOUT = "hitl.timeout"
    HITL_REQUESTER_ERROR = "hitl.requester_error"

    # ── Stage 17: Emit ──
    EMIT_START = "emit.start"
    EMIT_COMPLETE = "emit.complete"
    EMIT_TIMEOUT = "emit.timeout"
    EMIT_SKIPPED_BACKPRESSURE = "emit.skipped_backpressure"
    EMIT_SKIPPED_DEP_FAILED = "emit.skipped_dep_failed"
    EMIT_CYCLE_DETECTED = "emit.cycle_detected"
    EMIT_UNKNOWN_DEPENDENCY = "emit.unknown_dependency"

    # ── Stage 18: Memory (+ MemoryEvent spec values, MEMORY_SPEC.yaml) ──
    MEMORY_UPDATED = "memory.updated"
    MEMORY_PERSISTED = "memory.persisted"
    MEMORY_TURN_RECORDED = "memory.turn_recorded"
    MEMORY_EXECUTION_RECORDED = "memory.execution_recorded"
    MEMORY_INSIGHT = "memory.insight"
    MEMORY_PROMOTED = "memory.promoted"
    MEMORY_REINDEXED = "memory.reindexed"
    MEMORY_COST = "memory.cost"
    MEMORY_SNAPSHOT = "memory.snapshot"
    MEMORY_INSIGHT_RECORDED = "memory.insight_recorded"
    MEMORY_INSIGHT_INVALID = "memory.insight_invalid"
    MEMORY_REFLECTION_QUEUED = "memory.reflection_queued"
    MEMORY_STRUCTURED_REFLECTION_DONE = "memory.structured_reflection_done"
    MEMORY_PROVIDER_RECORDED = "memory.provider_recorded"
    MEMORY_RETRIEVE_BREAKDOWN = "memory.retrieve_breakdown"
    MEMORY_RETRIEVED_EMPTY = "memory.retrieved_empty"

    # ── Stage 19: Summarize ──
    SUMMARY_WRITTEN = "summary.written"
    SUMMARY_SKIPPED = "summary.skipped"
    SUMMARY_SESSION_CLOSED = "summary.session_closed"
    SUMMARY_SESSION_CLOSE_ERROR = "summary.session_close_error"
    SUMMARY_IMPORTANCE_ERROR = "summary.importance_error"
    SUMMARY_PROVIDER_RECORDED = "summary.provider_recorded"
    SUMMARY_PROVIDER_ERROR = "summary.provider_error"
    SUMMARY_SUMMARIZER_ERROR = "summary.summarizer_error"

    # ── Stage 20: Persist ──
    CHECKPOINT_WRITTEN = "checkpoint.written"
    CHECKPOINT_SKIPPED = "checkpoint.skipped"
    CHECKPOINT_PERSISTER_ERROR = "checkpoint.persister_error"

    # ── Stage 21: Yield ──
    YIELD_COMPLETE = "yield.complete"
    YIELD_SUMMARY = "yield.summary"

    # ── llm_client event_sink channel (boundary telemetry) ──
    LLM_CLIENT_FEATURE_UNSUPPORTED = "llm_client.feature_unsupported"
    LLM_CLIENT_PARAMETER_DROPPED = "llm_client.parameter_dropped"
    LLM_CLIENT_DRIFT_HEALED = "llm_client.drift_healed"
    LLM_CLIENT_UNKNOWN_WIRE_SHAPE = "llm_client.unknown_wire_shape"


#: Field-level payload documentation, one entry per :class:`EventTypes`
#: member: ``{field_name: description}``. Descriptive (what hosts can
#: rely on reading), not a validation schema — see module docstring for
#: the stability rules. ``…?`` marks fields that are present only in
#: some emissions of the event.
PAYLOADS: Dict[EventTypes, Dict[str, str]] = {
    EventTypes.PIPELINE_START: {
        "input": "str — the user turn, truncated to Pipeline.EVENT_DATA_TRUNCATE chars",
    },
    EventTypes.PIPELINE_COMPLETE: {
        "iterations": "int — loop iterations this turn",
        "result": "str? — full final text (run_stream only; never truncated)",
        "total_cost_usd": "float? — this turn's cost (run_stream only)",
    },
    EventTypes.PIPELINE_ERROR: {
        "error": "str — message (legacy field, always present)",
        "code": "str — stable ExecutorErrorCode value ('exec.*'), 'exec.unknown' fallback",
        "exception_type": "str — fully qualified exception class name",
        "total_cost_usd": "float? — this turn's cost (run_stream only)",
    },
    EventTypes.STAGE_ENTER: {},  # identity carried on the event envelope (stage/iteration)
    EventTypes.STAGE_EXIT: {},
    EventTypes.STAGE_BYPASS: {},
    EventTypes.STAGE_ERROR: {
        "error": "str — message",
        "code": "str — stable ExecutorErrorCode value",
        "exception_type": "str — fully qualified exception class name",
    },
    EventTypes.CONFIG_OVERRIDE_APPLIED: {
        "field": "str — ModelOverrides field name",
        "value": "Any — the override value applied for this run",
        "source": "str — 'per_run'",
    },
    EventTypes.RUNTIME_LLM_CLIENT_OVERRIDE: {
        "manifest_provider": "str — Stage 6 provider the manifest declared",
        "client_provider": "str — provider of the attached client that overrides it",
    },
    EventTypes.LOOP_FORCE_COMPLETE: {
        "reason": "str — 'max_iterations' | 'cost_budget'",
        "iteration": "int? — iteration count (max_iterations only)",
        "total_cost_usd": "float? — turn cost (cost_budget only)",
        "budget_usd": "float? — the configured budget (cost_budget only)",
    },
    EventTypes.LOOP_CONTINUE: {
        "iteration": "int",
        "signal": "str|None — completion signal if any",
        "pending_tools": "int — queued tool calls",
        "has_tool_results": "bool",
        "upstream_decision": "str — loop_decision before the controller ran",
    },
    EventTypes.LOOP_COMPLETE: {
        "iteration": "int",
        "signal": "str|None",
        "pending_tools": "int",
        "has_tool_results": "bool",
        "upstream_decision": "str",
    },
    EventTypes.LOOP_ERROR: {
        "iteration": "int",
        "signal": "str|None",
        "pending_tools": "int",
        "has_tool_results": "bool",
        "upstream_decision": "str",
    },
    EventTypes.LOOP_ESCALATE: {
        "iteration": "int",
        "signal": "str|None",
        "pending_tools": "int",
        "has_tool_results": "bool",
        "upstream_decision": "str",
    },
    EventTypes.INPUT_NORMALIZED: {
        "text_length": "int — normalized text length",
    },
    EventTypes.CONTEXT_BUILT: {
        "message_count": "int",
        "memory_refs": "int? — count of attached memory refs (stage form)",
        "estimated_tokens": "int?",
        "chunks": "int? — RetrievalResult.to_event form (provider-driven path)",
    },
    EventTypes.CONTEXT_COMPACTED: {
        "strategy": "str — compactor name/class",
        "trigger": "str? — 'proactive' (Stage 2) | 'guard' (Stage 4)",
        "messages_before": "int?",
        "messages_after": "int?",
        "saved_tokens_estimate": "int?",
    },
    EventTypes.CONTEXT_COMPACTION_FAILED: {
        "compactor": "str",
        "trigger": "str — 'proactive' | 'guard'",
        "error": "str",
    },
    EventTypes.CONTEXT_COMPACTION_RECORD_FAILED: {
        "compactor": "str",
        "error": "str",
    },
    EventTypes.MEMORY_COMPACTION_SUMMARIZED: {
        "model": "str",
        "provider": "str",
        "old_count": "int — messages before compaction",
        "summary_chars": "int",
    },
    EventTypes.MEMORY_COMPACTION_LLM_FAILED: {
        "error": "str",
        "compactor": "str",
    },
    EventTypes.SYSTEM_BUILT: {
        "prompt_type": "str — 'content_blocks' | 'string'",
        "prompt_length": "int — characters",
    },
    EventTypes.GUARD_CHECK: {
        "passed": "bool",
        "guard_name": "str — guard (or comma-joined chain) name",
        "message": "str",
        "violations": "list? — [{guard_name, message, action}] (chain form)",
    },
    EventTypes.GUARD_WARN: {
        "message": "str",
    },
    EventTypes.GUARD_COMPACTING: {
        "guard_name": "str — guard that signalled compaction (token_budget)",
        "reason": "str — the guard message",
    },
    EventTypes.CACHE_APPLIED: {
        "strategy": "str — cache strategy class name",
        "system_is_blocks": "bool",
        "cache_key": "str",
    },
    EventTypes.API_REQUEST: {
        "model": "str",
        "provider": "str",
        "message_count": "int",
        "has_tools": "bool",
        "has_thinking": "bool",
        "stream": "bool",
    },
    EventTypes.API_RESPONSE: {
        "stop_reason": "str",
        "text_length": "int",
        "tool_calls": "int",
        "input_tokens": "int",
        "output_tokens": "int",
    },
    EventTypes.API_RETRY: {
        "attempt": "int — 1-based attempt that just failed",
        "category": "str — ErrorCategory value",
        "code": "str? — ExecutorErrorCode value (non-stream path)",
        "delay": "float — backoff seconds before next attempt",
        "stream": "bool? — True on the streaming retry path",
    },
    EventTypes.API_ERROR: {
        "code": "str — stable ExecutorErrorCode value (e.g. 'exec.cli.auth_failed')",
        "category": "str — ErrorCategory value the retry machinery classified",
        "provider": "str — client provider name",
        "cli_version": "str? — CLI version when a CLI-backed client knows it",
        "message": "str — human-readable error text",
    },
    EventTypes.API_ROUTER_ERROR: {
        "router": "str",
        "error": "str",
    },
    EventTypes.API_MODEL_ROUTED: {
        "router": "str",
        "from": "str — baseline model",
        "to": "str — routed model",
    },
    EventTypes.API_TIMEOUT_UNSUPPORTED: {
        "provider": "str",
        "timeout_ms": "int — the configured-but-undeliverable timeout",
    },
    EventTypes.TEXT_DELTA: {
        "text": "str — one streamed text chunk",
    },
    EventTypes.THINKING_DELTA: {
        "text": "str — one streamed extended-thinking chunk",
    },
    EventTypes.API_TOOL_USE: {
        "id": "str|None — tool_use block id",
        "name": "str|None — tool name",
        "input": "dict — tool input (may be partial until input_json_delta completes)",
        "source": (
            "str — 'cli' (executed inside a CLI backend) | 'api' (Stage 10 "
            "will dispatch) | 'internal' (the Stage 6 internal loop is about "
            "to dispatch it)"
        ),
    },
    EventTypes.API_CLI_TOOL_CALL: {
        "id": "str|None",
        "name": "str|None",
        "input": "dict",
        "source": "str — always 'cli'; companion to api.tool_use for narrow subscriptions",
    },
    EventTypes.API_INPUT_JSON_DELTA: {
        "delta": "str — partial JSON fragment of the pending tool input",
    },
    EventTypes.API_CONTENT_BLOCK_STOP: {},
    EventTypes.API_TOOL_RESULT: {
        "tool_use_id": "str — id of the tool_use this result answers",
        "content": "Any — tool result content as the backend reported it",
        "is_error": "bool",
        "source": "str — 'cli' | 'api' | 'internal' (Stage 6 internal loop dispatched it)",
    },
    EventTypes.API_INTERNAL_LOOP_CAPPED: {
        "turns": "int — inner tool turns the loop completed before stopping",
        "reason": "str — 'max_inner_turns' | 'cost_budget'",
    },
    EventTypes.TOKEN_TRACKED: {
        "input_tokens": "int",
        "output_tokens": "int",
        "cache_write": "int",
        "cache_read": "int",
        "cost_usd": "float|None — this call's cost",
        "total_cost_usd": "float — turn accumulator after this call",
    },
    EventTypes.THINK_PROCESSED: {
        "thinking_block_count": "int",
        "total_thinking_tokens": "int",
    },
    EventTypes.THINK_BUDGET_APPLIED: {
        "planner": "str",
        "from": "int — previous thinking budget",
        "to": "int — newly applied budget",
    },
    EventTypes.PARSE_COMPLETE: {
        "text_length": "int",
        "tool_calls": "int",
        "signal": "str|None",
        "stop_reason": "str",
    },
    EventTypes.TOOL_EXECUTE_START: {
        "count": "int",
        "tools": "list[str] — tool names about to dispatch",
    },
    EventTypes.TOOL_EXECUTE_COMPLETE: {
        "count": "int",
        "errors": "int — results flagged is_error",
    },
    EventTypes.TOOL_CALL_START: {
        "tool_use_id": "str",
        "name": "str",
        "input": "dict",
    },
    EventTypes.TOOL_CALL_COMPLETE: {
        "tool_use_id": "str",
        "name": "str",
        "is_error": "bool",
        "duration_ms": "int",
    },
    EventTypes.TOOL_REVIEW_FLAG: {
        "reviewer": "str — ReviewFlag.to_dict()",
        "severity": "str",
        "message": "str",
    },
    EventTypes.TOOL_REVIEW_COMPLETED: {
        "reviewers": "list[str]",
        "flags": "int",
        "tool_calls": "int",
        "tool_results": "int",
    },
    EventTypes.TOOL_REVIEW_REVIEWER_ERROR: {
        "reviewer": "str",
        "error": "str",
    },
    EventTypes.AGENT_ORCHESTRATE_START: {
        "orchestrator": "str",
        "delegate_count": "int",
    },
    EventTypes.AGENT_ORCHESTRATE_COMPLETE: {
        "delegated": "bool",
        "sub_result_count": "int",
    },
    EventTypes.AGENT_DELEGATIONS_CAPPED: {
        "requested": "int — delegate requests queued this turn",
        "cap": "int — the max_delegations limit that truncated them",
    },
    EventTypes.TASK_REGISTERED: {
        "task_id": "str",
        "kind": "str",
        "status": "str — TaskStatus value",
    },
    EventTypes.TASK_DONE: {
        "task_id": "str",
        "kind": "str",
    },
    EventTypes.TASK_FAILED: {
        "task_id": "str",
        "kind": "str",
        "error": "str",
    },
    EventTypes.TASK_TIMEOUT: {
        "task_id": "str",
        "kind": "str",
        "timeout_seconds": "float",
    },
    EventTypes.TASK_REGISTRY_SYNCED: {
        "new": "int",
        "by_status": "dict[str, int]",
        "total": "int",
    },
    EventTypes.TASK_REGISTRY_INVALID_PAYLOAD: {
        "payload_repr": "str — repr of the rejected payload, truncated",
    },
    EventTypes.TASK_REGISTRY_POLICY_ERROR: {
        "policy": "str",
        "error": "str",
    },
    EventTypes.EVALUATE_START: {
        "strategy": "str",
    },
    EventTypes.EVALUATE_COMPLETE: {
        "passed": "bool",
        "score": "float|None",
        "decision": "str",
        "loop_decision": "str",
        "feedback": "str — truncated to 200 chars",
    },
    EventTypes.HITL_REQUEST: {
        "token": "str — HITLRequest.to_dict(); resolve via Pipeline.resume(token, ...)",
        "reason": "str",
        "severity": "str",
        "payload": "dict",
    },
    EventTypes.HITL_DECISION: {
        "token": "str",
        "decision": "str — approve | reject | cancel",
        "via": "str? — resolution channel (Stage 10 permission path)",
    },
    EventTypes.HITL_NO_DECISION: {
        "token": "str",
        "verdict": "str — timeout-policy verdict applied",
    },
    EventTypes.HITL_TIMEOUT: {
        "token": "str",
        "timeout_seconds": "float",
        "verdict": "str",
    },
    EventTypes.HITL_REQUESTER_ERROR: {
        "requester": "str",
        "error": "str",
    },
    EventTypes.EMIT_START: {
        "emitter_count": "int",
        "channels": "list[str]",
    },
    EventTypes.EMIT_COMPLETE: {
        "channels_emitted": "list[str]",
        "all_emitted": "bool",
    },
    EventTypes.EMIT_TIMEOUT: {
        "emitter": "str",
        "timeout_seconds": "float|None",
        "consecutive_timeouts": "int",
    },
    EventTypes.EMIT_SKIPPED_BACKPRESSURE: {
        "emitter": "str",
        "consecutive_timeouts": "int",
    },
    EventTypes.EMIT_SKIPPED_DEP_FAILED: {
        "emitter": "str",
        "deps": "list[str] — failed dependencies",
    },
    EventTypes.EMIT_CYCLE_DETECTED: {
        "ordered_count": "int",
        "total": "int",
        "emitters": "list[str]",
    },
    EventTypes.EMIT_UNKNOWN_DEPENDENCY: {
        "emitter": "str",
        "dependency": "str",
    },
    EventTypes.MEMORY_UPDATED: {
        "strategy": "str",
    },
    EventTypes.MEMORY_PERSISTED: {
        "session_id": "str",
        "message_count": "int",
        "persistence": "str",
    },
    EventTypes.MEMORY_TURN_RECORDED: {
        "role": "str",
        "bytes": "int",
    },
    EventTypes.MEMORY_EXECUTION_RECORDED: {
        "receipt": "ExecutionReceipt.to_event() fields (see memory/provider.py)",
    },
    EventTypes.MEMORY_INSIGHT: {
        "insight": "Insight.to_event() fields (see memory/provider.py)",
    },
    EventTypes.MEMORY_PROMOTED: {
        "ref": "dict — promoted MemoryRef",
        "from_scope": "str",
        "to_scope": "str",
    },
    EventTypes.MEMORY_REINDEXED: {
        "spec": "MemoryEvent spec slot — reserved; no engine emitter today",
    },
    EventTypes.MEMORY_COST: {
        "spec": "MemoryEvent spec slot — reserved; no engine emitter today",
    },
    EventTypes.MEMORY_SNAPSHOT: {
        "spec": "MemoryEvent spec slot — reserved; no engine emitter today",
    },
    EventTypes.MEMORY_INSIGHT_RECORDED: {
        "insight": "Insight.to_event() fields",
    },
    EventTypes.MEMORY_INSIGHT_INVALID: {
        "error": "str",
        "iteration": "int",
    },
    EventTypes.MEMORY_REFLECTION_QUEUED: {
        "message_count": "int",
        "iteration": "int",
    },
    EventTypes.MEMORY_STRUCTURED_REFLECTION_DONE: {
        "recorded": "int",
        "total": "int",
        "iteration": "int",
    },
    EventTypes.MEMORY_PROVIDER_RECORDED: {
        "count": "int — turns recorded into the provider",
        "total_messages": "int",
    },
    EventTypes.MEMORY_RETRIEVE_BREAKDOWN: {
        "query_preview": "str — truncated to 120 chars",
        "layers": "dict — per-layer hit counts",
        "total_chars": "int",
        "chunk_count": "int",
        "slim_mode": "bool",
    },
    EventTypes.MEMORY_RETRIEVED_EMPTY: {
        "query_preview": "str",
        "reason": "str",
        "session_id": "str",
    },
    EventTypes.SUMMARY_WRITTEN: {
        "record": "SummaryRecord.to_dict() fields (turn_id, importance, ...)",
    },
    EventTypes.SUMMARY_SKIPPED: {
        "summarizer": "str",
        "reason": "str",
    },
    EventTypes.SUMMARY_SESSION_CLOSED: {
        "chars": "int",
        "turns": "int",
        "decision": "str",
    },
    EventTypes.SUMMARY_SESSION_CLOSE_ERROR: {
        "error": "str",
    },
    EventTypes.SUMMARY_IMPORTANCE_ERROR: {
        "importance": "str",
        "error": "str",
    },
    EventTypes.SUMMARY_PROVIDER_RECORDED: {
        "turn_id": "str",
        "importance": "str",
    },
    EventTypes.SUMMARY_PROVIDER_ERROR: {
        "error": "str",
    },
    EventTypes.SUMMARY_SUMMARIZER_ERROR: {
        "summarizer": "str",
        "error": "str",
    },
    EventTypes.CHECKPOINT_WRITTEN: {
        "checkpoint_id": "str",
        "session_id": "str",
        "iteration": "int",
        "persister": "str",
    },
    EventTypes.CHECKPOINT_SKIPPED: {
        "frequency": "str",
        "iteration": "int",
    },
    EventTypes.CHECKPOINT_PERSISTER_ERROR: {
        "persister": "str",
        "error": "str",
    },
    EventTypes.YIELD_COMPLETE: {
        "text_length": "int",
        "iterations": "int",
        "total_cost_usd": "float",
    },
    EventTypes.YIELD_SUMMARY: {
        "text_length": "int",
        "iterations": "int",
        "total_cost_usd": "float",
    },
    EventTypes.LLM_CLIENT_FEATURE_UNSUPPORTED: {
        "provider": "str",
        "field": "str — request field the client cannot honour",
    },
    EventTypes.LLM_CLIENT_PARAMETER_DROPPED: {
        "provider": "str",
        "field": "str",
        "value": "Any — the discarded value",
    },
    EventTypes.LLM_CLIENT_DRIFT_HEALED: {
        "provider": "str",
        "model": "str",
        "field": "str — request field rebuilt after a vendor 400",
        "message": "str — the vendor error that named the problem",
    },
    EventTypes.LLM_CLIENT_UNKNOWN_WIRE_SHAPE: {
        "provider": "str",
        "unknown_type": "str|None — first unrecognised line type",
        "count": "int — unknown + malformed lines",
        "unknown_line_count": "int",
        "malformed_line_count": "int",
        "cli_version": "str",
    },
}


def known_event_types() -> List[str]:
    """All catalogued wire strings, sorted — UI/validation accessor."""
    return sorted(e.value for e in EventTypes)


__all__ = [
    "EVENT_CATALOG_VERSION",
    "EventTypes",
    "PAYLOADS",
    "known_event_types",
]


def _payload_completeness_check() -> None:  # pragma: no cover - import-time guard
    """Import-time invariant: every member documents its payload.

    Cheap (runs once per process) and turns "forgot the PAYLOADS entry"
    into an immediate import error rather than a doc-rot discovery.
    """
    missing = [e.value for e in EventTypes if e not in PAYLOADS]
    if missing:
        raise RuntimeError(
            f"events/catalog.py: PAYLOADS is missing entries for {missing} — "
            "every EventTypes member must document its payload fields."
        )


_payload_completeness_check()


# Typing helper referenced in docstrings; kept here so the module is
# self-contained for hosts that introspect it.
PayloadDoc = Dict[str, Any]
