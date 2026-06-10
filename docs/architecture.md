# Architecture

> Status: current for geny-executor 2.2.0.

## Design principles

geny-executor is a **harness** — a deliberately explicit pipeline that exposes every step of agent execution rather than hiding it behind framework magic. The two architectural commitments that drop out of this:

1. **Configuration is artifact.** A pipeline is fully described by an `EnvironmentManifest` (JSON). The manifest names every stage, picks one strategy per slot, and pins config values. Loading the manifest reconstructs the pipeline deterministically.
2. **Dual abstraction.** Two orthogonal extension points: swap an entire stage (Level 1) or swap a strategy *inside* a stage (Level 2). Both happen by editing the manifest — no code changes for the common reconfigurations.

These mean:
- Every behaviour change has a corresponding diffable artifact change.
- Tests can pin a manifest and assert end-to-end behaviour without mocking framework internals.
- Hosts (Geny, CI runners, etc.) can ship many environments off one binary.

## The 21-stage pipeline

```
Phase A — Setup (once per turn)
  1: Input  →  2: Context  →  3: System  →  4: Guard  →  5: Cache

Phase B — Generate + Dispatch (loop)
  6: API  →  7: Token  →  8: Think  →  9: Parse
  → 10: Tool  →  11: ToolReview  →  12: Agent  →  13: TaskRegistry
  → 14: Evaluate  →  15: HITL  →  16: Loop

Phase C — Surface (once)
  17: Emit  →  18: Memory  →  19: Summarize  →  20: Persist  →  21: Yield
```

A turn enters at Stage 1, traverses Phase A once, loops through Phase B until Stage 16 (Loop) returns "done", and surfaces through Phase C. Every stage decides whether to run via `should_bypass(state)` — pass-through is the default for many stages, which keeps the minimal preset cheap.

### Stage reference

| # | Stage | Purpose | Example strategies |
|---|---|---|---|
| 1 | **Input** | Validate & normalise user input | `default`, `strict`, `schema`, `multimodal` |
| 2 | **Context** | Load conversation history + memory | `simple_load`, `progressive_disclosure`, `vector_search` |
| 3 | **System** | Build the system prompt | `static`, `composable`, `adaptive`, `dynamic_persona` |
| 4 | **Guard** | Safety + budget enforcement | `token_budget`, `cost`, `iteration`, `permission` (chainable) |
| 5 | **Cache** | Anthropic prompt-cache control | `no_cache`, `system`, `aggressive`, `adaptive` |
| 6 | **API** | Call the LLM provider | `anthropic`, `openai`, `google`, `vllm`, `claude_code_cli` |
| 7 | **Token** | Track usage + compute cost | `default`, `detailed` + per-provider pricing |
| 8 | **Think** | Process extended-thinking blocks | `passthrough`, `extract_and_store`, `budget` |
| 9 | **Parse** | Parse response, detect completion signals | `default`, `structured_output`, `signal_detector` |
| 10 | **Tool** | Dispatch `tool_use` blocks | `sequential`, `parallel`, `partition`, `streaming` |
| 11 | **ToolReview** | Inspect tool results before re-prompt | `passthrough`, `flagging`, `escalate_to_reviewer` |
| 12 | **Agent** | Sub-agent orchestration | `single_agent`, `delegate`, `subagent_type_orchestrator` |
| 13 | **TaskRegistry** | Register / track long-running tasks | `passthrough`, `local`, `external_queue` |
| 14 | **Evaluate** | Judge quality + completion | `signal_based`, `criteria_based`, `agent_eval`, `adaptive` |
| 15 | **HITL** | Human-in-the-loop pause / approval | `passthrough`, `gated`, `timeout_based` |
| 16 | **Loop** | Continue or finish? | `standard`, `single_turn`, `budget_aware` |
| 17 | **Emit** | Surface output | `text`, `callback`, `streaming`, `vtuber`, `tts` |
| 18 | **Memory** | Persist conversation memory | `append_only`, `reflective`, `vault`, file / SQLite backends |
| 19 | **Summarize** | Roll up long histories | `passthrough`, `truncate`, `llm_summary` |
| 20 | **Persist** | Save session snapshot | `passthrough`, `file`, `sqlite` |
| 21 | **Yield** | Format the final result | `default`, `structured`, `streaming` |

The exact strategy class list per stage lives next to each stage's `artifact/` folder. Browse `src/geny_executor/stages/<sNN_name>/artifact/`.

## Dual abstraction

```
┌─ Level 1: Stage Abstraction ─────────────────────────┐
│   Swap an entire stage module in/out of the pipeline. │
│                                                       │
│  ┌─ Level 2: Strategy Abstraction ─────────────────┐  │
│  │   Swap internal logic within a stage.            │  │
│  │                                                  │  │
│  │   ContextStage can use:                          │  │
│  │     → SimpleLoad     (default)                   │  │
│  │     → ProgressiveDisclosure                      │  │
│  │     → VectorSearch                               │  │
│  │     → YourCustomStrategy                         │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

- **Level 1 (Stage)**: drop in a custom `APIStage` for a proprietary provider, replace `MemoryStage` with a Redis-backed one, etc.
- **Level 2 (Strategy)**: keep the standard `ContextStage` but switch from `SimpleLoad` to `VectorSearch` by editing the manifest's `stages[2].strategies.loader`.

Strategies are wired through `StrategySlot` (`core/slot.py`) which a stage owns one of. `SlotChain` (`core/slot.py`) handles ordered chains like Stage 4's guard list.

## State + events

`PipelineState` (`core/state.py`) is the per-turn working set: messages, pending tool calls, memory blocks, usage, completion signal, custom shared dict. Stages mutate state in place; the pipeline serialises mutations across iterations.

Every stage transition emits an event onto the pipeline's event bus: lifecycle (`pipeline.*`, `stage.*`), per-stage domain events (`api.*`, `tool.*`, `hitl.*`, `memory.*`, …), and streaming chunks (`text.delta`, `thinking.delta`, …). Since 2.2.0 the complete taxonomy is published as the `EventTypes` enum (`geny_executor.events.catalog`) with field-level payload docs — see [events.md](events.md) for the generated catalogue.

Subscribe with `pipeline.on(event_type, handler)`, stream a single run with `pipeline.run_stream(...)`, or attach a multi-subscriber tap with `pipeline.events(replay_from=...)` (2.2.0). Error events carry a stable `code` field (since 2.1.0) — see [error_codes.md](error_codes.md).

## Mutation + snapshot

Pipelines are **live-mutable** between stages. `core/mutation.py` exposes `PipelineMutator` which can:
- swap a strategy in a slot
- update a stage's config dict
- enable/disable a stage
- replace the entire stage chain

`MutationLocked` is raised if the target stage is currently executing; otherwise mutations apply on the next iteration boundary. `PipelineSnapshot` (`core/snapshot.py`) freezes the current pipeline shape into a manifest-equivalent dict for diffing.

## Manifest is single source of truth

Provider selection is pinned at `stages[6].config["provider"]`. Strict-load rejects manifests that use the legacy `strategies["provider"]` slot. The same single-source rule applies to model / max_tokens / max_iterations / cost budget — each lives in exactly one manifest field.

See [manifest.md](manifest.md) for the schema and [providers.md](providers.md) for the provider catalog.

## Configuration precedence

Five channels can influence what a run executes with. Highest wins; every channel has exactly one lifetime, so "why did this run use that model?" always has a one-line answer.

| Precedence | Channel | Lifetime | What it may set |
|---|---|---|---|
| 1 (highest) | **Per-run `ModelOverrides`** — `run(..., overrides=...)` / `run_stream(..., overrides=...)` | **One run.** Applied to state after the config stomp; the next run's stomp reverts it by construction. Each applied field emits `config.override_applied`. | model, max_tokens, temperature, top_p, thinking_enabled, thinking_budget_tokens |
| 2 | **`PipelineMutator` mutations / `refresh_runtime(**kwargs)`** — between-turn live mutation | **Until cleared / re-mutated.** Refused mid-run (`MutationLocked` / `RuntimeError`); a refreshed `llm_client` bumps the client generation so reused states re-resolve. | strategy swaps + strategy/stage/model/pipeline config (mutator); every `attach_runtime` kwarg (refresh) |
| 3 | **`attach_runtime` runtime objects** — construction-time wiring, before the first run | **Construction.** One-shot by contract (gate raises after the first run — use `refresh_runtime` afterwards). `llm_client` is guarded: a provider mismatch against the manifest raises `ConfigError` unless `override_manifest=True` is acknowledged (announced via `runtime.llm_client_override` at the next run start). | memory/system/tool strategies, tool_context, llm_client, session_runtime, hook_runner, mcp_manager, permission rules/mode, subagent_registry |
| 4 | **Manifest** (`EnvironmentManifest`) | **Declarative.** The single source of truth on disk; each setting has exactly ONE home (model block at the top level, provider at `stages[6].config["provider"]`). `validate_manifest` flags dual-home declarations. | everything reconstructible: stages, strategies + configs, chains, tools, model block, pipeline block, subagents, memory |
| 5 (lowest) | **`PipelineConfig` defaults** | **Default.** Dataclass defaults — what you get for anything no higher channel set. | model `claude-sonnet-4-6`, max_tokens 8192, max_iterations 50, stream on, … |

Reading order at run start: `PipelineConfig.apply_to_state` stomps the state (4/5, as mutated by 2), then per-run overrides land on top (1). Runtime objects (3) are not state values — they are the live collaborators (clients, managers, strategies) the stages call into; the manifest names *which* to build, attach/refresh supply *the instances*.

## Tool execution modes

Stage 6's `tool_loop` strategy slot (2.3.0) decides WHERE the agentic loop runs — manifest-selectable per environment:

| Strategy | Shape | Choose it when |
|---|---|---|
| `"pipeline"` (default) | One client call per pipeline iteration; tool_use blocks flow to Stage 9 → Stage 10 dispatch → Stage 16 loops the whole pipeline. | You want full per-round-trip stage control: guards re-checked, tokens tracked, review/evaluation run per tool exchange. The pre-2.3.0 behaviour, byte-identical. |
| `"internal"` | Stage 6 resolves tool calls inside the stage (call → dispatch → call …) and returns only the final response — the execution shape the `claude_code_cli` backend has always had (its subprocess runs the loop; the terminal response carries no tool blocks, so Stage 9/10 naturally no-op). | You want CLI-parity efficiency: no Stage 2-5/7/14 re-run per tool round-trip. `strategy_configs: {"tool_loop": {"max_inner_turns": N, "parallel_tools": bool}}`. |

The internal loop dispatches through `state.tool_dispatcher` — a thin handle over the registered Tool stage's own machinery (`ToolDispatcher`, `stages/s10_tool/dispatcher.py`): same `ToolRegistry` instance, same permission ladder (matrix → posture → ASK→HITL → hooks), same large-result persistence, same `tool.call_start`/`tool.call_complete` timing events. There is exactly one permission-decision implementation in the engine. Every inner call emits its own `api.request`/`api.response` pair plus `api.tool_use {source:"internal"}` / `api.tool_result` events; the returned response's usage is the sum over all inner calls, so Stage 7 prices the whole turn. Caps (`max_inner_turns`, the per-turn cost budget) emit `api.internal_loop_capped` and hand any leftover tool calls back to the pipeline path — graceful degradation, never dropped work. Subprocess backends and tool-less clients degrade to `"pipeline"` behaviour with a one-time warning.

## 2.2.0 surfaces

The 2.2.0 cycle (audit 2026-06-09) promoted the patterns hosts had been hand-rolling into owned library APIs:

- **`EventTypes` catalogue** (`geny_executor.events.catalog`) — the published, versioned enumeration of every event name the engine emits, value == wire string, with per-event payload field docs (`PAYLOADS`). Append-only within a major version; a completeness test keeps emit sites and catalogue in lockstep. Generated reference: [events.md](events.md).

- **`pipeline.events(replay_from=...)`** — multi-subscriber async-iterator tap over the unified event stream (bus events + bridged state events, stamped with `seq` / `run_id` / `session_id`). A bounded ring journal supports cursor replay so a UI attaching mid-session catches up without polling.

- **Full chunk forwarding** — Stage 6 forwards every canonical streaming chunk as events (`text.delta`, `thinking.delta`, `api.tool_use`, `api.input_json_delta`, `api.content_block_stop`, `api.tool_result`, CLI-executed tool calls as `api.cli_tool_call`), not just text deltas. Structured `api.error` events carry the stable `code`.

- **`build_manifest(preset, *, provider, model, ...)`** (`geny_executor.core.manifest_factory`) — library-owned preset→manifest factory. Returns a ready-to-build 21-stage `EnvironmentManifest` that passes `from_manifest(strict=True)` and round-trips `to_dict`/`from_dict` unchanged. Unknown preset/provider fails at factory time.

- **`validate_manifest(manifest) -> list[ManifestIssue]`** (`geny_executor.core.environment`) — write-time contract checking: unknown stages/strategies, configs no strategy consumes, misplaced keys, missing required stages. `error` findings block `strict` builds; `warning` findings log. Pure and offline — env editors call it on save.

- **`Pipeline.aclose()`** — required host teardown: cancels pending HITL futures, closes live `events()` taps, disconnects MCP servers (reaps stdio children), shuts down tool providers. Idempotent and best-effort.

- **`Pipeline.refresh_runtime(**kwargs)`** — between-turn runtime update with `attach_runtime` semantics, legal at any turn boundary (raises if a run is in progress). Replaces host-side private-setter queues for credential rotation / tool-context swaps.

- **`ModelOverrides`** (`geny_executor.core.config`) — frozen per-run override value passed as `run(...)` / `run_stream(..., overrides=...)`. Non-`None` fields win over manifest/config for exactly one run and emit `config.override_applied` events; the next run reverts automatically.

- **`PipelineState.begin_turn()`** — the per-turn reset contract for long-lived states (loop counters, decisions, per-turn outputs, in-flight tool work) — called automatically when a reused state re-enters `run()` / `run_stream()`; sticky session fields are untouched.

Hosts migrating off compensation layers: see [migration-2.2.md](migration-2.2.md).
