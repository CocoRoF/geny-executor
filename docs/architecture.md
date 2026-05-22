# Architecture

> Status: current for geny-executor 2.1.0.

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

Every stage transition emits an event onto the pipeline's event bus:

- `pipeline.start` / `pipeline.complete` / `pipeline.error`
- `stage.enter` / `stage.exit` / `stage.error` / `stage.bypass`
- `api.retry` (Stage 6 on recoverable error)
- `tool.execute_start` / `tool.call_start` / `tool.call_complete` / `tool.execute_complete`
- `tool_review.*`, `hitl.*`, `mcp.*` — domain-specific events

Subscribe with `pipeline.on(event_type, handler)` or stream with `pipeline.run_stream(...)`. Error events carry a stable `code` field (since 2.1.0) — see [error_codes.md](error_codes.md).

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
