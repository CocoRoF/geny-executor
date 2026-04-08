# geny-executor: Implementation Progress

> **Last Updated**: 2026-04-08
> **Status**: Core Library Complete (v0.1.0)
> **Tests**: 81/81 passing

---

## Architecture Summary

16-stage harness-engineered pipeline with **Dual Abstraction** (Stage + Strategy).

```
Phase A (once):  [1: Input]
Phase B (loop):  [2: Context] → [3: System] → [4: Guard] → [5: Cache]
                 → [6: API] → [7: Token] → [8: Think] → [9: Parse]
                 → [10: Tool] → [11: Agent] → [12: Evaluate] → [13: Loop]
Phase C (once):  [14: Emit] → [15: Memory] → [16: Yield]
```

---

## Phase Completion

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| **Phase 1** | Core Engine + Minimal Pipeline | **COMPLETE** | 12/12 |
| **Phase 2** | Agent Loop + Tool System | **COMPLETE** | 9/9 |
| **Phase 3** | Context + Memory + Cache + Session | **COMPLETE** | 11/11 |
| **Phase 4** | Think + Agent + Evaluate | **COMPLETE** | 21/21 |
| **Phase 5** | Emit + Presets + MCP | **COMPLETE** | 17/17 |
| **Phase 6** | Integration Tests | **COMPLETE** | 11/11 |

**Total: 81 tests, 83 source files, ~7,000 lines of code**

---

## All 16 Stages Implemented

| # | Stage | Category | Level 2 Strategies |
|---|-------|----------|-------------------|
| 1 | **Input** | Ingress | InputValidator (Default/Passthrough/Strict), InputNormalizer (Default/Multimodal) |
| 2 | **Context** | Preparation | ContextStrategy (SimpleLoad/Hybrid/ProgressiveDisclosure), HistoryCompactor (Truncate/Summary/SlidingWindow), MemoryRetriever (Null/Static) |
| 3 | **System** | Preparation | PromptBuilder (Static/Composable), PromptBlocks (Persona/Rules/DateTime/MemoryContext/ToolInstructions/Custom) |
| 4 | **Guard** | Preparation | GuardChain + Guards (TokenBudget/CostBudget/Iteration/Permission) |
| 5 | **Cache** | Preparation | CacheStrategy (NoCache/SystemCache/AggressiveCache) |
| 6 | **API** | Execution | APIProvider (Anthropic/Mock/Recording), RetryStrategy (ExponentialBackoff/NoRetry/RateLimitAware) |
| 7 | **Token** | Execution | TokenTracker (Default/Detailed), CostCalculator (AnthropicPricing/Custom) |
| 8 | **Think** | Execution | ThinkingProcessor (Passthrough/ExtractAndStore/Filter) |
| 9 | **Parse** | Execution | ResponseParser (Default/StructuredOutput), CompletionSignalDetector (Regex/Structured/Hybrid) |
| 10 | **Tool** | Execution | ToolExecutor (Sequential/Parallel), ToolRouter (Registry) |
| 11 | **Agent** | Execution | AgentOrchestrator (SingleAgent/Delegate/Evaluator), SubPipelineFactory |
| 12 | **Evaluate** | Decision | EvaluationStrategy (SignalBased/CriteriaBased/Agent), QualityScorer (No/Weighted) |
| 13 | **Loop** | Decision | LoopController (Standard/SingleTurn/BudgetAware) |
| 14 | **Emit** | Egress | Emitter (Text/Callback/VTuber/TTS), EmitterChain |
| 15 | **Memory** | Egress | MemoryUpdateStrategy (AppendOnly/NoMemory/Reflective), ConversationPersistence (InMemory/File) |
| 16 | **Yield** | Egress | ResultFormatter (Default/Structured/Streaming) |

---

## Additional Components

| Component | Description | Status |
|-----------|-------------|--------|
| **PipelineBuilder** | Fluent declarative API for pipeline construction | COMPLETE |
| **PipelinePresets** | Pre-configured pipelines (minimal/chat/agent/evaluator/geny_vtuber) | COMPLETE |
| **Session** | Pipeline + State execution unit with freshness tracking | COMPLETE |
| **SessionManager** | Session CRUD and lifecycle management | COMPLETE |
| **FreshnessPolicy** | Session freshness (FRESH/STALE_WARN/STALE_IDLE/STALE_COMPACT/STALE_RESET) | COMPLETE |
| **EventBus** | Pub/sub with exact/wildcard/prefix matching | COMPLETE |
| **ToolRegistry** | Tool registration, discovery, API format conversion | COMPLETE |
| **MCPManager** | MCP server connection management (structural placeholder) | COMPLETE |
| **MCPToolAdapter** | Wraps MCP server tools as Tool interface | COMPLETE |
| **Error Hierarchy** | GenyExecutorError → Pipeline/Stage/Guard/API/ToolExecution errors | COMPLETE |

---

## Dependencies

- `anthropic>=0.52.0` — Anthropic SDK (only external API dependency)
- `mcp>=1.0.0` — MCP SDK (for MCP server integration)
- `pydantic>=2.0` — Data validation
- **NO** LangChain, LangGraph, or other framework dependencies

---

## Source Structure

```
src/geny_executor/
├── __init__.py               # Public API exports
├── core/
│   ├── pipeline.py           # Pipeline engine (Phase A/B/C)
│   ├── stage.py              # Stage + Strategy ABCs (Dual Abstraction)
│   ├── state.py              # PipelineState dataclass
│   ├── config.py             # PipelineConfig + ModelConfig
│   ├── result.py             # PipelineResult
│   ├── errors.py             # Error hierarchy
│   ├── builder.py            # PipelineBuilder fluent API
│   └── presets.py            # PipelinePresets
├── events/
│   ├── bus.py                # EventBus
│   └── types.py              # PipelineEvent
├── stages/
│   ├── s01_input/            # Input validation + normalization
│   ├── s02_context/          # Context loading + history compaction
│   ├── s03_system/           # System prompt building
│   ├── s04_guard/            # Safety guards (budget/iteration/permission)
│   ├── s05_cache/            # Prompt caching strategies
│   ├── s06_api/              # API providers + retry strategies
│   ├── s07_token/            # Token tracking + cost calculation
│   ├── s08_think/            # Extended Thinking processing
│   ├── s09_parse/            # Response parsing + signal detection
│   ├── s10_tool/             # Tool execution + routing
│   ├── s11_agent/            # Multi-agent orchestration
│   ├── s12_evaluate/         # Response quality evaluation
│   ├── s13_loop/             # Loop control
│   ├── s14_emit/             # Result output (text/VTuber/TTS)
│   ├── s15_memory/           # Memory persistence
│   └── s16_yield/            # Final result formatting
├── tools/
│   ├── base.py               # Tool ABC + ToolResult + ToolContext
│   ├── registry.py           # ToolRegistry
│   └── mcp/                  # MCP integration
│       ├── manager.py        # MCPManager
│       └── adapter.py        # MCPToolAdapter
└── session/
    ├── session.py            # Session class
    ├── manager.py            # SessionManager
    └── freshness.py          # FreshnessPolicy
```
