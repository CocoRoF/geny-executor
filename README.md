# geny-executor

[![PyPI version](https://img.shields.io/pypi/v/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml)

**A harness-engineered agent pipeline library — 21 stages, 5 LLM providers, MCP-native, fully introspectable.**

geny-executor implements a **21-stage pipeline** with **dual-abstraction architecture** (stage slots × strategy slots). Inspired by Claude Code's agent loop and Anthropic's harness design principles. No LangChain. No LangGraph. Just an explicit, modular pipeline where every step is observable, mutatable, and swappable.

[한국어 README](README_ko.md) · [Architecture](docs/architecture.md) · [Providers](docs/providers.md) · [Error codes](docs/error_codes.md) · [Claude Code CLI host](docs/claude_code_cli.md)

---

## The Geny ecosystem

These projects are built to work together. **Geny** is the product at the top of the stack; everything below is a building block you can also use on its own. **➡️ marks where you are.**

| Project | What it is | Role in the stack |
|---|---|---|
| [**Geny**](https://github.com/CocoRoF/Geny) | Multi-agent VTuber + autonomous-worker platform | The product — uses every project below |
| ➡️ [**geny-executor**](https://github.com/CocoRoF/geny-executor) | 21-stage, manifest-driven agent pipeline · PyPI · Apache-2.0 | The engine everything runs on |
| [**GAPT**](https://github.com/CocoRoF/geny-adapted-project-toolkit) | Self-hosted AI DevOps platform — sandbox · edit · build · deploy | Where agents safely touch real repos |
| [**geny-avatar**](https://github.com/CocoRoF/geny-avatar) | 2D live-avatar editor with AI texture generation | Where Geny's faces are made |

<details>
<summary>How they fit together</summary>

```
                  Geny — the product (uses everything below)
                    │
      ┌─────────────┼──────────────┐
 agent engine    avatars      sandbox + deploy
      │             │              │
      ▼             ▼              ▼
 geny-executor  geny-avatar      GAPT
  (the engine)  (avatar editor)  (AI DevOps platform)
```

</details>

---

<!-- 📸 IMAGE NEEDED: hero banner — the 21-stage pipeline as a clean horizontal flow graphic -->
> 📸 **Image needed** — _hero banner: the 21-stage pipeline rendered as a polished flow graphic._

---

## Why geny-executor?

| Problem | geny-executor's answer |
|---|---|
| Frameworks hide too much behind abstractions | Every one of the 21 stages is explicit, inspectable, and individually swappable. |
| Hard to customize one part without rewriting everything | **Dual abstraction**: swap a whole stage *or* swap a strategy inside a stage. Manifest-driven so config = artifact. |
| Vendor lock-in across LLM providers | One contract, five providers wired in (`anthropic` / `openai` / `google` / `vllm` / `claude_code_cli`). Switch by editing one config field. |
| Agent loops are opaque black boxes | Event-bus + stable structured error codes ([`exec.cli.auth_failed`, …](docs/error_codes.md)) — every failure groups cleanly in your logs / Sentry / i18n layer. |
| MCP integration is a side concern | First-class. Host-attached MCP servers + per-session MCP wraps for CLI backends (e.g. Claude Code CLI) ship out of the box. |
| Cost tracking is an afterthought | Built into Stage 7 (Token). Per-call cost, per-session ledger, budget guards. |

---

## Architecture at a glance

### The 21-stage pipeline

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

The full stage list with strategy options lives in [`docs/architecture.md`](docs/architecture.md).

### Dual abstraction — two levels of swap

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

- **Stage Abstraction** — replace a whole stage (e.g. drop a custom `APIStage` for a private provider).
- **Strategy Abstraction** — change behaviour *inside* a stage (e.g. switch context loading from `SimpleLoad` to `VectorSearch`) without touching the surrounding pipeline.

---

## Installation

```bash
pip install geny-executor
```

Optional extras:

```bash
pip install geny-executor[memory]   # numpy for vector retrieval
pip install geny-executor[all]      # everything
pip install geny-executor[dev]      # dev/test tooling
```

**Requirements**: Python 3.11+. At least one provider's credentials (Anthropic API key, OpenAI API key, …) or a local CLI binary (`claude` for `claude_code_cli`).

---

## Quick start

### Minimal pipeline

```python
import asyncio
from geny_executor import PipelinePresets

async def main():
    pipeline = PipelinePresets.minimal(api_key="sk-ant-...")
    result = await pipeline.run("What is the capital of France?")
    print(result.text)

asyncio.run(main())
```

### Chat pipeline (history + system prompt + optional tools)

```python
from geny_executor import PipelinePresets

pipeline = PipelinePresets.chat(
    api_key="sk-ant-...",
    system_prompt="You are a helpful coding assistant.",
)

result = await pipeline.run("Explain Python decorators")
print(result.text)
print(f"Cost: ${result.total_cost_usd:.4f}")
```

### Full agent (all 21 stages — tools, evaluation, memory, loop control)

```python
from geny_executor import PipelinePresets
from geny_executor.tools import ToolRegistry, Tool, ToolResult, ToolContext

class SearchTool(Tool):
    @property
    def name(self) -> str: return "search"
    @property
    def description(self) -> str: return "Search the web for information"
    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    async def execute(self, input, context):
        return ToolResult(content=f"Results for: {input['query']}")

registry = ToolRegistry()
registry.register(SearchTool())

pipeline = PipelinePresets.agent(
    api_key="sk-ant-...",
    system_prompt="You are a research assistant. Use tools to find answers.",
    tools=registry,
    max_turns=20,
)

result = await pipeline.run("Find the latest Python release version")
```

### Custom pipeline with builder

```python
from geny_executor import PipelineBuilder

pipeline = (
    PipelineBuilder("my-agent", api_key="sk-ant-...")
    .with_model(model="claude-sonnet-4-6", max_tokens=4096)
    .with_system(prompt="You are a concise assistant.")
    .with_context()
    .with_guard(cost_budget_usd=1.0, max_iterations=30)
    .with_cache(strategy="aggressive")
    .with_tools(registry=my_registry)
    .with_think(enabled=True, budget_tokens=10000)
    .with_evaluate()
    .with_loop(max_turns=30)
    .with_memory()
    .build()
)

result = await pipeline.run("Complex multi-step task here")
```

### Manifest-driven pipeline (recommended for hosts)

```python
from geny_executor import Pipeline, CredentialBundle, ProviderCredentials, EnvironmentManifest

manifest = EnvironmentManifest.load("./envs/my_env.json")
credentials = CredentialBundle(by_provider={
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
})
pipeline = await Pipeline.from_manifest_async(manifest, credentials=credentials)
result = await pipeline.run("Hello!")
```

See [`docs/manifest.md`](docs/manifest.md) for the full schema.

---

## Five LLM providers, one contract

| Provider | Notes |
|---|---|
| `anthropic` | Claude family. Full streaming, native `tool_use`, thinking blocks. |
| `openai` | GPT-4.1 / o-series. Streaming, tools, JSON-schema structured output. |
| `google` | Gemini 3.x / 2.5. Streaming, tools, thinking blocks. |
| `vllm` | Any model on a local vLLM endpoint. OpenAI-compatible. Tools opt-in via `configure_capabilities()`. |
| `claude_code_cli` | Subprocess-driven Claude Code CLI. **Hosts attach a per-session MCP bridge** to surface their own tool registry to the spawned CLI's LLM. See [`docs/claude_code_cli.md`](docs/claude_code_cli.md). |

A session picks its provider via `stages[6].config["provider"]` in the manifest. Credentials flow through a single `CredentialBundle` channel — see [`docs/providers.md`](docs/providers.md).

---

## Error codes (2.1.0+)

Every executor exception carries a stable `exec.<component>.<reason>` code:

```python
from geny_executor import APIError, ExecutorErrorCode, ErrorCategory

try:
    result = await pipeline.run("...")
except APIError as e:
    if e.code is ExecutorErrorCode.EXEC_CLI_AUTH_FAILED:
        print("Please re-login to Claude Code CLI.")
    elif e.category.is_recoverable:
        print(f"Recoverable failure ({e.code.value}); retrying.")
```

Structured event payloads also carry the code:

```json
{
  "type": "pipeline.error",
  "data": {
    "error": "Claude Code CLI is not authenticated …",
    "code": "exec.cli.auth_failed",
    "exception_type": "geny_executor.core.errors.APIError"
  }
}
```

Codes are **stable across releases** — see [`docs/error_codes.md`](docs/error_codes.md) for the full table, recoverability, and how to add a new code.

---

## Sessions

Persistent state across multiple interactions:

```python
from geny_executor import PipelinePresets
from geny_executor.session import SessionManager

manager = SessionManager()
pipeline = PipelinePresets.chat(api_key="sk-ant-...")
session = manager.create(pipeline)

await session.run("My name is Alice")
result = await session.run("What's my name?")

for info in manager.list_sessions():
    print(f"{info.session_id}: {info.message_count} msgs, ${info.total_cost_usd:.4f}")
```

---

## Event system + observability

```python
@pipeline.on("stage.enter")
async def _(event):
    print(f"→ {event.stage}")

@pipeline.on("pipeline.error")
async def _(event):
    print(f"❌ {event.data['code']}: {event.data['error']}")

@pipeline.on("*")
async def _(event):
    pass   # firehose
```

Streaming:

```python
async for event in pipeline.run_stream("Solve step by step"):
    if event.type == "stage.enter":
        print(f"Stage: {event.stage}")
    elif event.type == "pipeline.complete":
        print(f"Final: {event.data['result'].text}")
```

---

## Tools + MCP

```python
from geny_executor.tools import Tool, ToolResult, ToolContext, ToolRegistry

class Calculator(Tool):
    @property
    def name(self): return "calculator"
    @property
    def description(self): return "Perform arithmetic."
    @property
    def input_schema(self):
        return {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    async def execute(self, input, context):
        return ToolResult(content=str(eval(input["expression"])))   # use a safe evaluator!

registry = ToolRegistry()
registry.register(Calculator())
```

Connect a host-attached MCP server:

```python
from geny_executor.tools.mcp import MCPManager

mcp = MCPManager()
await mcp.connect("filesystem", command="npx", args=["-y", "@anthropic/mcp-filesystem"])
for tool in mcp.list_tools():
    registry.register(tool)
```

For the **CLI-side** MCP wrap (your tool registry exposed *into* a spawned Claude Code CLI's LLM), see [`docs/claude_code_cli.md`](docs/claude_code_cli.md).

---

## Pipeline presets

| Preset | Active stages | Use case |
|---|---|---|
| `PipelinePresets.minimal()` | Input → API → Parse → Yield | Quick Q&A, smoke tests |
| `PipelinePresets.chat()` | + Context, System, Guard, Cache, Token, Tool, Loop, Memory | Conversational chatbot |
| `PipelinePresets.agent()` | All 21 stages active | Autonomous agent with tools, eval, memory, summarisation, persistence |
| `PipelinePresets.evaluator()` | Input → System → API → Parse → Evaluate → Yield | Generator/Evaluator quality pass |
| `PipelinePresets.geny_vtuber()` | All 21 stages + VTuber/TTS emitters | Reference reproduction of the Geny VTuber harness |

---

## Custom stages + strategies

```python
from geny_executor.core.stage import Strategy

class MyContextStrategy(Strategy):
    name = "my_context"
    description = "Custom context loading with RAG"

    def configure(self, config: dict) -> None:
        self.top_k = config.get("top_k", 5)

    async def load(self, state):
        ...   # your RAG retrieval
```

```python
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState

class LoggingStage(Stage[dict, dict]):
    name = "logging"
    order = 7      # after API, before Think
    category = "execution"

    async def execute(self, input, state: PipelineState):
        print(f"[{state.iteration}] API response received")
        return input

pipeline.register_stage(LoggingStage())
```

---

## Project structure

```
geny-executor/
├── src/geny_executor/
│   ├── __init__.py          # Public API surface
│   ├── py.typed             # PEP 561 type marker
│   ├── core/                # Pipeline engine, errors, manifest, mutation, snapshot
│   ├── stages/              # 21 pipeline stages (s01–s21)
│   ├── llm_client/          # 5 providers + ClientRegistry + CredentialBundle + CLI runtime
│   ├── tools/               # Tool ABC, registry, router, MCP integration
│   ├── hooks/               # PRE/POST tool-use lifecycle hooks
│   ├── memory/              # Memory v2 retrieval, vault map, vector store
│   ├── skills/              # SkillProvider + skill loading
│   ├── subagents/           # Stage 12 sub-agent orchestration
│   ├── permission/          # Per-tool ACL evaluated by RegistryRouter
│   ├── channels/            # Output channel adapters (text, callback, TTS, …)
│   ├── cron/                # Scheduled trigger support
│   ├── events/              # EventBus pub/sub
│   ├── history/             # Conversation history primitives
│   ├── telemetry/           # Event / metric exporters
│   └── session/             # Session manager + freshness checks
├── docs/                    # Architecture, providers, manifest, error codes, MCP, hooks
├── tests/                   # 3100+ unit, conformance, contract, integration tests
├── pyproject.toml           # Package configuration (Hatch)
└── LICENSE                  # Apache-2.0
```

---

## Development

```bash
git clone https://github.com/CocoRoF/geny-executor.git
cd geny-executor

pip install -e ".[dev]"

pytest                                                       # full suite (~30s, 3100+ tests)
pytest tests/contract/test_error_codes_stability.py          # error code stability check
pytest --cov=geny_executor --cov-report=term-missing         # coverage

ruff check src/ tests/
ruff format src/ tests/
```

---

## Versioning

| Version | Highlights |
|---|---|
| **2.1.0** | `ExecutorErrorCode` taxonomy + structured `pipeline.error` / `stage.error` / `api.retry` payloads. `docs/error_codes.md`. |
| **2.0.6** | Removed `copilot_cli` provider (text-only, can't host tool round-trip). Upstreamed Geny's claude_code_cli compat patches (`--verbose` injection, `--bare` strip, drop auto-`--tools ""`, `tool_use` strip from finalize). |
| **2.0.5** | `APIRequest.mcp_config` per-request override + auto-emit `--strict-mcp-config`. Foundational support for the host MCP wrap. |
| **2.0.0** | Provider abstraction (`ClientRegistry`, `CredentialBundle`). Manifest single source of truth for Stage 6 provider. |
| **1.x** | Original 16-stage pipeline; Anthropic-only. |

See [CHANGELOG](https://github.com/CocoRoF/geny-executor/releases) for the full history.

---

## License

[Apache License 2.0](LICENSE). Copyright 2026 CocoRoF — see [NOTICE](NOTICE).

---

## Related projects

**The Geny ecosystem** (sibling projects built on this engine) → see [The Geny ecosystem](#the-geny-ecosystem) above:
[Geny](https://github.com/CocoRoF/Geny) · [GAPT](https://github.com/CocoRoF/geny-adapted-project-toolkit) · [geny-avatar](https://github.com/CocoRoF/geny-avatar)

**Built on & interoperates with:**

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Google GenAI SDK](https://github.com/googleapis/python-genai)
- [vLLM](https://github.com/vllm-project/vllm)
- [Claude Code CLI](https://docs.anthropic.com/claude/code/) — geny-executor hosts it via the `claude_code_cli` provider
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol; both host-attached servers and per-session CLI wraps are first-class
