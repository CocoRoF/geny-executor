# geny-executor

[![PyPI version](https://img.shields.io/pypi/v/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml)

**Harness-engineered agent pipeline library built on the Anthropic API.**

geny-executor implements a **16-stage pipeline** with **dual-abstraction architecture** — inspired by Claude Code's agent loop and Anthropic's harness design principles. No LangChain. No LangGraph. Just a clean, modular pipeline that gives you full control over every step of agent execution.

[한국어 README](README_ko.md)

---

## Why geny-executor?

| Problem | geny-executor's Answer |
|---------|----------------------|
| Frameworks hide too much behind abstractions | Every stage is explicit and inspectable |
| Hard to customize one part without rewriting everything | **Dual Abstraction** — swap stages *or* strategies within stages |
| Agent loops are opaque black boxes | 16 clearly defined stages with event-driven observability |
| Vendor lock-in across multiple LLM providers | Single dependency: Anthropic SDK. One provider, done right |
| Cost tracking is an afterthought | Built-in token tracking, cost calculation, and budget guards |

---

## Architecture

### The 16-Stage Pipeline

```
Phase A (once):   [1: Input]
Phase B (loop):   [2: Context] → [3: System] → [4: Guard] → [5: Cache]
                  → [6: API] → [7: Token] → [8: Think] → [9: Parse]
                  → [10: Tool] → [11: Agent] → [12: Evaluate] → [13: Loop]
Phase C (once):   [14: Emit] → [15: Memory] → [16: Yield]
```

| # | Stage | Purpose | Example Strategies |
|---|-------|---------|--------------------|
| 1 | **Input** | Validate & normalize user input | Default, Strict, Schema, Multimodal |
| 2 | **Context** | Load conversation history & memory | SimpleLoad, ProgressiveDisclosure, VectorSearch |
| 3 | **System** | Build system prompt | Static, Composable, Adaptive |
| 4 | **Guard** | Safety checks & budget enforcement | TokenBudget, Cost, RateLimit, Permission |
| 5 | **Cache** | Optimize prompt caching | NoCache, System, Aggressive, Adaptive |
| 6 | **API** | Call Anthropic Messages API | Anthropic, Mock, Recording, Replay |
| 7 | **Token** | Track usage & calculate costs | Default, Detailed + AnthropicPricing |
| 8 | **Think** | Process extended thinking blocks | Passthrough, ExtractAndStore, Filter |
| 9 | **Parse** | Parse response & detect completion | Default, StructuredOutput + SignalDetector |
| 10 | **Tool** | Execute tool calls | Sequential, Parallel + RegistryRouter |
| 11 | **Agent** | Multi-agent orchestration | SingleAgent, Delegate, Evaluator |
| 12 | **Evaluate** | Judge quality & completion | SignalBased, CriteriaBased, AgentEval |
| 13 | **Loop** | Decide: continue or finish? | Standard, SingleTurn, BudgetAware |
| 14 | **Emit** | Output results | Text, Callback, VTuber, TTS, Streaming |
| 15 | **Memory** | Persist conversation memory | AppendOnly, Reflective + File/SQLite |
| 16 | **Yield** | Format final result | Default, Structured, Streaming |

### Dual Abstraction

```
┌─ Level 1: Stage Abstraction ─────────────────────────┐
│  Swap entire stage modules in/out of the pipeline     │
│                                                       │
│  ┌─ Level 2: Strategy Abstraction ─────────────────┐  │
│  │  Swap internal logic within a stage              │  │
│  │                                                  │  │
│  │  ContextStage can use:                           │  │
│  │    → SimpleLoadStrategy (default)                │  │
│  │    → ProgressiveDisclosureStrategy               │  │
│  │    → VectorSearchStrategy                        │  │
│  │    → YourCustomStrategy                          │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Level 1:** Replace an entire stage (e.g., swap `APIStage` for a custom provider).  
**Level 2:** Change behavior *within* a stage (e.g., switch context loading strategy from simple to vector search).

---

## Installation

```bash
pip install geny-executor
```

With optional dependencies:

```bash
# Memory features (numpy for vector operations)
pip install geny-executor[memory]

# All optional dependencies
pip install geny-executor[all]

# Development
pip install geny-executor[dev]
```

### Requirements

- Python 3.11+
- Anthropic API key

---

## Quick Start

### Minimal Pipeline

The simplest possible agent — input, API call, parse, output:

```python
import asyncio
from geny_executor import PipelinePresets

async def main():
    pipeline = PipelinePresets.minimal(api_key="sk-ant-...")
    result = await pipeline.run("What is the capital of France?")
    print(result.text)

asyncio.run(main())
```

### Chat Pipeline

Full conversational agent with history, system prompt, and tool support:

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

### Agent Pipeline (All 16 Stages)

Autonomous agent with tools, evaluation, memory, and loop control:

```python
from geny_executor import PipelinePresets
from geny_executor.tools import ToolRegistry, Tool, ToolResult, ToolContext

class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web for information"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        # Your search implementation
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

### Custom Pipeline with Builder

Fine-grained control over every stage:

```python
from geny_executor import PipelineBuilder

pipeline = (
    PipelineBuilder("my-agent", api_key="sk-ant-...")
    .with_model(model="claude-sonnet-4-20250514", max_tokens=4096)
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

### Manual Pipeline Construction

For maximum control, assemble stages directly:

```python
from geny_executor import Pipeline, PipelineConfig
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage

config = PipelineConfig(name="custom", api_key="sk-ant-...")
pipeline = Pipeline(config)

pipeline.register_stage(InputStage())
pipeline.register_stage(APIStage(provider=MockProvider(responses=["Hello!"])))
pipeline.register_stage(ParseStage())
pipeline.register_stage(YieldStage())

result = await pipeline.run("Test input")
```

---

## Sessions

Persistent state management across multiple interactions:

```python
from geny_executor import PipelinePresets
from geny_executor.session import SessionManager

manager = SessionManager()
pipeline = PipelinePresets.chat(api_key="sk-ant-...")

# Create a session — state persists across runs
session = manager.create(pipeline)
result1 = await session.run("My name is Alice")
result2 = await session.run("What's my name?")  # Remembers context

# List active sessions
for info in manager.list_sessions():
    print(f"Session {info.session_id}: {info.message_count} messages, ${info.total_cost_usd:.4f}")
```

---

## Event System

Real-time observability with pub/sub events:

```python
from geny_executor import PipelinePresets

pipeline = PipelinePresets.agent(api_key="sk-ant-...")

# Subscribe to specific events
@pipeline.on("stage.enter")
async def on_stage_enter(event):
    print(f"  → Entering: {event.stage}")

@pipeline.on("stage.exit")
async def on_stage_exit(event):
    print(f"  ← Exiting: {event.stage}")

@pipeline.on("pipeline.complete")
async def on_complete(event):
    print(f"Done! Iterations: {event.data.get('iterations')}")

# Wildcard: listen to all events
@pipeline.on("*")
async def on_any(event):
    pass  # Log everything

result = await pipeline.run("Hello")
```

### Streaming

```python
async for event in pipeline.run_stream("Solve this step by step"):
    if event.type == "stage.enter":
        print(f"Stage: {event.stage}")
    elif event.type == "api.response":
        print(f"Response received")
    elif event.type == "pipeline.complete":
        print(f"Final: {event.data['result'].text}")
```

---

## Tool System

### Creating Tools

```python
from geny_executor.tools import Tool, ToolResult, ToolContext

class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform arithmetic calculations"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        try:
            result = eval(input["expression"])  # Use a safe evaluator in production
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=str(e), is_error=True)
```

### Tool Registry

```python
from geny_executor.tools import ToolRegistry

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(SearchTool())

# Filter tools per request
math_tools = registry.filter(include=["calculator"])
api_format = registry.to_api_format()  # Anthropic API format
```

### MCP Integration

Connect to Model Context Protocol servers:

```python
from geny_executor.tools.mcp import MCPManager

mcp = MCPManager()
await mcp.connect("filesystem", command="npx", args=["-y", "@anthropic/mcp-filesystem"])

# MCP tools are automatically adapted to the Tool interface
for tool in mcp.list_tools():
    registry.register(tool)
```

---

## Error Handling

Structured error hierarchy with automatic classification:

```python
from geny_executor import (
    GenyExecutorError,   # Base exception
    PipelineError,       # Pipeline-level errors
    StageError,          # Stage execution errors
    GuardRejectError,    # Guard rejection (budget, permissions)
    APIError,            # Anthropic API errors (with category)
    ToolExecutionError,  # Tool failures
    ErrorCategory,       # rate_limited, timeout, token_limit, etc.
)

try:
    result = await pipeline.run("input")
except GuardRejectError as e:
    print(f"Blocked by guard: {e}")
except APIError as e:
    print(f"API error ({e.category}): {e}")
    if e.category == ErrorCategory.rate_limited:
        # Handle rate limiting
        pass
except GenyExecutorError as e:
    print(f"Pipeline error: {e}")
```

---

## Pipeline Presets

Five ready-to-use configurations:

| Preset | Stages | Use Case |
|--------|--------|----------|
| `PipelinePresets.minimal()` | 1→6→9→16 | Simple Q&A, testing |
| `PipelinePresets.chat()` | 1→2→3→4→5→6→7→9→13→16 | Conversational chatbot |
| `PipelinePresets.agent()` | All 16 | Autonomous agent with tools |
| `PipelinePresets.evaluator()` | 1→3→6→9→12→16 | Quality evaluation |
| `PipelinePresets.geny_vtuber()` | All 16 + VTuber emit | Geny VTuber system |

---

## Custom Stages & Strategies

### Creating a Custom Strategy

```python
from geny_executor.core.stage import Strategy

class MyContextStrategy(Strategy):
    @property
    def name(self) -> str:
        return "my_context"

    @property
    def description(self) -> str:
        return "Custom context loading with RAG"

    def configure(self, config: dict) -> None:
        self.top_k = config.get("top_k", 5)

    async def load(self, state):
        # Your custom context loading logic
        ...
```

### Creating a Custom Stage

```python
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState

class LoggingStage(Stage[dict, dict]):
    @property
    def name(self) -> str:
        return "logging"

    @property
    def order(self) -> int:
        return 7  # After API, before Think

    @property
    def category(self) -> str:
        return "execution"

    async def execute(self, input: dict, state: PipelineState) -> dict:
        print(f"[{state.iteration}] API response received")
        return input  # Pass through

# Register into pipeline
pipeline.register_stage(LoggingStage())
```

---

## Configuration Reference

### ModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Anthropic model ID |
| `max_tokens` | `int` | `8192` | Max output tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `thinking_enabled` | `bool` | `False` | Enable extended thinking |
| `thinking_budget_tokens` | `int` | `10000` | Thinking token budget |

### PipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Pipeline name |
| `api_key` | `str` | required | Anthropic API key |
| `model` | `ModelConfig` | default | Model configuration |
| `max_iterations` | `int` | `50` | Max loop iterations |
| `cost_budget_usd` | `float?` | `None` | Cost budget limit |
| `context_window_budget` | `int` | `200_000` | Context window token limit |
| `stream` | `bool` | `False` | Enable streaming mode |
| `single_turn` | `bool` | `False` | Single turn (no loop) |

---

## Project Structure

```
geny-executor/
├── src/geny_executor/
│   ├── __init__.py          # Public API
│   ├── py.typed             # PEP 561 type marker
│   ├── core/                # Pipeline engine, config, state, errors
│   ├── events/              # EventBus pub/sub system
│   ├── stages/              # 16 pipeline stages (s01-s16)
│   ├── tools/               # Tool system + MCP integration
│   └── session/             # Session management + freshness
├── tests/                   # 81 unit & integration tests
├── pyproject.toml           # Package configuration (Hatch)
└── LICENSE                  # MIT License
```

---

## Development

```bash
# Clone
git clone https://github.com/CocoRoF/geny-executor.git
cd geny-executor

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=geny_executor --cov-report=term-missing

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

---

## Roadmap

- [ ] Streaming response support (full implementation)
- [ ] OpenTelemetry integration for tracing
- [ ] Additional memory backends (Redis, PostgreSQL)
- [ ] WebUI pipeline configurator
- [ ] Plugin system for community stages
- [ ] Batch execution mode

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Related Projects

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — The foundation geny-executor is built on
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol for tool integration
- [Claude Code](https://claude.ai/code) — The agent loop that inspired this architecture
