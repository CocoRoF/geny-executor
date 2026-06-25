# geny-executor

[![PyPI version](https://img.shields.io/pypi/v/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml)

**하네스 엔지니어링 기반 Agent 파이프라인 라이브러리 — 21단계, 5개 LLM provider, MCP 네이티브, 완전 introspectable.**

geny-executor는 **21단계 파이프라인**과 **이중 추상화 (Dual Abstraction)** 아키텍처(stage slot × strategy slot)를 구현합니다. Claude Code의 agent loop과 Anthropic의 하네스 설계 원칙에서 영감을 받았습니다. LangChain 없음. LangGraph 없음. 모든 단계가 명시적이며 관찰 가능하고, 변경/교체 가능한 파이프라인입니다.

[English README](README.md) · [아키텍처](docs/architecture.md) · [Providers](docs/providers.md) · [Error codes](docs/error_codes.md) · [Claude Code CLI 호스트](docs/claude_code_cli.md)

<!-- 📸 IMAGE NEEDED: hero 배너 — 21단계 파이프라인을 깔끔한 흐름 그래픽으로, 또는 geny-executor-web 실행 스크린샷 -->
> 📸 **이미지 필요** — _hero 배너: 21단계 파이프라인 흐름 그래픽 (또는 [geny-executor-web](https://github.com/CocoRoF/geny-executor-web) 실시간 스크린샷)._

---

## 🌐 Geny 에코시스템

geny-executor 는 스택 맨 아래의 **엔진**입니다. 에코시스템의 나머지는 전부 이 위에 세워지며, 제품인 **Geny** 가 최종적으로 그 모두를 사용합니다.

<!-- 📸 IMAGE NEEDED: 아래 ASCII 맵을 대체할 다이어그램 -->

```
                  Geny — 최상위 제품 (아래 전부를 사용)
                    │
      ┌─────────────┼──────────────┐
   에이전트 엔진     아바타        샌드박스 + 배포
      │             │              │
      ▼             ▼              ▼
 geny-executor  geny-avatar      GAPT
  (엔진)        (아바타 에디터)  (AI DevOps 플랫폼)
      ▲
      │ 엔진을 시각화 / 구동
      │
 geny-executor-web (파이프라인 스튜디오)
```

| 프로젝트 | 무엇인가 | 스택에서의 역할 |
|---|---|---|
| [**Geny**](https://github.com/CocoRoF/Geny) | 멀티 에이전트 VTuber + 자율 워커 플랫폼 | 🏛️ 최상위 제품 — 아래 전부를 사용 |
| [**geny-executor**](https://github.com/CocoRoF/geny-executor) | 21단계 manifest 기반 에이전트 파이프라인 · PyPI · MIT | ⚙️ 모든 것이 돌아가는 엔진 |
| [**geny-executor-web**](https://github.com/CocoRoF/geny-executor-web) | 파이프라인 시각화 스튜디오 — React Flow + 실시간 WebSocket | 🔬 엔진을 보고·검사하고·실행 |
| [**GAPT**](https://github.com/CocoRoF/geny-adapted-project-toolkit) | 셀프호스트 AI DevOps 플랫폼 — 샌드박스·편집·빌드·배포 | 🛠️ 에이전트가 실제 레포를 안전하게 다루는 곳 |
| [**geny-avatar**](https://github.com/CocoRoF/geny-avatar) | AI 텍스처 생성 기반 2D 라이브 아바타 에디터 | 🎭 Geny 의 얼굴이 만들어지는 곳 |

> **➡️ 현재 위치: `geny-executor`** — 나머지 전부가 그 위에 세워지는 엔진.

---

## 왜 geny-executor 인가?

| 문제 | geny-executor의 답 |
|---|---|
| 프레임워크가 너무 많이 숨김 | 21단계 stage 하나하나가 명시적이고 introspectable, 각각 swap 가능. |
| 한 부분만 바꾸려면 전체 다시 써야 함 | **이중 추상화**: stage 통째로 교체하거나 stage 내부의 strategy만 교체. manifest-driven으로 config = artifact. |
| LLM provider vendor lock-in | 하나의 contract, 5개 provider 즉시 사용 가능 (`anthropic` / `openai` / `google` / `vllm` / `claude_code_cli`). config field 하나 바꾸면 끝. |
| Agent loop이 불투명한 블랙박스 | event-bus + stable structured error codes (예: [`exec.cli.auth_failed`](docs/error_codes.md)) — 모든 실패가 로그/Sentry/i18n 레이어에서 깔끔하게 그룹핑됨. |
| MCP 통합이 사이드 컨선 | first-class. 호스트가 attach한 MCP 서버 + CLI backend용 per-session MCP wrap 둘 다 기본 지원. |
| Cost tracking이 후순위 | Stage 7 (Token)에 내장. per-call 비용, per-session 원장, budget guard. |

---

## 아키텍처 한눈에

### 21단계 파이프라인

```
Phase A — Setup (턴마다 1회)
  1: Input  →  2: Context  →  3: System  →  4: Guard  →  5: Cache

Phase B — Generate + Dispatch (loop)
  6: API  →  7: Token  →  8: Think  →  9: Parse
  → 10: Tool  →  11: ToolReview  →  12: Agent  →  13: TaskRegistry
  → 14: Evaluate  →  15: HITL  →  16: Loop

Phase C — Surface (1회)
  17: Emit  →  18: Memory  →  19: Summarize  →  20: Persist  →  21: Yield
```

각 stage의 strategy 옵션을 포함한 전체 리스트는 [`docs/architecture.md`](docs/architecture.md) 참조.

> 💡 **실시간으로 보세요.** 모든 stage·이벤트·비용이 [**geny-executor-web**](https://github.com/CocoRoF/geny-executor-web) 로 스트리밍됩니다 — 실행하는 그대로 파이프라인을 그려주는 인터랙티브 React Flow 스튜디오.
>
> <!-- 📸 IMAGE NEEDED: geny-executor-web 스크린샷 — 실행 중인 21단계 흐름 그래프 + 이벤트 로그 -->
> 📸 _이미지 필요: geny-executor-web 실행 중 화면 (stage 그래프 + 실시간 이벤트 로그)._

### 이중 추상화 — 두 단계의 swap

```
┌─ Level 1: Stage Abstraction ─────────────────────────┐
│   Stage 모듈 통째로 파이프라인에 in/out swap.          │
│                                                       │
│  ┌─ Level 2: Strategy Abstraction ─────────────────┐  │
│  │   stage 내부 로직만 swap.                        │  │
│  │                                                  │  │
│  │   ContextStage 에 적용 가능한 strategy:          │  │
│  │     → SimpleLoad     (기본)                      │  │
│  │     → ProgressiveDisclosure                      │  │
│  │     → VectorSearch                               │  │
│  │     → 사용자 정의                                │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

- **Stage Abstraction** — stage 전체 교체 (예: private provider용 custom `APIStage` drop-in).
- **Strategy Abstraction** — stage *내부* 동작만 교체 (예: context loading을 `SimpleLoad` → `VectorSearch` 로 전환). 주변 파이프라인 건드리지 않음.

---

## 설치

```bash
pip install geny-executor
```

선택적 extras:

```bash
pip install geny-executor[memory]   # vector retrieval용 numpy
pip install geny-executor[all]      # 전체
pip install geny-executor[dev]      # 개발/테스트 도구
```

**요구사항**: Python 3.11+. 최소 1개 provider의 자격증명 (Anthropic API key, OpenAI API key, …) 또는 로컬 CLI binary (`claude_code_cli` provider용 `claude`).

---

## Quick start

### 최소 파이프라인

```python
import asyncio
from geny_executor import PipelinePresets

async def main():
    pipeline = PipelinePresets.minimal(api_key="sk-ant-...")
    result = await pipeline.run("프랑스의 수도는?")
    print(result.text)

asyncio.run(main())
```

### 채팅 파이프라인 (history + system prompt + 선택적 tools)

```python
from geny_executor import PipelinePresets

pipeline = PipelinePresets.chat(
    api_key="sk-ant-...",
    system_prompt="당신은 친절한 코딩 도우미입니다.",
)

result = await pipeline.run("Python 데코레이터 설명해줘")
print(result.text)
print(f"비용: ${result.total_cost_usd:.4f}")
```

### 풀 agent (21단계 전체 — tools, evaluation, memory, loop control)

```python
from geny_executor import PipelinePresets
from geny_executor.tools import ToolRegistry, Tool, ToolResult, ToolContext

class SearchTool(Tool):
    @property
    def name(self) -> str: return "search"
    @property
    def description(self) -> str: return "웹 검색"
    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    async def execute(self, input, context):
        return ToolResult(content=f"검색 결과: {input['query']}")

registry = ToolRegistry()
registry.register(SearchTool())

pipeline = PipelinePresets.agent(
    api_key="sk-ant-...",
    system_prompt="당신은 리서치 보조 agent입니다. tool을 활용해 답을 찾으세요.",
    tools=registry,
    max_turns=20,
)

result = await pipeline.run("최신 Python 릴리즈 버전 찾기")
```

### Builder로 커스텀 파이프라인

```python
from geny_executor import PipelineBuilder

pipeline = (
    PipelineBuilder("my-agent", api_key="sk-ant-...")
    .with_model(model="claude-sonnet-4-6", max_tokens=4096)
    .with_system(prompt="간결한 답변만 주세요.")
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

result = await pipeline.run("복잡한 멀티스텝 작업")
```

### Manifest 기반 파이프라인 (호스트 권장)

```python
from geny_executor import Pipeline, CredentialBundle, ProviderCredentials, EnvironmentManifest

manifest = EnvironmentManifest.load("./envs/my_env.json")
credentials = CredentialBundle(by_provider={
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
})
pipeline = await Pipeline.from_manifest_async(manifest, credentials=credentials)
result = await pipeline.run("안녕!")
```

전체 스키마는 [`docs/manifest.md`](docs/manifest.md) 참조.

---

## 5개 LLM provider, 하나의 contract

| Provider | 특징 |
|---|---|
| `anthropic` | Claude 패밀리. 완전한 streaming, native `tool_use`, thinking blocks. |
| `openai` | GPT-4.1 / o-series. Streaming, tools, JSON-schema structured output. |
| `google` | Gemini 3.x / 2.5. Streaming, tools, thinking blocks. |
| `vllm` | 로컬 vLLM endpoint의 어떤 모델이든. OpenAI 호환. Tools는 `configure_capabilities()` 로 opt-in. |
| `claude_code_cli` | Subprocess 기반 Claude Code CLI. **호스트가 per-session MCP bridge** 를 attach해서 자신의 tool registry를 spawned CLI의 LLM에게 노출시킴. 자세히는 [`docs/claude_code_cli.md`](docs/claude_code_cli.md). |

세션은 manifest의 `stages[6].config["provider"]` 로 provider 선택. 자격증명은 하나의 `CredentialBundle` 채널로 흐름 — [`docs/providers.md`](docs/providers.md) 참조.

---

## Error codes (2.1.0+)

모든 executor exception은 stable한 `exec.<component>.<reason>` 코드를 carry함:

```python
from geny_executor import APIError, ExecutorErrorCode, ErrorCategory

try:
    result = await pipeline.run("...")
except APIError as e:
    if e.code is ExecutorErrorCode.EXEC_CLI_AUTH_FAILED:
        print("Claude Code CLI 재로그인 필요.")
    elif e.category.is_recoverable:
        print(f"복구 가능 ({e.code.value}); 재시도.")
```

구조화된 event payload도 code를 carry:

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

코드는 **버전 간 안정성 보장** — 전체 표, 복구 가능성, 새 코드 추가 가이드는 [`docs/error_codes.md`](docs/error_codes.md) 참조.

---

## Session

여러 호출에 걸친 상태 유지:

```python
from geny_executor import PipelinePresets
from geny_executor.session import SessionManager

manager = SessionManager()
pipeline = PipelinePresets.chat(api_key="sk-ant-...")
session = manager.create(pipeline)

await session.run("내 이름은 Alice야")
result = await session.run("내 이름이 뭐였지?")

for info in manager.list_sessions():
    print(f"{info.session_id}: {info.message_count}개 메시지, ${info.total_cost_usd:.4f}")
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
async for event in pipeline.run_stream("단계별로 풀어줘"):
    if event.type == "stage.enter":
        print(f"Stage: {event.stage}")
    elif event.type == "pipeline.complete":
        print(f"최종: {event.data['result'].text}")
```

---

## Tools + MCP

```python
from geny_executor.tools import Tool, ToolResult, ToolContext, ToolRegistry

class Calculator(Tool):
    @property
    def name(self): return "calculator"
    @property
    def description(self): return "산술 연산"
    @property
    def input_schema(self):
        return {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    async def execute(self, input, context):
        return ToolResult(content=str(eval(input["expression"])))   # 실제로는 안전한 evaluator 사용!

registry = ToolRegistry()
registry.register(Calculator())
```

호스트가 MCP 서버 attach:

```python
from geny_executor.tools.mcp import MCPManager

mcp = MCPManager()
await mcp.connect("filesystem", command="npx", args=["-y", "@anthropic/mcp-filesystem"])
for tool in mcp.list_tools():
    registry.register(tool)
```

**CLI 측** MCP wrap (호스트의 tool registry를 spawned Claude Code CLI의 LLM 안으로 노출)은 [`docs/claude_code_cli.md`](docs/claude_code_cli.md) 참조.

---

## Pipeline presets

| Preset | 활성 stage | 용도 |
|---|---|---|
| `PipelinePresets.minimal()` | Input → API → Parse → Yield | 빠른 Q&A, smoke test |
| `PipelinePresets.chat()` | + Context, System, Guard, Cache, Token, Tool, Loop, Memory | 대화형 챗봇 |
| `PipelinePresets.agent()` | 21단계 전체 활성 | tools, eval, memory, summarisation, persistence 갖춘 자율 agent |
| `PipelinePresets.evaluator()` | Input → System → API → Parse → Evaluate → Yield | Generator/Evaluator 품질 검증 |
| `PipelinePresets.geny_vtuber()` | 21단계 전체 + VTuber/TTS emitter | Geny VTuber 하네스 reference |

---

## 커스텀 stage + strategy

```python
from geny_executor.core.stage import Strategy

class MyContextStrategy(Strategy):
    name = "my_context"
    description = "RAG 기반 context loading"

    def configure(self, config: dict) -> None:
        self.top_k = config.get("top_k", 5)

    async def load(self, state):
        ...   # 사용자 RAG 검색
```

```python
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState

class LoggingStage(Stage[dict, dict]):
    name = "logging"
    order = 7      # API 다음, Think 이전
    category = "execution"

    async def execute(self, input, state: PipelineState):
        print(f"[{state.iteration}] API 응답 수신")
        return input

pipeline.register_stage(LoggingStage())
```

---

## 프로젝트 구조

```
geny-executor/
├── src/geny_executor/
│   ├── __init__.py          # Public API
│   ├── py.typed             # PEP 561 type marker
│   ├── core/                # Pipeline engine, errors, manifest, mutation, snapshot
│   ├── stages/              # 21단계 (s01–s21)
│   ├── llm_client/          # 5 provider + ClientRegistry + CredentialBundle + CLI runtime
│   ├── tools/               # Tool ABC, registry, router, MCP 통합
│   ├── hooks/               # PRE/POST tool-use lifecycle hooks
│   ├── memory/              # Memory v2 retrieval, vault map, vector store
│   ├── skills/              # SkillProvider + skill loading
│   ├── subagents/           # Stage 12 sub-agent orchestration
│   ├── permission/          # RegistryRouter가 평가하는 per-tool ACL
│   ├── channels/            # Output channel adapter (text, callback, TTS, …)
│   ├── cron/                # Scheduled trigger
│   ├── events/              # EventBus pub/sub
│   ├── history/             # Conversation history primitive
│   ├── telemetry/           # Event / metric exporter
│   └── session/             # Session manager + freshness check
├── docs/                    # Architecture, providers, manifest, error codes, MCP, hooks
├── tests/                   # 3100+ unit / conformance / contract / integration test
├── pyproject.toml           # Package config (Hatch)
└── LICENSE                  # MIT
```

---

## 개발

```bash
git clone https://github.com/CocoRoF/geny-executor.git
cd geny-executor

pip install -e ".[dev]"

pytest                                                       # 전체 (~30s, 3100+ tests)
pytest tests/contract/test_error_codes_stability.py          # error code 안정성 검사
pytest --cov=geny_executor --cov-report=term-missing         # 커버리지

ruff check src/ tests/
ruff format src/ tests/
```

---

## 버전 히스토리

| 버전 | 주요 변경 |
|---|---|
| **2.1.0** | `ExecutorErrorCode` taxonomy + 구조화된 `pipeline.error` / `stage.error` / `api.retry` payload. `docs/error_codes.md`. |
| **2.0.6** | `copilot_cli` provider 제거 (text-only, tool round-trip 불가). Geny 측 claude_code_cli 호환 patch 4종 upstream (`--verbose` 주입, `--bare` strip, auto-`--tools ""` drop, finalize에서 `tool_use` strip). |
| **2.0.5** | `APIRequest.mcp_config` per-request override + `--strict-mcp-config` 자동 emit. 호스트 MCP wrap 토대. |
| **2.0.0** | Provider 추상화 (`ClientRegistry`, `CredentialBundle`). Stage 6 provider의 manifest single source of truth. |
| **1.x** | 원형 16단계 파이프라인; Anthropic only. |

전체 히스토리는 [CHANGELOG](https://github.com/CocoRoF/geny-executor/releases) 참조.

---

## 라이선스

MIT — [LICENSE](LICENSE) 참조.

---

## 관련 프로젝트

**Geny 에코시스템** (이 엔진 위에 세워진 형제 프로젝트) → 위 [Geny 에코시스템](#-geny-에코시스템) 섹션 참조:
[Geny](https://github.com/CocoRoF/Geny) · [geny-executor-web](https://github.com/CocoRoF/geny-executor-web) · [GAPT](https://github.com/CocoRoF/geny-adapted-project-toolkit) · [geny-avatar](https://github.com/CocoRoF/geny-avatar)

**기반 / 상호운용:**

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Google GenAI SDK](https://github.com/googleapis/python-genai)
- [vLLM](https://github.com/vllm-project/vllm)
- [Claude Code CLI](https://docs.anthropic.com/claude/code/) — geny-executor가 `claude_code_cli` provider로 host
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol; 호스트-attached 서버 + per-session CLI wrap 둘 다 first-class
