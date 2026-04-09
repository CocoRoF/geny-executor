# geny-executor

[![PyPI version](https://img.shields.io/pypi/v/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/geny-executor.svg)](https://pypi.org/project/geny-executor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/CocoRoF/geny-executor/actions/workflows/ci.yml)

**Anthropic API 기반 하네스 엔지니어링 에이전트 파이프라인 라이브러리**

geny-executor는 Claude Code의 에이전트 루프와 Anthropic의 하네스 설계 원칙에서 영감을 받아 설계된 **16단계 파이프라인**과 **이중 추상화(Dual Abstraction) 아키텍처**를 구현합니다. LangChain 없음. LangGraph 없음. 에이전트 실행의 모든 단계를 완전히 제어할 수 있는 깔끔하고 모듈화된 파이프라인입니다.

[English README](README.md)

---

## 왜 geny-executor인가?

| 문제 | geny-executor의 해답 |
|------|---------------------|
| 프레임워크가 너무 많은 것을 추상화 뒤에 숨김 | 모든 단계가 명시적이고 검사 가능 |
| 한 부분을 커스터마이즈하려면 전체를 다시 작성해야 함 | **이중 추상화** — Stage 자체 *또는* Stage 내부의 Strategy를 교체 |
| 에이전트 루프가 불투명한 블랙박스 | 16개의 명확한 단계 + 이벤트 기반 관찰 가능성 |
| 여러 LLM 프로바이더에 걸친 벤더 종속 | 단일 의존성: Anthropic SDK. 하나의 프로바이더를 제대로 |
| 비용 추적이 뒷전 | 내장된 토큰 추적, 비용 계산, 예산 가드 |

---

## 아키텍처

### 16단계 파이프라인

```
Phase A (1회):    [1: Input]
Phase B (루프):   [2: Context] → [3: System] → [4: Guard] → [5: Cache]
                  → [6: API] → [7: Token] → [8: Think] → [9: Parse]
                  → [10: Tool] → [11: Agent] → [12: Evaluate] → [13: Loop]
Phase C (1회):    [14: Emit] → [15: Memory] → [16: Yield]
```

| # | Stage | 목적 | 전략 예시 |
|---|-------|------|----------|
| 1 | **Input** | 사용자 입력 검증 및 정규화 | Default, Strict, Schema, Multimodal |
| 2 | **Context** | 대화 히스토리 및 메모리 로드 | SimpleLoad, ProgressiveDisclosure, VectorSearch |
| 3 | **System** | 시스템 프롬프트 구성 | Static, Composable, Adaptive |
| 4 | **Guard** | 안전 검사 및 예산 제한 | TokenBudget, Cost, RateLimit, Permission |
| 5 | **Cache** | 프롬프트 캐싱 최적화 | NoCache, System, Aggressive, Adaptive |
| 6 | **API** | Anthropic Messages API 호출 | Anthropic, Mock, Recording, Replay |
| 7 | **Token** | 사용량 추적 및 비용 계산 | Default, Detailed + AnthropicPricing |
| 8 | **Think** | 확장 사고(Extended Thinking) 처리 | Passthrough, ExtractAndStore, Filter |
| 9 | **Parse** | 응답 파싱 및 완료 신호 감지 | Default, StructuredOutput + SignalDetector |
| 10 | **Tool** | 도구 호출 실행 | Sequential, Parallel + RegistryRouter |
| 11 | **Agent** | 멀티 에이전트 오케스트레이션 | SingleAgent, Delegate, Evaluator |
| 12 | **Evaluate** | 품질 판단 및 완료 결정 | SignalBased, CriteriaBased, AgentEval |
| 13 | **Loop** | 계속할지 종료할지 결정 | Standard, SingleTurn, BudgetAware |
| 14 | **Emit** | 결과 출력 | Text, Callback, VTuber, TTS, Streaming |
| 15 | **Memory** | 대화 메모리 영속화 | AppendOnly, Reflective + File/SQLite |
| 16 | **Yield** | 최종 결과 포맷팅 | Default, Structured, Streaming |

### 이중 추상화 (Dual Abstraction)

```
┌─ Level 1: Stage 추상화 ──────────────────────────────┐
│  파이프라인에서 Stage 모듈 전체를 교체 가능             │
│                                                       │
│  ┌─ Level 2: Strategy 추상화 ──────────────────────┐  │
│  │  Stage 내부의 로직을 교체 가능                    │  │
│  │                                                  │  │
│  │  ContextStage가 사용할 수 있는 전략:              │  │
│  │    → SimpleLoadStrategy (기본값)                  │  │
│  │    → ProgressiveDisclosureStrategy               │  │
│  │    → VectorSearchStrategy                        │  │
│  │    → 사용자 정의 Strategy                         │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Level 1:** Stage 전체를 교체 (예: `APIStage`를 커스텀 프로바이더로 교체)  
**Level 2:** Stage *내부*의 동작을 변경 (예: 컨텍스트 로딩 전략을 Simple에서 Vector Search로 전환)

---

## 설치

```bash
pip install geny-executor
```

선택적 의존성 포함:

```bash
# 메모리 기능 (벡터 연산을 위한 numpy)
pip install geny-executor[memory]

# 모든 선택적 의존성
pip install geny-executor[all]

# 개발 환경
pip install geny-executor[dev]
```

### 요구사항

- Python 3.11+
- Anthropic API 키

---

## 빠른 시작

### 최소 파이프라인

가장 간단한 에이전트 — 입력, API 호출, 파싱, 출력:

```python
import asyncio
from geny_executor import PipelinePresets

async def main():
    pipeline = PipelinePresets.minimal(api_key="sk-ant-...")
    result = await pipeline.run("프랑스의 수도는 어디인가요?")
    print(result.text)

asyncio.run(main())
```

### 채팅 파이프라인

히스토리, 시스템 프롬프트, 도구 지원이 포함된 대화형 에이전트:

```python
from geny_executor import PipelinePresets

pipeline = PipelinePresets.chat(
    api_key="sk-ant-...",
    system_prompt="당신은 도움이 되는 코딩 어시스턴트입니다.",
)

result = await pipeline.run("Python 데코레이터에 대해 설명해주세요")
print(result.text)
print(f"비용: ${result.total_cost_usd:.4f}")
```

### 에이전트 파이프라인 (16단계 전체)

도구, 평가, 메모리, 루프 제어를 갖춘 자율 에이전트:

```python
from geny_executor import PipelinePresets
from geny_executor.tools import ToolRegistry, Tool, ToolResult, ToolContext

class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "웹에서 정보를 검색합니다"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"}
            },
            "required": ["query"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        # 검색 구현
        return ToolResult(content=f"검색 결과: {input['query']}")

registry = ToolRegistry()
registry.register(SearchTool())

pipeline = PipelinePresets.agent(
    api_key="sk-ant-...",
    system_prompt="당신은 리서치 어시스턴트입니다. 도구를 사용해 답을 찾으세요.",
    tools=registry,
    max_turns=20,
)

result = await pipeline.run("최신 Python 릴리스 버전을 찾아주세요")
```

### 빌더를 이용한 커스텀 파이프라인

모든 단계를 세밀하게 제어:

```python
from geny_executor import PipelineBuilder

pipeline = (
    PipelineBuilder("my-agent", api_key="sk-ant-...")
    .with_model(model="claude-sonnet-4-20250514", max_tokens=4096)
    .with_system(prompt="간결하게 답변하는 어시스턴트입니다.")
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

### 수동 파이프라인 조립

최대한의 제어를 위해 Stage를 직접 조립:

```python
from geny_executor import Pipeline, PipelineConfig
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage

config = PipelineConfig(name="custom", api_key="sk-ant-...")
pipeline = Pipeline(config)

pipeline.register_stage(InputStage())
pipeline.register_stage(APIStage(provider=MockProvider(responses=["안녕하세요!"])))
pipeline.register_stage(ParseStage())
pipeline.register_stage(YieldStage())

result = await pipeline.run("테스트 입력")
```

---

## 세션

여러 인터랙션에 걸친 영속적 상태 관리:

```python
from geny_executor import PipelinePresets
from geny_executor.session import SessionManager

manager = SessionManager()
pipeline = PipelinePresets.chat(api_key="sk-ant-...")

# 세션 생성 — 실행 간 상태 유지
session = manager.create(pipeline)
result1 = await session.run("제 이름은 Alice입니다")
result2 = await session.run("제 이름이 뭐였죠?")  # 컨텍스트 기억

# 활성 세션 목록
for info in manager.list_sessions():
    print(f"세션 {info.session_id}: {info.message_count}개 메시지, ${info.total_cost_usd:.4f}")
```

---

## 이벤트 시스템

pub/sub 기반 실시간 관찰 가능성:

```python
from geny_executor import PipelinePresets

pipeline = PipelinePresets.agent(api_key="sk-ant-...")

# 특정 이벤트 구독
@pipeline.on("stage.enter")
async def on_stage_enter(event):
    print(f"  → 진입: {event.stage}")

@pipeline.on("stage.exit")
async def on_stage_exit(event):
    print(f"  ← 종료: {event.stage}")

@pipeline.on("pipeline.complete")
async def on_complete(event):
    print(f"완료! 반복 횟수: {event.data.get('iterations')}")

# 와일드카드: 모든 이벤트 수신
@pipeline.on("*")
async def on_any(event):
    pass  # 모든 것을 로깅

result = await pipeline.run("안녕하세요")
```

### 스트리밍

```python
async for event in pipeline.run_stream("단계별로 풀어주세요"):
    if event.type == "stage.enter":
        print(f"Stage: {event.stage}")
    elif event.type == "api.response":
        print(f"응답 수신")
    elif event.type == "pipeline.complete":
        print(f"최종: {event.data['result'].text}")
```

---

## 도구 시스템

### 도구 만들기

```python
from geny_executor.tools import Tool, ToolResult, ToolContext

class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "산술 계산을 수행합니다"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "계산할 수학 표현식"}
            },
            "required": ["expression"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        try:
            result = eval(input["expression"])  # 프로덕션에서는 안전한 평가기를 사용하세요
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=str(e), is_error=True)
```

### 도구 레지스트리

```python
from geny_executor.tools import ToolRegistry

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(SearchTool())

# 요청별 도구 필터링
math_tools = registry.filter(include=["calculator"])
api_format = registry.to_api_format()  # Anthropic API 포맷
```

### MCP 통합

Model Context Protocol 서버 연결:

```python
from geny_executor.tools.mcp import MCPManager

mcp = MCPManager()
await mcp.connect("filesystem", command="npx", args=["-y", "@anthropic/mcp-filesystem"])

# MCP 도구는 자동으로 Tool 인터페이스에 맞게 어댑트됨
for tool in mcp.list_tools():
    registry.register(tool)
```

---

## 에러 처리

자동 분류가 포함된 구조화된 에러 계층:

```python
from geny_executor import (
    GenyExecutorError,   # 기본 예외
    PipelineError,       # 파이프라인 수준 에러
    StageError,          # Stage 실행 에러
    GuardRejectError,    # Guard 거부 (예산, 권한)
    APIError,            # Anthropic API 에러 (카테고리 포함)
    ToolExecutionError,  # 도구 실행 실패
    ErrorCategory,       # rate_limited, timeout, token_limit 등
)

try:
    result = await pipeline.run("입력")
except GuardRejectError as e:
    print(f"Guard에 의해 차단됨: {e}")
except APIError as e:
    print(f"API 에러 ({e.category}): {e}")
    if e.category == ErrorCategory.rate_limited:
        # Rate limiting 처리
        pass
except GenyExecutorError as e:
    print(f"파이프라인 에러: {e}")
```

---

## 파이프라인 프리셋

바로 사용할 수 있는 5가지 구성:

| 프리셋 | Stage | 용도 |
|--------|-------|------|
| `PipelinePresets.minimal()` | 1→6→9→16 | 단순 Q&A, 테스트 |
| `PipelinePresets.chat()` | 1→2→3→4→5→6→7→9→13→16 | 대화형 챗봇 |
| `PipelinePresets.agent()` | 전체 16개 | 도구를 사용하는 자율 에이전트 |
| `PipelinePresets.evaluator()` | 1→3→6→9→12→16 | 품질 평가 |
| `PipelinePresets.geny_vtuber()` | 전체 16개 + VTuber emit | Geny VTuber 시스템 |

---

## 커스텀 Stage & Strategy

### 커스텀 Strategy 만들기

```python
from geny_executor.core.stage import Strategy

class MyContextStrategy(Strategy):
    @property
    def name(self) -> str:
        return "my_context"

    @property
    def description(self) -> str:
        return "RAG를 이용한 커스텀 컨텍스트 로딩"

    def configure(self, config: dict) -> None:
        self.top_k = config.get("top_k", 5)

    async def load(self, state):
        # 커스텀 컨텍스트 로딩 로직
        ...
```

### 커스텀 Stage 만들기

```python
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState

class LoggingStage(Stage[dict, dict]):
    @property
    def name(self) -> str:
        return "logging"

    @property
    def order(self) -> int:
        return 7  # API 이후, Think 이전

    @property
    def category(self) -> str:
        return "execution"

    async def execute(self, input: dict, state: PipelineState) -> dict:
        print(f"[{state.iteration}] API 응답 수신")
        return input  # 통과

# 파이프라인에 등록
pipeline.register_stage(LoggingStage())
```

---

## 설정 참조

### ModelConfig

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Anthropic 모델 ID |
| `max_tokens` | `int` | `8192` | 최대 출력 토큰 수 |
| `temperature` | `float` | `0.0` | 샘플링 온도 |
| `thinking_enabled` | `bool` | `False` | 확장 사고 활성화 |
| `thinking_budget_tokens` | `int` | `10000` | 사고 토큰 예산 |

### PipelineConfig

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `name` | `str` | 필수 | 파이프라인 이름 |
| `api_key` | `str` | 필수 | Anthropic API 키 |
| `model` | `ModelConfig` | 기본값 | 모델 설정 |
| `max_iterations` | `int` | `50` | 최대 루프 반복 횟수 |
| `cost_budget_usd` | `float?` | `None` | 비용 예산 한도 |
| `context_window_budget` | `int` | `200_000` | 컨텍스트 윈도우 토큰 제한 |
| `stream` | `bool` | `False` | 스트리밍 모드 활성화 |
| `single_turn` | `bool` | `False` | 단일 턴 (루프 없음) |

---

## 프로젝트 구조

```
geny-executor/
├── src/geny_executor/
│   ├── __init__.py          # Public API
│   ├── py.typed             # PEP 561 타입 마커
│   ├── core/                # 파이프라인 엔진, 설정, 상태, 에러
│   ├── events/              # EventBus pub/sub 시스템
│   ├── stages/              # 16개 파이프라인 Stage (s01-s16)
│   ├── tools/               # 도구 시스템 + MCP 통합
│   └── session/             # 세션 관리 + 신선도 정책
├── tests/                   # 81개 유닛 및 통합 테스트
├── pyproject.toml           # 패키지 설정 (Hatch)
└── LICENSE                  # MIT 라이선스
```

---

## 개발

```bash
# 클론
git clone https://github.com/CocoRoF/geny-executor.git
cd geny-executor

# 개발 의존성과 함께 설치
pip install -e ".[dev]"

# 테스트 실행
pytest

# 커버리지 포함 테스트
pytest --cov=geny_executor --cov-report=term-missing

# 린트
ruff check src/ tests/
ruff format src/ tests/
```

---

## 로드맵

- [ ] 스트리밍 응답 지원 (완전한 구현)
- [ ] OpenTelemetry 트레이싱 통합
- [ ] 추가 메모리 백엔드 (Redis, PostgreSQL)
- [ ] WebUI 파이프라인 설정기
- [ ] 커뮤니티 Stage를 위한 플러그인 시스템
- [ ] 배치 실행 모드

---

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다 — 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 관련 프로젝트

- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — geny-executor가 구축된 기반
- [MCP](https://modelcontextprotocol.io/) — 도구 통합을 위한 Model Context Protocol
- [Claude Code](https://claude.ai/code) — 이 아키텍처에 영감을 준 에이전트 루프
