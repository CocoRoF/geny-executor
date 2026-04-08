# geny-executor: Harness-Engineered Agent Pipeline Library

> **Version**: 0.2.0-draft  
> **Date**: 2026-04-08  
> **Status**: Planning Phase (2nd Iteration)  
> **Author**: Geny Team

---

## 1. Executive Summary

`geny-executor`는 Anthropic API를 직접 사용하는 **harness-engineered agent pipeline library**이다.  
Claude Code의 Agent Loop에서 영감을 받되, Geny VTuber 시스템의 철학과 요구사항을 반영한 **범용 에이전트 실행 파이프라인**을 구축한다.

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **No Framework Dependency** | LangChain, LangGraph 없음. Anthropic SDK + 자체 파이프라인만 사용 |
| **Harness as Architecture** | 파이프라인의 각 단계가 곧 아키텍처. 모든 실행은 반드시 파이프라인을 통과 |
| **Dual Abstraction** | Stage 자체의 추상화 + Stage 내부 로직의 추상화. 두 레벨 모두 교체 가능 |
| **Single Pipeline Ownership** | 하나의 파이프라인이 모든 Harness를 관장. 별도 세션은 독립 작업 분리 시에만 |
| **Modular Bypass** | 각 단계는 독립 모듈. 불필요 시 bypass 가능하나 구조적 위치는 항상 존재 |
| **Interface-First** | 강력한 인터페이스가 구현보다 선행. 확장은 인터페이스를 통해서만 |
| **Geny-Compatible** | 기존 Geny의 모든 기능을 이 파이프라인 위에서 재현 가능 |
| **WebUI-Ready** | Pipeline UI — 단계는 고정, 개별 Stage의 구현체를 교체하는 형식 |

### 핵심 설계 철학: Dual Abstraction

```
┌─────────────────────────────────────────────────────────────┐
│  Level 1: Stage Abstraction (파이프라인 수준)                 │
│  ─────────────────────────────────────────                   │
│  Stage 인터페이스를 구현한 모듈이 파이프라인 슬롯에 장착됨.      │
│  Stage 자체를 통째로 교체 가능.                                │
│                                                             │
│  Level 2: Strategy Abstraction (Stage 내부 수준)             │
│  ─────────────────────────────────────────                   │
│  각 Stage 내부의 핵심 로직이 Strategy 인터페이스로 추상화됨.     │
│  동일 Stage라도 내부 전략을 교체하면 완전히 다른 동작 수행.      │
│                                                             │
│  예시:                                                       │
│  ┌─ ContextStage (Level 1) ─────────────────────────┐       │
│  │                                                   │       │
│  │  ┌─ ContextStrategy (Level 2) ──────────────┐    │       │
│  │  │  impl A: SimpleLoadStrategy              │    │       │
│  │  │  impl B: ProgressiveDisclosureStrategy   │    │       │
│  │  │  impl C: VectorSearchStrategy            │    │       │
│  │  │  impl D: CustomStrategy (user-defined)   │    │       │
│  │  └──────────────────────────────────────────┘    │       │
│  │                                                   │       │
│  │  ┌─ HistoryCompactor (Level 2) ─────────────┐    │       │
│  │  │  impl A: TruncateCompactor               │    │       │
│  │  │  impl B: SummaryCompactor                │    │       │
│  │  │  impl C: SlidingWindowCompactor          │    │       │
│  │  └──────────────────────────────────────────┘    │       │
│  └───────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Reference Analysis

### 2.1 Claude Code Agent Loop (11 Stages)

```
1.Input → 2.Message → 3.History → 4.System → 5.API → 6.Tokens → 7.Tools? → 8.Loop → 9.Render → 10.Hooks → 11.Await
```

### 2.2 Anthropic의 Harness Design 원칙

1. **Generator/Evaluator 분리 (GAN-inspired)**: 생성과 평가를 독립 에이전트로 분리
2. **주관적 품질 → 측정 가능한 기준 변환**: 구체적 평가 기준 정의
3. **가장 단순한 솔루션 우선**: 복잡도는 필요할 때만 증가
4. **실험 → 트레이스 읽기 → 튜닝**: 반복적 개선 루프
5. **Context 관리는 필수**: 장시간 실행 시 압축/리셋 전략 필수
6. **Sprint Contract 패턴**: 작업 전 완료 기준 합의

### 2.3 OpenAI의 Harness Engineering 원칙

1. **Repository = System of Record**: 레포에 없으면 존재하지 않음
2. **Map, not Manual**: 진입점은 작게, 점진적 상세화
3. **Enforce Invariants, not Implementations**: 규칙은 코드로 강제
4. **Rigid Layering**: Types → Config → Service → Runtime → UI
5. **Boring Technology**: 안정적이고 학습 데이터에 풍부한 기술 선택
6. **Isolation per Task**: 작업별 독립 환경

### 2.4 기존 Geny에서 채용할 핵심 패턴

| 패턴 | 출처 | geny-executor 적용 |
|------|------|-------------------|
| **Single Execution Path** | `agent_executor.py` | 모든 실행이 Pipeline.run()을 통과 |
| **State + Reducers** | `state.py` | PipelineState + reducer 함수로 상태 누적 |
| **Guard/Post Node 패턴** | `autonomous_graph.py` | 각 Stage에 on_enter/on_exit 훅 |
| **Completion Signal Protocol** | `resilience_nodes.py` | `[CONTINUE]`, `[COMPLETE]`, `[BLOCKED]` 유지 |
| **Memory Deduplication** | `state.py` | MemoryStage에서 동일 로직 |
| **Freshness/Auto-Revival** | `session_freshness.py` | SessionManager에서 동일 정책 |
| **Fire-and-Forget + SSE** | `controller/` | EventBus 기반 실시간 상태 전파 |
| **Error Classification** | `model_fallback.py` | ErrorClassifier (terminal/recoverable) |
| **Tool Preset System** | `tools/tool_loader.py` | ToolRegistry 프리셋 기반 필터링 |

---

## 3. The 16-Stage Pipeline

### 3.1 Stage Overview

Claude Code 11단계 + Geny 철학 반영 5단계 = **16단계 파이프라인**

추가된 5단계:
- **Cache** (Stage 5): Prompt Caching 전략 관리 — 비용 최적화의 핵심
- **Think** (Stage 8): Extended Thinking 처리 — 추론 과정의 독립적 관리
- **Agent** (Stage 11): Multi-Agent 오케스트레이션 — 단일 파이프라인 내 에이전트 위임
- **Evaluate** (Stage 12): 응답 품질 평가 — Generator/Evaluator 분리 패턴
- **Memory** (Stage 15): 메모리 업데이트 — Geny의 지식 관리 체계

```
                            ┌─────────────────────────────────────────────────┐
                            │           geny-executor Pipeline (16 Stages)     │
                            │                                                 │
 ┌───────┐  ┌─────────┐  ┌─┴───────┐  ┌────────┐  ┌───────┐  ┌───────┐     │
 │   1   │→│    2    │→│    3    │→│   4    │→│   5   │→│   6   │     │
 │ Input │  │ Context │  │ System │  │ Guard  │  │ Cache │  │  API  │     │
 │       │  │         │  │        │  │        │  │       │  │       │     │
 └───────┘  └─────────┘  └────────┘  └────────┘  └───────┘  └───┬───┘     │
                                                                  │         │
 ┌───────┐  ┌─────────┐  ┌────────┐  ┌────────┐  ┌───────┐  ┌───┴───┐     │
 │  12   │←│   11    │←│   10   │←│    9   │←│   8   │←│   7   │     │
 │Evaluate│ │  Agent  │  │  Tool  │  │ Parse  │  │ Think │  │ Token │     │
 │       │  │         │  │        │  │        │  │       │  │       │     │
 └──┬────┘  └─────────┘  └────────┘  └────────┘  └───────┘  └───────┘     │
    │                                                                       │
 ┌──┴────┐  ┌─────────┐  ┌────────┐  ┌────────┐                            │
 │  13   │→│   14    │→│   15   │→│   16   │                            │
 │ Loop  │  │  Emit   │  │ Memory │  │ Yield  │                            │
 │       │  │         │  │        │  │        │                            │
 └───────┘  └─────────┘  └────────┘  └────────┘                            │
    │ ↓(continue)                                                           │
    └→ Stage 2 (Context) 재진입 ────────────────────────────────────────────┘

 ※ Stage 11 (Agent) → 새로운 Pipeline 세션 위임 가능 (별도 harness)
```

### 3.2 Stage 분류 체계

```
┌─ Ingress ──────────────────────────────────────────────┐
│  Stage  1: Input      사용자 입력 수신 및 정규화          │
│  Stage  2: Context    컨텍스트 수집 (히스토리, 메모리)     │
│  Stage  3: System     시스템 프롬프트 조립                │
└────────────────────────────────────────────────────────┘

┌─ Pre-Flight ───────────────────────────────────────────┐
│  Stage  4: Guard      안전장치 & 사전 검증               │
│  Stage  5: Cache      Prompt Caching 전략 적용           │
└────────────────────────────────────────────────────────┘

┌─ Execution ────────────────────────────────────────────┐
│  Stage  6: API        Anthropic Messages API 호출       │
│  Stage  7: Token      토큰 사용량 추적 & 예산 관리        │
│  Stage  8: Think      Extended Thinking 처리            │
│  Stage  9: Parse      응답 파싱 (text, tool_use)         │
│  Stage 10: Tool       도구 호출 실행                     │
│  Stage 11: Agent      Multi-Agent 오케스트레이션          │
└────────────────────────────────────────────────────────┘

┌─ Decision ─────────────────────────────────────────────┐
│  Stage 12: Evaluate   응답 품질 평가 & 완료 판단          │
│  Stage 13: Loop       에이전트 루프 제어                  │
└────────────────────────────────────────────────────────┘

┌─ Egress ───────────────────────────────────────────────┐
│  Stage 14: Emit       결과 출력 (스트리밍, VTuber, TTS)  │
│  Stage 15: Memory     메모리 업데이트 & 정리              │
│  Stage 16: Yield      최종 결과 반환                     │
└────────────────────────────────────────────────────────┘
```

---

## 4. Stage 상세 설계 — Dual Abstraction

모든 Stage는 두 레벨의 추상화를 갖는다:
- **Level 1**: Stage 인터페이스 (파이프라인 슬롯에 장착되는 단위)
- **Level 2**: Strategy 인터페이스 (Stage 내부 핵심 로직의 교체 가능한 전략)

### Stage 1: Input

```
분류: Ingress
역할: 사용자 입력 수신, 검증, 정규화

Level 1 — Stage 인터페이스:
  입력: Raw user input (Any)
  출력: NormalizedInput

Level 2 — Strategy 인터페이스:

  ┌─ InputValidator ──────────────────────────────────────┐
  │  역할: 입력 검증                                        │
  │  impl: DefaultValidator     — 길이, 인코딩 체크          │
  │  impl: StrictValidator      — 금지어, 보안 패턴 검사      │
  │  impl: PassthroughValidator — 검증 없이 통과             │
  │  impl: SchemaValidator      — JSON Schema 기반 검증      │
  └──────────────────────────────────────────────────────┘

  ┌─ InputNormalizer ─────────────────────────────────────┐
  │  역할: 입력 전처리 및 정규화                              │
  │  impl: DefaultNormalizer    — 트리밍, 유니코드 정규화     │
  │  impl: MultimodalNormalizer — 이미지/파일 첨부 처리       │
  │  impl: CommandNormalizer    — 슬래시 커맨드 파싱          │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 없음 (항상 실행)
```

### Stage 2: Context

```
분류: Ingress
역할: 실행에 필요한 컨텍스트 수집 및 조립

Level 1 — Stage 인터페이스:
  입력: NormalizedInput + PipelineState
  출력: ExecutionContext (히스토리 + 메모리 + 참조문서)

Level 2 — Strategy 인터페이스:

  ┌─ ContextStrategy ─────────────────────────────────────┐
  │  역할: 컨텍스트 수집 전략 (어떤 정보를 얼마나 가져올지)     │
  │                                                       │
  │  impl: SimpleLoadStrategy                             │
  │    — 전체 히스토리 + 최근 메모리 단순 로딩                 │
  │    — 짧은 대화, 단순 질의에 적합                          │
  │                                                       │
  │  impl: ProgressiveDisclosureStrategy                  │
  │    — OpenAI 방식: 작은 진입점 → 필요 시 점진적 상세화      │
  │    — 먼저 요약 로딩 → 관련 부분만 상세 로딩                │
  │    — 긴 대화, 복잡한 작업에 적합                          │
  │                                                       │
  │  impl: VectorSearchStrategy                           │
  │    — 입력과 의미적으로 관련된 컨텍스트만 선별 로딩          │
  │    — 방대한 메모리 풀에서 관련 정보 추출 시 적합            │
  │                                                       │
  │  impl: HybridStrategy                                 │
  │    — 최근 N턴 히스토리 + 벡터 검색 메모리 + 고정 참조      │
  │    — Geny의 기본 전략                                   │
  └──────────────────────────────────────────────────────┘

  ┌─ HistoryCompactor ────────────────────────────────────┐
  │  역할: 히스토리가 토큰 예산을 초과할 때 압축 전략          │
  │                                                       │
  │  impl: TruncateCompactor                              │
  │    — 오래된 메시지부터 잘라냄 (가장 단순)                  │
  │                                                       │
  │  impl: SummaryCompactor                               │
  │    — 오래된 대화를 요약으로 교체 (API 호출 필요)           │
  │    — Anthropic이 권장하는 방식                           │
  │                                                       │
  │  impl: SlidingWindowCompactor                         │
  │    — 고정 윈도우 크기 유지, 초과분 요약                    │
  │                                                       │
  │  impl: ResetCompactor                                 │
  │    — 전체 컨텍스트 리셋 + 구조화된 핸드오프                │
  │    — Anthropic 블로그의 context reset 패턴               │
  └──────────────────────────────────────────────────────┘

  ┌─ MemoryRetriever ─────────────────────────────────────┐
  │  역할: 메모리 저장소에서 관련 정보 조회                    │
  │                                                       │
  │  impl: FileMemoryRetriever     — 파일 기반 메모리 조회   │
  │  impl: VectorMemoryRetriever   — 벡터 유사도 검색        │
  │  impl: SQLiteMemoryRetriever   — SQLite 쿼리 기반       │
  │  impl: CompositeRetriever      — 여러 소스 병합          │
  └──────────────────────────────────────────────────────┘

Bypass 조건: stateless 모드 (단일 질의, 히스토리 불필요)
```

### Stage 3: System

```
분류: Ingress
역할: 시스템 프롬프트 최종 조립

Level 1 — Stage 인터페이스:
  입력: ExecutionContext + PipelineConfig
  출력: SystemPrompt (string)

Level 2 — Strategy 인터페이스:

  ┌─ PromptBuilder ───────────────────────────────────────┐
  │  역할: 시스템 프롬프트 구성 전략                          │
  │                                                       │
  │  impl: TemplatePromptBuilder                          │
  │    — Jinja2/string template 기반 조립                   │
  │    — 페르소나 + 규칙 + 도구설명 + 컨텍스트 삽입            │
  │                                                       │
  │  impl: ComposablePromptBuilder                        │
  │    — 프롬프트를 블록 단위로 조합 (순서/포함여부 설정)       │
  │    — blocks: [persona, rules, tools, memory, datetime] │
  │    — 블록별 cache_control 지정 가능                      │
  │                                                       │
  │  impl: StaticPromptBuilder                            │
  │    — 외부에서 완성된 프롬프트를 그대로 전달                 │
  │                                                       │
  │  impl: AdaptivePromptBuilder                          │
  │    — 대화 상태에 따라 프롬프트 동적 조정                   │
  │    — 난이도 분류 결과, 현재 작업 등에 반응                 │
  └──────────────────────────────────────────────────────┘

  ┌─ ToolDescriptionFormatter ────────────────────────────┐
  │  역할: 등록된 도구의 설명을 시스템 프롬프트에 삽입하는 방식  │
  │                                                       │
  │  impl: APIToolFormat   — tools 파라미터로 전달 (기본)    │
  │  impl: InlineFormat    — 시스템 프롬프트 내 텍스트로 삽입  │
  │  impl: HybridFormat    — 핵심 도구는 API, 보조는 inline  │
  └──────────────────────────────────────────────────────┘

Bypass 조건: system prompt가 외부에서 완성된 채로 전달될 때
```

### Stage 4: Guard

```
분류: Pre-Flight
역할: 실행 전 안전장치 & 사전 검증

Level 1 — Stage 인터페이스:
  입력: SystemPrompt + ExecutionContext + PipelineState
  출력: GuardResult (pass / reject / modify)

Level 2 — Strategy 인터페이스:

  ┌─ GuardChain ──────────────────────────────────────────┐
  │  역할: 여러 Guard를 체인으로 연결, 모두 통과해야 진행      │
  │                                                       │
  │  guard: TokenBudgetGuard                              │
  │    — 남은 컨텍스트 윈도우 확인, 부족 시 compaction 트리거  │
  │                                                       │
  │  guard: CostBudgetGuard                               │
  │    — 누적 비용 vs 세션 예산 확인                         │
  │                                                       │
  │  guard: RateLimitGuard                                │
  │    — API rate limit 사전 확인                           │
  │                                                       │
  │  guard: IterationGuard                                │
  │    — 루프 반복 횟수 확인 (무한 루프 방지)                  │
  │                                                       │
  │  guard: PermissionGuard                               │
  │    — 도구 사용 권한 검증                                 │
  │                                                       │
  │  guard: RelevanceGuard                                │
  │    — Geny의 relevance_gate: 채팅이 현재 맥락과 관련?     │
  │                                                       │
  │  guard: ContentSafetyGuard                            │
  │    — 입력 내용 안전성 검사                               │
  │                                                       │
  │  ※ Guard는 List로 등록. 순서대로 실행. 하나라도 reject면 중단│
  │  ※ 커스텀 Guard 추가 자유                                │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 신뢰된 내부 호출 (guard_bypass=True)
```

### Stage 5: Cache

```
분류: Pre-Flight
역할: Anthropic Prompt Caching 전략 관리

Level 1 — Stage 인터페이스:
  입력: SystemPrompt + Messages + Tools
  출력: CacheOptimizedRequest (cache_control 마커가 삽입된 요청)

Level 2 — Strategy 인터페이스:

  ┌─ CacheStrategy ───────────────────────────────────────┐
  │  역할: 어디에 cache breakpoint를 삽입할지 결정             │
  │                                                       │
  │  impl: NoCacheStrategy                                │
  │    — 캐싱 비활성화 (cache_control 없음)                  │
  │                                                       │
  │  impl: SystemCacheStrategy                            │
  │    — 시스템 프롬프트에만 cache_control: ephemeral 적용    │
  │    — 가장 기본적. 시스템 프롬프트가 변하지 않으면 유효      │
  │                                                       │
  │  impl: AggressiveCacheStrategy                        │
  │    — system + tools + 히스토리의 안정적 부분에 모두 적용   │
  │    — 비용 절감 극대화, 다회 턴 대화에서 효과적              │
  │    — cache breakpoint 위치:                             │
  │      1. system prompt 끝                               │
  │      2. tools 정의 끝                                   │
  │      3. 히스토리의 마지막 안정 지점                       │
  │                                                       │
  │  impl: AdaptiveCacheStrategy                          │
  │    — 이전 턴의 cache_read_tokens 기반으로 전략 자동 조정   │
  │    — 캐시 히트율이 낮으면 breakpoint 재배치               │
  │                                                       │
  │  impl: ManualCacheStrategy                            │
  │    — 사용자가 직접 cache breakpoint 위치 지정             │
  └──────────────────────────────────────────────────────┘

  ┌─ CacheAnalyzer ───────────────────────────────────────┐
  │  역할: 캐시 효율 분석 (Stage 7 Token에서 피드백 수신)      │
  │                                                       │
  │  metrics: cache_hit_rate, cache_creation_cost,         │
  │           estimated_savings_usd                        │
  │                                                       │
  │  ※ 분석 결과는 PipelineState에 기록                      │
  │  ※ AdaptiveCacheStrategy가 이 데이터를 소비              │
  └──────────────────────────────────────────────────────┘

Bypass 조건: NoCacheStrategy 선택 시 (마커 삽입 없이 통과)

API 연동 방식:
  Anthropic API의 cache_control은 메시지 content block에 삽입:
  {
    "type": "text",
    "text": "...",
    "cache_control": {"type": "ephemeral"}
  }
  이 Stage에서 적절한 위치에 cache_control을 삽입한 후 Stage 6(API)에 전달.
```

### Stage 6: API

```
분류: Execution
역할: Anthropic Messages API 호출

Level 1 — Stage 인터페이스:
  입력: CacheOptimizedRequest (messages, system, tools, config)
  출력: RawAPIResponse

Level 2 — Strategy 인터페이스:

  ┌─ APIProvider ─────────────────────────────────────────┐
  │  역할: 실제 API 호출 수행                                │
  │                                                       │
  │  impl: AnthropicProvider                              │
  │    — Anthropic SDK 사용, 실제 API 호출                  │
  │    — Streaming / Non-streaming 모드 지원                │
  │                                                       │
  │  impl: MockProvider                                   │
  │    — 테스트용 목 응답 반환                               │
  │    — 시나리오별 fixture 로딩 가능                        │
  │                                                       │
  │  impl: RecordingProvider                              │
  │    — 실제 API 호출 + 요청/응답 기록 (테스트 fixture 생성) │
  │                                                       │
  │  impl: ReplayProvider                                 │
  │    — 기록된 요청/응답 재생 (결정론적 테스트)               │
  │                                                       │
  │  impl: FallbackProvider                               │
  │    — Primary 실패 시 Secondary 모델로 자동 전환           │
  │    — e.g., opus → sonnet fallback                     │
  └──────────────────────────────────────────────────────┘

  ┌─ RetryStrategy ───────────────────────────────────────┐
  │  역할: API 호출 실패 시 재시도 전략                       │
  │                                                       │
  │  impl: ExponentialBackoffRetry                        │
  │    — 기본 전략. 2^n * base_delay + jitter               │
  │                                                       │
  │  impl: RateLimitAwareRetry                            │
  │    — retry-after 헤더 기반 대기                         │
  │                                                       │
  │  impl: NoRetry                                        │
  │    — 재시도 없음 (테스트, 빠른 실패 원할 때)              │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 없음 (파이프라인의 핵심)
```

### Stage 7: Token

```
분류: Execution
역할: 토큰 사용량 추적, 비용 계산, 예산 관리

Level 1 — Stage 인터페이스:
  입력: RawAPIResponse
  출력: TokenAnalysis (usage 정보가 첨부된 응답)

Level 2 — Strategy 인터페이스:

  ┌─ TokenTracker ─────────────────────────────────────────┐
  │  역할: 토큰 사용량 기록 및 누적                           │
  │                                                        │
  │  impl: DefaultTracker                                  │
  │    — API response.usage에서 직접 추출                    │
  │    — input_tokens, output_tokens,                      │
  │      cache_creation_input_tokens, cache_read_input_tokens│
  │                                                        │
  │  impl: DetailedTracker                                 │
  │    — 턴별, Stage별, 도구별 토큰 사용량 세분화 추적         │
  └────────────────────────────────────────────────────────┘

  ┌─ CostCalculator ──────────────────────────────────────┐
  │  역할: 모델별 토큰 가격 기반 비용 계산                    │
  │                                                       │
  │  impl: AnthropicPricingCalculator                     │
  │    — 모델별 input/output 단가 테이블                    │
  │    — cache_write / cache_read 할인율 반영               │
  │    — 자동 업데이트 가능한 pricing config                 │
  │                                                       │
  │  impl: CustomPricingCalculator                        │
  │    — 사용자 정의 단가 (proxy/자체 호스팅 등)              │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 비용 추적 비활성화 시 (tracker만 skip, 응답은 통과)
```

### Stage 8: Think

```
분류: Execution
역할: Extended Thinking (추론 과정) 독립 관리

Level 1 — Stage 인터페이스:
  입력: TokenAnalysis (API 응답 포함)
  출력: ThinkingResult (thinking blocks + response 분리)

Level 2 — Strategy 인터페이스:

  ┌─ ThinkingProcessor ───────────────────────────────────┐
  │  역할: thinking content blocks 처리 방식                │
  │                                                       │
  │  impl: PassthroughProcessor                           │
  │    — thinking blocks를 그대로 보존 (분리만)              │
  │    — thinking이 없으면 아무 처리 없이 통과               │
  │                                                       │
  │  impl: ExtractAndStoreProcessor                       │
  │    — thinking 내용을 추출하여 별도 저장                   │
  │    — 추론 과정 로깅, 디버깅, 분석에 활용                  │
  │    — PipelineState.thinking_history에 누적              │
  │                                                       │
  │  impl: ThinkingSummaryProcessor                       │
  │    — thinking 내용을 요약하여 컨텍스트에 재주입            │
  │    — 긴 추론 과정을 압축하여 효율적 활용                  │
  │                                                       │
  │  impl: ThinkingFilterProcessor                        │
  │    — 특정 패턴의 thinking만 보존/제거                    │
  │    — 보안상 민감한 추론 과정 필터링                       │
  └──────────────────────────────────────────────────────┘

  ┌─ ThinkingBudget ──────────────────────────────────────┐
  │  역할: Extended Thinking의 토큰 예산 관리                │
  │                                                       │
  │  budget_tokens: int       — thinking에 할당할 최대 토큰  │
  │  enabled: bool            — extended thinking 활성화    │
  │                                                       │
  │  ※ Stage 6(API) 호출 시 thinking 파라미터로 전달:        │
  │    {"type": "enabled", "budget_tokens": 10000}         │
  └──────────────────────────────────────────────────────┘

Bypass 조건: Extended Thinking 비활성화 시 (thinking blocks 없으면 자동 통과)

연동 방식:
  API 응답의 content blocks에서 type="thinking" 블록을 분리.
  thinking blocks → ThinkingProcessor로 처리
  나머지 blocks (text, tool_use) → Stage 9 (Parse)로 전달
```

### Stage 9: Parse

```
분류: Execution
역할: API 응답을 구조화된 형태로 파싱

Level 1 — Stage 인터페이스:
  입력: ThinkingResult (thinking 분리 후 응답)
  출력: ParsedResponse

Level 2 — Strategy 인터페이스:

  ┌─ ResponseParser ──────────────────────────────────────┐
  │  역할: API 응답 content blocks 파싱                     │
  │                                                       │
  │  impl: DefaultParser                                  │
  │    — TextBlock → text 추출                             │
  │    — ToolUseBlock → tool_name, input, id 추출          │
  │    — stop_reason 분류                                  │
  │                                                       │
  │  impl: StructuredOutputParser                         │
  │    — JSON/XML 구조화 출력 파싱 + 검증                   │
  │    — Pydantic 모델 기반 자동 변환                       │
  │                                                       │
  │  impl: StreamingParser                                │
  │    — 스트리밍 청크를 점진적으로 파싱                      │
  │    — 부분 텍스트/도구 호출 실시간 방출                    │
  └──────────────────────────────────────────────────────┘

  ┌─ CompletionSignalDetector ────────────────────────────┐
  │  역할: 텍스트에서 완료 신호 감지                          │
  │                                                       │
  │  signals:                                             │
  │    [CONTINUE: next_action]  → keep looping             │
  │    [TASK_COMPLETE]          → task finished             │
  │    [BLOCKED: reason]        → cannot proceed           │
  │    [ERROR: description]     → fatal error              │
  │    [DELEGATE: agent_type]   → delegate to sub-agent    │
  │                                                       │
  │  impl: RegexDetector        — 정규식 기반 감지          │
  │  impl: StructuredDetector   — JSON 기반 구조화 감지     │
  │  impl: HybridDetector       — regex + JSON 복합        │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 없음 (항상 실행)
```

### Stage 10: Tool

```
분류: Execution
역할: 도구 호출 실행 및 결과 수집

Level 1 — Stage 인터페이스:
  입력: ParsedResponse
  출력: ToolExecutionResult (tool_result 메시지 포함)

Level 2 — Strategy 인터페이스:

  ┌─ ToolExecutor ────────────────────────────────────────┐
  │  역할: 도구 실행 방식                                    │
  │                                                       │
  │  impl: SequentialExecutor                             │
  │    — 도구를 순서대로 하나씩 실행                          │
  │                                                       │
  │  impl: ParallelExecutor                               │
  │    — 독립적인 도구를 asyncio.gather로 병렬 실행           │
  │    — 종속성 있는 도구는 순차 실행                         │
  │                                                       │
  │  impl: SandboxedExecutor                              │
  │    — 격리된 환경에서 실행 (보안 강화)                     │
  │    — 위험한 도구(bash, file_write)에 적용                │
  └──────────────────────────────────────────────────────┘

  ┌─ ToolRouter ──────────────────────────────────────────┐
  │  역할: tool_name으로 실행 대상을 라우팅                   │
  │                                                       │
  │  impl: RegistryRouter                                 │
  │    — ToolRegistry에서 이름으로 조회 후 실행               │
  │                                                       │
  │  impl: MCPRouter                                      │
  │    — MCP 서버에 라우팅 (MCPManager 경유)                 │
  │                                                       │
  │  impl: CompositeRouter                                │
  │    — Registry 먼저, 없으면 MCP로 fallback                │
  │    — Geny의 기본 라우터                                 │
  └──────────────────────────────────────────────────────┘

  ┌─ ToolPermission ──────────────────────────────────────┐
  │  역할: 도구 실행 전 권한 확인                             │
  │                                                       │
  │  impl: AllowAllPermission      — 모든 도구 허용         │
  │  impl: AllowListPermission     — 허용 목록 기반          │
  │  impl: InteractivePermission   — 사용자 승인 요청        │
  │  impl: PresetPermission        — 프리셋 기반 필터링      │
  └──────────────────────────────────────────────────────┘

Bypass 조건: tool_use 블록이 없을 때 (자동 bypass)
```

### Stage 11: Agent

```
분류: Execution
역할: Multi-Agent 오케스트레이션 — 단일 파이프라인 내 에이전트 위임

Level 1 — Stage 인터페이스:
  입력: ToolExecutionResult + ParsedResponse
  출력: AgentResult

Level 2 — Strategy 인터페이스:

  ┌─ AgentOrchestrator ───────────────────────────────────┐
  │  역할: 멀티 에이전트 오케스트레이션 방식 결정              │
  │                                                       │
  │  impl: SingleAgentOrchestrator                        │
  │    — 에이전트 위임 없음, 단일 에이전트 실행               │
  │    — 기본값. 대부분의 경우 이것으로 충분                   │
  │                                                       │
  │  impl: DelegateOrchestrator                           │
  │    — [DELEGATE: agent_type] 신호 감지 시                │
  │    — 새로운 Pipeline 세션을 생성하여 하위 작업 위임        │
  │    — 위임된 세션은 독립적인 harness pipeline 실행          │
  │    — 결과를 수신하여 현재 파이프라인에 통합                 │
  │                                                       │
  │  impl: EvaluatorOrchestrator                          │
  │    — Anthropic의 Generator/Evaluator 패턴               │
  │    — 현재 파이프라인 = Generator                        │
  │    — 별도 경량 Pipeline = Evaluator                     │
  │    — Evaluator가 Generator의 출력을 평가                 │
  │    — 평가 결과를 Stage 12(Evaluate)로 전달               │
  │                                                       │
  │  impl: SwarmOrchestrator                              │
  │    — 여러 에이전트에게 동시 위임 후 결과 병합              │
  │    — 브레인스토밍, 다관점 분석 등에 활용                   │
  │                                                       │
  │  impl: ChainOrchestrator                              │
  │    — 에이전트 체인: A의 출력 → B의 입력 → C의 입력         │
  │    — Planner → Generator → Reviewer 패턴               │
  └──────────────────────────────────────────────────────┘

  ┌─ SubPipelineFactory ──────────────────────────────────┐
  │  역할: 위임용 하위 Pipeline 생성                         │
  │                                                       │
  │  create(agent_type: str) → Pipeline                   │
  │    — agent_type별 사전 정의된 Pipeline 구성 반환          │
  │    — e.g., "evaluator" → 경량 평가 전용 파이프라인        │
  │    — e.g., "researcher" → 검색 도구 중심 파이프라인       │
  │    — e.g., "coder" → 코드 도구 중심 파이프라인            │
  │                                                       │
  │  ※ 하위 Pipeline도 동일한 Stage 인터페이스 사용           │
  │  ※ 하위 Pipeline의 이벤트도 부모 EventBus로 전파          │
  └──────────────────────────────────────────────────────┘

핵심 원칙:
  "하나의 파이프라인이 모든 Harness를 관장한다."
  "개별적인 처리가 분리되는 경우만 별도 세션(별도 harness)으로 관리."
  
  예시 — Generator/Evaluator:
  ┌─ Main Pipeline (Generator) ─────────────────────────┐
  │  Stage 1~10: 정상 실행                                │
  │  Stage 11 (Agent):                                   │
  │    → EvaluatorOrchestrator 활성화 시:                  │
  │    → Sub-Pipeline 생성 (Evaluator 전용)               │
  │    → Generator 출력을 Evaluator에 전달                 │
  │    → Evaluator 평가 결과 수신                          │
  │    → 결과를 AgentResult에 포함                         │
  │  Stage 12 (Evaluate): Evaluator 결과 기반 판단         │
  │  Stage 13 (Loop): 재시도 필요 시 Stage 2로 재진입       │
  └────────────────────────────────────────────────────┘

Bypass 조건: SingleAgentOrchestrator 선택 시 (위임 없이 통과)
```

### Stage 12: Evaluate

```
분류: Decision
역할: 응답 품질 평가 & 완료 판단

Level 1 — Stage 인터페이스:
  입력: AgentResult (에이전트 결과 + 평가자 피드백 포함)
  출력: EvaluationResult

Level 2 — Strategy 인터페이스:

  ┌─ EvaluationStrategy ──────────────────────────────────┐
  │  역할: 응답 평가 방식                                    │
  │                                                       │
  │  impl: SignalBasedEvaluation                          │
  │    — CompletionSignal 기반 단순 판단                    │
  │    — [COMPLETE] → pass, [BLOCKED] → escalate 등        │
  │    — 기본값. 대부분의 경우 충분                           │
  │                                                       │
  │  impl: CriteriaBasedEvaluation                        │
  │    — 사전 정의된 품질 기준표 기반 채점                    │
  │    — Anthropic의 "주관적 품질 → 측정 가능 기준" 패턴      │
  │    — criteria: [{name, weight, threshold}]             │
  │                                                       │
  │  impl: AgentEvaluation                                │
  │    — Stage 11의 Evaluator 에이전트 결과 활용             │
  │    — 독립 평가자의 피드백 기반 판단                       │
  │                                                       │
  │  impl: ContractBasedEvaluation                        │
  │    — Anthropic의 Sprint Contract 패턴                   │
  │    — 실행 전 합의된 완료 기준 대비 검증                   │
  │    — contract: {criteria, verification_method}         │
  │                                                       │
  │  impl: CompositeEvaluation                            │
  │    — 여러 평가 전략 조합                                 │
  │    — e.g., Signal + Criteria + Agent 복합 판단          │
  └──────────────────────────────────────────────────────┘

  ┌─ QualityScorer ───────────────────────────────────────┐
  │  역할: 수치적 품질 점수 산출 (선택적)                     │
  │                                                       │
  │  impl: NoScorer          — 점수 없음 (pass/fail만)     │
  │  impl: WeightedScorer    — 가중 평균 점수               │
  │  impl: RubricScorer      — 루브릭 기반 상세 채점         │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 단순 대화 모드 (평가 불필요, SignalBased로 자동 처리)
```

### Stage 13: Loop

```
분류: Decision
역할: 에이전트 루프 제어 (계속/종료/재시도)

Level 1 — Stage 인터페이스:
  입력: EvaluationResult + PipelineState
  출력: LoopDecision

Level 2 — Strategy 인터페이스:

  ┌─ LoopController ──────────────────────────────────────┐
  │  역할: 루프 지속 여부 결정                               │
  │                                                       │
  │  impl: StandardLoopController                         │
  │    — tool_use → CONTINUE (도구 결과 다음 턴 전달)        │
  │    — COMPLETE signal → COMPLETE                        │
  │    — max_turns 도달 → FORCE_COMPLETE                    │
  │    — BLOCKED signal → ESCALATE                         │
  │    — 기본값                                            │
  │                                                       │
  │  impl: BudgetAwareLoopController                      │
  │    — 토큰/비용 예산 잔여량 기반 동적 판단                 │
  │    — 예산 부족 시 강제 종료 + 요약 생성                   │
  │                                                       │
  │  impl: QualityGatedLoopController                     │
  │    — 품질 점수가 threshold 이상이면 종료                  │
  │    — 미달이면 재시도 (with feedback injection)            │
  │                                                       │
  │  impl: SingleTurnController                           │
  │    — 1회 실행 후 무조건 종료 (루프 없음)                  │
  └──────────────────────────────────────────────────────┘

LoopDecision:
  CONTINUE   → Stage 2 (Context)로 재진입
  COMPLETE   → Stage 14 (Emit)로 진행
  ERROR      → 에러 핸들링 후 종료
  ESCALATE   → 사용자 개입 요청 후 대기
```

### Stage 14: Emit

```
분류: Egress
역할: 실행 결과를 외부 소비자에게 전달

Level 1 — Stage 인터페이스:
  입력: FinalResponse + PipelineState
  출력: EmittedResult

Level 2 — Strategy 인터페이스:

  ┌─ EmitterChain ────────────────────────────────────────┐
  │  역할: 여러 Emitter를 체인으로 연결, 모두 실행            │
  │                                                       │
  │  emitter: TextEmitter                                 │
  │    — 텍스트 응답 포맷팅 및 전달                          │
  │                                                       │
  │  emitter: StreamEmitter                               │
  │    — SSE/WebSocket 스트리밍 청크 전달                    │
  │                                                       │
  │  emitter: VTuberEmitter                               │
  │    — 감정 추출 (EmotionExtractor)                       │
  │    — 아바타 상태 업데이트                                │
  │    — Live2D 파라미터 생성                               │
  │                                                       │
  │  emitter: TTSEmitter                                  │
  │    — 텍스트를 TTS 엔진에 전달                            │
  │    — 음성 오디오 청크 생성 및 전달                        │
  │                                                       │
  │  emitter: NotificationEmitter                         │
  │    — 외부 서비스 알림 (Slack, Discord 등)                │
  │                                                       │
  │  ※ Emitter는 List로 등록. 병렬 실행.                    │
  │  ※ 커스텀 Emitter 추가 자유                             │
  └──────────────────────────────────────────────────────┘

  ┌─ EmotionExtractor (VTuber 전용) ──────────────────────┐
  │  역할: 텍스트에서 감정 상태 추출                          │
  │                                                       │
  │  impl: RuleBasedExtractor   — 키워드/패턴 기반           │
  │  impl: ModelBasedExtractor  — 경량 분류 모델 사용        │
  │  impl: LLMBasedExtractor    — Claude로 감정 분석        │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 내부 파이프라인 (결과가 부모 파이프라인으로만 반환)
```

### Stage 15: Memory

```
분류: Egress
역할: 메모리 업데이트 및 정리

Level 1 — Stage 인터페이스:
  입력: PipelineState + FinalResponse
  출력: MemoryUpdateResult

Level 2 — Strategy 인터페이스:

  ┌─ MemoryUpdateStrategy ────────────────────────────────┐
  │  역할: 메모리 업데이트 방식                               │
  │                                                       │
  │  impl: AppendOnlyStrategy                             │
  │    — 새 대화를 히스토리에 추가만                          │
  │    — 가장 단순, stateless에 가까운 모드                  │
  │                                                       │
  │  impl: ReflectiveStrategy                             │
  │    — 대화 완료 후 메모리 반영 (reflection)               │
  │    — 중요 정보 추출 → 장기 메모리에 저장                  │
  │    — 별도 경량 API 호출로 요약 생성 가능                  │
  │                                                       │
  │  impl: ConsolidationStrategy                          │
  │    — 주기적으로 단기 메모리를 장기 메모리로 통합            │
  │    — 중복 제거, 모순 해소, 정보 병합                      │
  │                                                       │
  │  impl: NoMemoryStrategy                               │
  │    — 메모리 업데이트 없음 (완전 stateless)               │
  └──────────────────────────────────────────────────────┘

  ┌─ ConversationPersistence ─────────────────────────────┐
  │  역할: 대화 히스토리 영속화                               │
  │                                                       │
  │  impl: InMemoryPersistence     — 메모리 내 (테스트용)    │
  │  impl: FilePersistence         — JSON 파일 저장          │
  │  impl: SQLitePersistence       — SQLite DB              │
  │  impl: PostgresPersistence     — PostgreSQL              │
  └──────────────────────────────────────────────────────┘

Bypass 조건: stateless 모드 (NoMemoryStrategy)
```

### Stage 16: Yield

```
분류: Egress
역할: 최종 결과 패키징 및 반환

Level 1 — Stage 인터페이스:
  입력: 모든 Stage 결과의 종합 (PipelineState)
  출력: PipelineResult

Level 2 — Strategy 인터페이스:

  ┌─ ResultFormatter ─────────────────────────────────────┐
  │  역할: 최종 결과 포맷팅                                  │
  │                                                       │
  │  impl: DefaultFormatter                               │
  │    — text + metadata + usage 패키징                     │
  │                                                       │
  │  impl: StructuredFormatter                            │
  │    — Pydantic 모델 기반 구조화 결과                      │
  │                                                       │
  │  impl: StreamingFormatter                             │
  │    — 최종 요약 이벤트 방출 (스트리밍 완료 신호)            │
  └──────────────────────────────────────────────────────┘

  ┌─ SessionSnapshot ─────────────────────────────────────┐
  │  역할: 세션 상태 스냅샷 저장                              │
  │                                                       │
  │  impl: FullSnapshot     — 전체 state 직렬화 저장         │
  │  impl: DeltaSnapshot    — 변경분만 저장 (효율적)          │
  │  impl: NoSnapshot       — 저장 없음                      │
  └──────────────────────────────────────────────────────┘

Bypass 조건: 없음 (항상 실행)
```

---

## 5. Core Interfaces

### 5.1 Stage Interface (Level 1 Abstraction)

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Any, Dict, List

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


class Stage(ABC, Generic[T_In, T_Out]):
    """파이프라인의 개별 단계 — Level 1 추상화.
    
    모든 Stage는 이 인터페이스를 구현해야 하며,
    execute()가 핵심 실행 로직, should_bypass()가 건너뛰기 판단을 담당한다.
    Stage 자체를 통째로 교체할 수 있다.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Stage 고유 이름 (e.g., 'input', 'context', 'api')"""
        ...
    
    @property
    @abstractmethod
    def order(self) -> int:
        """파이프라인 내 실행 순서 (1-16)"""
        ...
    
    @property
    def category(self) -> str:
        """Stage 분류 (ingress, pre_flight, execution, decision, egress)"""
        return "execution"
    
    @abstractmethod
    async def execute(self, input: T_In, state: "PipelineState") -> T_Out:
        """Stage의 핵심 실행 로직."""
        ...
    
    def should_bypass(self, state: "PipelineState") -> bool:
        """이 Stage를 건너뛸지 판단. 기본값 False."""
        return False
    
    async def on_enter(self, state: "PipelineState") -> None:
        """Stage 진입 시 호출되는 훅 (선택적)."""
        pass
    
    async def on_exit(self, result: T_Out, state: "PipelineState") -> None:
        """Stage 종료 시 호출되는 훅 (선택적)."""
        pass
    
    async def on_error(self, error: Exception, state: "PipelineState") -> Optional[T_Out]:
        """에러 발생 시 호출. None이면 전파, 값이면 복구."""
        return None
    
    def describe(self) -> "StageDescription":
        """Stage의 메타데이터 반환 (UI 표시용)."""
        return StageDescription(
            name=self.name,
            order=self.order,
            category=self.category,
            strategies=self.list_strategies(),
        )
    
    def list_strategies(self) -> List["StrategyInfo"]:
        """이 Stage에서 사용 가능한 Strategy 목록 (UI 표시용)."""
        return []
```

### 5.2 Strategy Interface (Level 2 Abstraction)

```python
class Strategy(ABC):
    """Stage 내부 로직의 교체 가능한 전략 — Level 2 추상화.
    
    각 Stage는 하나 이상의 Strategy 슬롯을 가지며,
    동일 Stage라도 Strategy 교체로 완전히 다른 동작을 수행할 수 있다.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """전략 고유 이름 (e.g., 'progressive_disclosure')"""
        ...
    
    @property
    def description(self) -> str:
        """전략 설명 (UI 표시용)"""
        return ""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """전략별 설정 주입. Stage 초기화 시 호출."""
        pass


# Stage에서 Strategy를 사용하는 패턴
class ContextStage(Stage[NormalizedInput, ExecutionContext]):
    """컨텍스트 수집 Stage — Level 2 Strategy를 조합하여 동작."""
    
    def __init__(
        self,
        context_strategy: "ContextStrategy",         # 필수
        history_compactor: "HistoryCompactor" = None, # 선택
        memory_retriever: "MemoryRetriever" = None,   # 선택
    ):
        self._context_strategy = context_strategy
        self._history_compactor = history_compactor or TruncateCompactor()
        self._memory_retriever = memory_retriever
    
    @property
    def name(self) -> str:
        return "context"
    
    @property
    def order(self) -> int:
        return 2
    
    async def execute(self, input: NormalizedInput, state: PipelineState) -> ExecutionContext:
        # Level 2 Strategy들에게 위임
        context = await self._context_strategy.build_context(input, state)
        
        if self._memory_retriever:
            memories = await self._memory_retriever.retrieve(input.text, state)
            context.memory_refs = memories
        
        if context.exceeds_budget(state.context_window_budget):
            context = await self._history_compactor.compact(context, state)
        
        return context
    
    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo("context_strategy", type(self._context_strategy).__name__),
            StrategyInfo("history_compactor", type(self._history_compactor).__name__),
            StrategyInfo("memory_retriever", type(self._memory_retriever).__name__ if self._memory_retriever else "None"),
        ]
```

### 5.3 PipelineState

```python
@dataclass
class PipelineState:
    """파이프라인 전체 실행 상태.
    
    모든 Stage에서 읽기/쓰기 가능하며, 루프 시 상태가 누적된다.
    """
    
    # ── Identity ──
    session_id: str = ""
    pipeline_id: str = ""
    
    # ── Messages (Anthropic API format) ──
    system: Union[str, List[Dict]] = ""     # str 또는 content blocks (cache용)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # ── Execution tracking ──
    iteration: int = 0
    max_iterations: int = 50
    current_stage: str = ""
    stage_history: List[str] = field(default_factory=list)
    
    # ── Model config ──
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.0
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Dict] = None
    
    # ── Extended Thinking ──
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # ── Token & Cost tracking ──
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_token_usage: List[TokenUsage] = field(default_factory=list)  # per-turn
    total_cost_usd: float = 0.0
    cost_budget_usd: Optional[float] = None
    
    # ── Cache tracking ──
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)
    
    # ── Context ──
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    context_window_budget: int = 200_000
    
    # ── Loop control ──
    loop_decision: str = "continue"
    completion_signal: Optional[str] = None
    completion_detail: Optional[str] = None
    
    # ── Tool execution ──
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # ── Agent orchestration ──
    delegate_requests: List[Dict[str, Any]] = field(default_factory=list)
    agent_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # ── Evaluation ──
    evaluation_score: Optional[float] = None
    evaluation_feedback: Optional[str] = None
    
    # ── Output ──
    final_text: str = ""
    final_output: Optional[Any] = None
    
    # ── Metadata ──
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ── Event log ──
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class CacheMetrics:
    total_cache_writes: int = 0
    total_cache_reads: int = 0
    estimated_savings_usd: float = 0.0
    cache_hit_rate: float = 0.0
```

### 5.4 Pipeline Engine

```python
class Pipeline:
    """Stage들을 순서대로 실행하는 파이프라인 엔진.
    
    실행 모델:
      Phase A: Input (Stage 1, 1회)
      Phase B: Agent Loop (Stage 2~13, 반복)
      Phase C: Finalize (Stage 14~16, 1회)
    """
    
    def __init__(self, config: "PipelineConfig"):
        self._stages: Dict[int, Stage] = {}
        self._event_bus: EventBus = EventBus()
        self._config = config
    
    # ── Stage 관리 ──
    
    def register_stage(self, stage: Stage) -> "Pipeline":
        """Stage 등록/교체. 체이닝 지원."""
        self._stages[stage.order] = stage
        return self
    
    def replace_stage(self, order: int, stage: Stage) -> "Pipeline":
        """기존 Stage를 새 Stage로 교체."""
        self._stages[order] = stage
        return self
    
    def remove_stage(self, order: int) -> "Pipeline":
        """Stage 제거 (해당 순서 자동 bypass)."""
        self._stages.pop(order, None)
        return self
    
    def get_stage(self, order: int) -> Optional[Stage]:
        """등록된 Stage 조회."""
        return self._stages.get(order)
    
    # ── 실행 ──
    
    async def run(self, input: Any, state: PipelineState = None) -> "PipelineResult":
        """파이프라인 전체 실행.
        
        Phase A: Stage 1 (Input) — 1회 실행
        Phase B: Stage 2~13 (Agent Loop) — 반복 실행
        Phase C: Stage 14~16 (Finalize) — 1회 실행
        """
        state = state or PipelineState()
        await self._event_bus.emit(PipelineEvent("pipeline.start", data={"input": str(input)}))
        
        try:
            # Phase A: Input
            current = await self._run_stage(1, input, state)
            
            # Phase B: Agent Loop
            while True:
                for order in range(2, 14):  # Stage 2 ~ 13
                    stage = self._stages.get(order)
                    if stage is None or stage.should_bypass(state):
                        await self._event_bus.emit(PipelineEvent("stage.bypass", stage=stage.name if stage else f"stage_{order}"))
                        continue
                    current = await self._run_stage(order, current, state)
                
                if state.loop_decision != "continue":
                    break
                state.iteration += 1
            
            # Phase C: Finalize
            for order in range(14, 17):  # Stage 14 ~ 16
                stage = self._stages.get(order)
                if stage is None or stage.should_bypass(state):
                    continue
                current = await self._run_stage(order, current, state)
            
            await self._event_bus.emit(PipelineEvent("pipeline.complete"))
            return PipelineResult.from_state(state)
            
        except Exception as e:
            await self._event_bus.emit(PipelineEvent("pipeline.error", data={"error": str(e)}))
            raise
    
    async def run_stream(self, input: Any, state: PipelineState = None) -> AsyncIterator["PipelineEvent"]:
        """스트리밍 모드. 각 Stage 이벤트를 실시간 yield."""
        ...
    
    # ── 이벤트 ──
    
    def on(self, event_type: str, handler: Callable) -> None:
        """이벤트 핸들러 등록."""
        self._event_bus.on(event_type, handler)
    
    # ── UI 메타데이터 ──
    
    def describe(self) -> "PipelineDescription":
        """파이프라인 전체 구조 반환 (Pipeline UI 렌더링용)."""
        return PipelineDescription(
            stages=[
                stage.describe() for stage in sorted(self._stages.values(), key=lambda s: s.order)
            ]
        )
    
    # ── 내부 ──
    
    async def _run_stage(self, order: int, input: Any, state: PipelineState) -> Any:
        stage = self._stages.get(order)
        if stage is None:
            return input
        
        state.current_stage = stage.name
        state.stage_history.append(stage.name)
        await self._event_bus.emit(PipelineEvent("stage.enter", stage=stage.name, iteration=state.iteration))
        
        await stage.on_enter(state)
        try:
            result = await stage.execute(input, state)
            await stage.on_exit(result, state)
            await self._event_bus.emit(PipelineEvent("stage.exit", stage=stage.name, iteration=state.iteration))
            return result
        except Exception as e:
            await self._event_bus.emit(PipelineEvent("stage.error", stage=stage.name, data={"error": str(e)}))
            recovery = await stage.on_error(e, state)
            if recovery is not None:
                return recovery
            raise
```

### 5.5 Pipeline Builder (편의 API)

```python
class PipelineBuilder:
    """파이프라인을 선언적으로 구성하는 빌더.
    
    사용 예:
        pipeline = (
            PipelineBuilder("my-agent")
            .with_model("claude-sonnet-4-20250514")
            .with_input(validator=StrictValidator())
            .with_context(
                strategy=ProgressiveDisclosureStrategy(),
                compactor=SummaryCompactor(),
                retriever=VectorMemoryRetriever(index=my_index),
            )
            .with_system(builder=ComposablePromptBuilder(blocks=[
                PersonaBlock("Geny VTuber"),
                RulesBlock(rules),
                DateTimeBlock(),
            ]))
            .with_guard(guards=[TokenBudgetGuard(), CostBudgetGuard(max_usd=5.0)])
            .with_cache(strategy=AggressiveCacheStrategy())
            .with_api(provider=AnthropicProvider(api_key=key))
            .with_tools(registry=tool_registry, executor=ParallelExecutor())
            .with_agent(orchestrator=SingleAgentOrchestrator())
            .with_evaluate(strategy=SignalBasedEvaluation())
            .with_loop(controller=StandardLoopController(max_turns=20))
            .with_emit(emitters=[TextEmitter(), VTuberEmitter()])
            .with_memory(strategy=ReflectiveStrategy(), persistence=SQLitePersistence())
            .build()
        )
    """
    
    def __init__(self, name: str): ...
    def with_model(self, model: str) -> "PipelineBuilder": ...
    def with_input(self, **kwargs) -> "PipelineBuilder": ...
    def with_context(self, **kwargs) -> "PipelineBuilder": ...
    def with_system(self, **kwargs) -> "PipelineBuilder": ...
    def with_guard(self, **kwargs) -> "PipelineBuilder": ...
    def with_cache(self, **kwargs) -> "PipelineBuilder": ...
    def with_api(self, **kwargs) -> "PipelineBuilder": ...
    def with_token(self, **kwargs) -> "PipelineBuilder": ...
    def with_think(self, **kwargs) -> "PipelineBuilder": ...
    def with_parse(self, **kwargs) -> "PipelineBuilder": ...
    def with_tools(self, **kwargs) -> "PipelineBuilder": ...
    def with_agent(self, **kwargs) -> "PipelineBuilder": ...
    def with_evaluate(self, **kwargs) -> "PipelineBuilder": ...
    def with_loop(self, **kwargs) -> "PipelineBuilder": ...
    def with_emit(self, **kwargs) -> "PipelineBuilder": ...
    def with_memory(self, **kwargs) -> "PipelineBuilder": ...
    def with_yield(self, **kwargs) -> "PipelineBuilder": ...
    def build(self) -> Pipeline: ...
```

### 5.6 Preset Pipelines (사전 구성)

```python
class PipelinePresets:
    """자주 사용되는 파이프라인 구성을 사전 정의."""
    
    @staticmethod
    def minimal(api_key: str, model: str = "claude-sonnet-4-20250514") -> Pipeline:
        """최소 파이프라인 — 단순 질의/응답.
        
        활성 Stage: Input → API → Parse → Yield
        """
        ...
    
    @staticmethod
    def chat(api_key: str, model: str = "claude-sonnet-4-20250514") -> Pipeline:
        """대화형 파이프라인 — 히스토리 유지, 기본 도구.
        
        활성 Stage: Input → Context → System → Guard → Cache
                     → API → Token → Parse → Tool → Loop
                     → Emit → Memory → Yield
        """
        ...
    
    @staticmethod
    def agent(api_key: str, model: str = "claude-sonnet-4-20250514") -> Pipeline:
        """에이전트 파이프라인 — 풀 스택 자율 에이전트.
        
        활성 Stage: 전체 16 Stage
        """
        ...
    
    @staticmethod
    def geny_vtuber(api_key: str, config: "GenyConfig") -> Pipeline:
        """Geny VTuber 파이프라인 — 기존 Geny 완전 재현.
        
        활성 Stage: 전체 16 Stage + VTuber/TTS Emitter
        """
        ...
    
    @staticmethod
    def evaluator(api_key: str, criteria: List) -> Pipeline:
        """평가 전용 경량 파이프라인 — Generator/Evaluator 패턴의 Evaluator.
        
        활성 Stage: Input → System → API → Parse → Evaluate → Yield
        """
        ...
```

---

## 6. Tool & MCP System

### 6.1 Tool Interface

```python
class Tool(ABC):
    """도구 인터페이스. Anthropic API의 tool 정의와 1:1 매핑."""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod
    def description(self) -> str: ...
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    async def execute(self, input: Dict[str, Any], context: "ToolContext") -> "ToolResult": ...
    
    def to_api_format(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """도구 레지스트리 — 등록, 조회, 프리셋 관리."""
    
    def register(self, tool: Tool) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> Optional[Tool]: ...
    def list_all(self) -> List[Tool]: ...
    def apply_preset(self, preset: "ToolPreset") -> List[Tool]: ...
    def to_api_format(self) -> List[Dict]: ...
```

### 6.2 MCP Integration

```python
class MCPManager:
    """MCP 서버 관리 — SDK 직접 사용 (CLI 경유 X)."""
    
    async def connect(self, name: str, config: MCPServerConfig) -> None: ...
    async def disconnect(self, name: str) -> None: ...
    async def discover_tools(self) -> List[Tool]: ...
    async def call_tool(self, server_name: str, tool_name: str, args: Dict) -> Any: ...
    
    @classmethod
    def from_config_file(cls, path: str) -> "MCPManager":
        """기존 .mcp.json 호환 로딩."""
        ...


class MCPToolAdapter(Tool):
    """MCP 서버 도구를 Tool 인터페이스로 래핑."""
    
    def __init__(self, server: "MCPServerConnection", definition: Dict): ...
    
    async def execute(self, input: Dict, context: "ToolContext") -> "ToolResult":
        return await self._server.call_tool(self.name, input)
```

---

## 7. Session Management

### 7.1 Session Lifecycle

```
CREATED → READY → RUNNING ⇄ IDLE → STALE → CLOSED
                     ↑                  │
                     └── revive() ──────┘
```

### 7.2 Session & SessionManager

```python
class Session:
    """에이전트 세션 — Pipeline + State의 실행 단위."""
    
    def __init__(self, session_id: str, pipeline: Pipeline, config: SessionConfig): ...
    
    async def run(self, input: str) -> PipelineResult: ...
    async def run_stream(self, input: str) -> AsyncIterator[PipelineEvent]: ...
    
    @property
    def state(self) -> PipelineState: ...
    
    @property
    def freshness(self) -> FreshnessStatus: ...


class SessionManager:
    """세션 CRUD + 라이프사이클 관리."""
    
    async def create(self, config: CreateSessionConfig) -> Session: ...
    async def get(self, session_id: str) -> Optional[Session]: ...
    async def delete(self, session_id: str) -> None: ...
    async def list(self) -> List[SessionInfo]: ...
    async def revive(self, session_id: str) -> Session: ...
```

### 7.3 FreshnessPolicy

```python
class FreshnessPolicy:
    """세션 신선도 정책 — 기존 Geny의 SessionFreshness 이식."""
    
    def evaluate(self, session: Session) -> FreshnessStatus:
        ...
    
    # FreshnessStatus: FRESH | STALE_WARN | STALE_IDLE | STALE_COMPACT | STALE_RESET
```

---

## 8. Event System

### 8.1 EventBus

```python
class EventBus:
    """파이프라인 이벤트 시스템 — 모든 Stage 전이를 실시간 전파."""
    
    async def emit(self, event: PipelineEvent) -> None: ...
    def on(self, event_type: str, handler: Callable) -> Callable: ...  # returns unsubscribe
    def off(self, event_type: str, handler: Callable) -> None: ...


@dataclass
class PipelineEvent:
    type: str               # "stage.enter", "api.stream.chunk", "tool.execute", ...
    stage: str = ""
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
```

### 8.2 Event Types

```
Pipeline Lifecycle:
  pipeline.start            파이프라인 시작
  pipeline.complete         파이프라인 완료
  pipeline.error            파이프라인 에러

Stage Transitions:
  stage.enter               Stage 진입
  stage.exit                Stage 종료
  stage.bypass              Stage 건너뜀
  stage.error               Stage 에러

API Events:
  api.request               API 호출 시작
  api.response              API 응답 수신
  api.stream.start          스트리밍 시작
  api.stream.chunk          스트리밍 청크 (text_delta, tool_use 등)
  api.stream.end            스트리밍 종료
  api.retry                 재시도 발생

Tool Events:
  tool.execute              도구 실행 시작
  tool.result               도구 실행 결과
  tool.error                도구 실행 에러

Agent Events:
  agent.delegate            하위 에이전트 위임
  agent.result              하위 에이전트 결과

Thinking Events:
  think.start               Extended Thinking 시작
  think.content             Thinking 내용 청크
  think.end                 Extended Thinking 종료

Loop Events:
  loop.continue             루프 계속 (재진입)
  loop.complete             루프 종료
  loop.escalate             에스컬레이션

Memory Events:
  memory.load               메모리 로딩
  memory.save               메모리 저장

Cache Events:
  cache.hit                 캐시 히트
  cache.miss                캐시 미스
  cache.write               캐시 생성

Emit Events:
  emit.text                 텍스트 출력
  emit.vtuber.emotion       VTuber 감정 상태
  emit.tts.audio            TTS 오디오 청크
```

---

## 9. Package Structure

```
geny-executor/
├── pyproject.toml
├── README.md
├── PLAN.md
│
├── src/
│   └── geny_executor/
│       ├── __init__.py                  # Public API
│       ├── py.typed
│       │
│       ├── core/                        # 핵심 엔진
│       │   ├── __init__.py
│       │   ├── pipeline.py              # Pipeline 엔진
│       │   ├── stage.py                 # Stage ABC
│       │   ├── strategy.py              # Strategy ABC
│       │   ├── state.py                 # PipelineState, TokenUsage, CacheMetrics
│       │   ├── config.py                # PipelineConfig, ModelConfig
│       │   ├── result.py                # PipelineResult
│       │   ├── builder.py               # PipelineBuilder
│       │   ├── presets.py               # PipelinePresets
│       │   └── errors.py               # 에러 분류 체계
│       │
│       ├── stages/                      # 16개 기본 Stage + 내부 Strategy들
│       │   ├── __init__.py
│       │   │
│       │   ├── s01_input/               # Stage 1: Input
│       │   │   ├── __init__.py          #   InputStage
│       │   │   ├── stage.py             #   Stage 구현
│       │   │   ├── validators.py        #   InputValidator impls
│       │   │   └── normalizers.py       #   InputNormalizer impls
│       │   │
│       │   ├── s02_context/             # Stage 2: Context
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── strategies.py        #   ContextStrategy impls
│       │   │   ├── compactors.py        #   HistoryCompactor impls
│       │   │   └── retrievers.py        #   MemoryRetriever impls
│       │   │
│       │   ├── s03_system/              # Stage 3: System
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── builders.py          #   PromptBuilder impls
│       │   │   └── formatters.py        #   ToolDescriptionFormatter impls
│       │   │
│       │   ├── s04_guard/               # Stage 4: Guard
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   └── guards.py            #   Guard impls (chain)
│       │   │
│       │   ├── s05_cache/               # Stage 5: Cache
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── strategies.py        #   CacheStrategy impls
│       │   │   └── analyzer.py          #   CacheAnalyzer
│       │   │
│       │   ├── s06_api/                 # Stage 6: API
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── providers.py         #   APIProvider impls
│       │   │   ├── streaming.py         #   StreamHandler
│       │   │   └── retry.py             #   RetryStrategy impls
│       │   │
│       │   ├── s07_token/               # Stage 7: Token
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── trackers.py          #   TokenTracker impls
│       │   │   └── pricing.py           #   CostCalculator impls
│       │   │
│       │   ├── s08_think/               # Stage 8: Think
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   └── processors.py        #   ThinkingProcessor impls
│       │   │
│       │   ├── s09_parse/               # Stage 9: Parse
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── parsers.py           #   ResponseParser impls
│       │   │   └── signals.py           #   CompletionSignalDetector impls
│       │   │
│       │   ├── s10_tool/                # Stage 10: Tool
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── executors.py         #   ToolExecutor impls
│       │   │   ├── routers.py           #   ToolRouter impls
│       │   │   └── permissions.py       #   ToolPermission impls
│       │   │
│       │   ├── s11_agent/               # Stage 11: Agent
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── orchestrators.py     #   AgentOrchestrator impls
│       │   │   └── factory.py           #   SubPipelineFactory
│       │   │
│       │   ├── s12_evaluate/            # Stage 12: Evaluate
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── strategies.py        #   EvaluationStrategy impls
│       │   │   └── scorers.py           #   QualityScorer impls
│       │   │
│       │   ├── s13_loop/                # Stage 13: Loop
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   └── controllers.py       #   LoopController impls
│       │   │
│       │   ├── s14_emit/                # Stage 14: Emit
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   └── emitters.py          #   Emitter impls (chain)
│       │   │
│       │   ├── s15_memory/              # Stage 15: Memory
│       │   │   ├── __init__.py
│       │   │   ├── stage.py
│       │   │   ├── strategies.py        #   MemoryUpdateStrategy impls
│       │   │   └── persistence.py       #   ConversationPersistence impls
│       │   │
│       │   └── s16_yield/               # Stage 16: Yield
│       │       ├── __init__.py
│       │       ├── stage.py
│       │       ├── formatters.py        #   ResultFormatter impls
│       │       └── snapshots.py         #   SessionSnapshot impls
│       │
│       ├── tools/                       # 도구 시스템
│       │   ├── __init__.py
│       │   ├── base.py                  # Tool ABC, ToolResult
│       │   ├── registry.py              # ToolRegistry
│       │   ├── presets.py               # ToolPreset
│       │   ├── mcp/                     # MCP 통합
│       │   │   ├── __init__.py
│       │   │   ├── manager.py           # MCPManager
│       │   │   ├── adapter.py           # MCPToolAdapter
│       │   │   └── config.py            # MCPServerConfig
│       │   └── builtin/                 # 내장 도구
│       │       ├── __init__.py
│       │       ├── bash.py
│       │       ├── file_ops.py
│       │       ├── web.py
│       │       └── knowledge.py
│       │
│       ├── memory/                      # 메모리 시스템
│       │   ├── __init__.py
│       │   ├── provider.py              # MemoryProvider ABC
│       │   ├── file_memory.py
│       │   ├── vector_memory.py
│       │   └── conversation.py          # ConversationStore
│       │
│       ├── session/                     # 세션 관리
│       │   ├── __init__.py
│       │   ├── manager.py               # SessionManager
│       │   ├── session.py               # Session 클래스
│       │   ├── freshness.py             # FreshnessPolicy
│       │   └── store.py                 # SessionStore (영속화)
│       │
│       ├── events/                      # 이벤트 시스템
│       │   ├── __init__.py
│       │   ├── bus.py                   # EventBus
│       │   ├── types.py                 # PipelineEvent, event type constants
│       │   └── handlers.py              # 기본 핸들러 (로거 등)
│       │
│       └── extensions/                  # Geny 확장 (선택적)
│           ├── __init__.py
│           ├── vtuber/
│           │   ├── emotion.py           # EmotionExtractor impls
│           │   └── avatar.py            # 아바타 상태 관리
│           ├── tts/
│           │   └── emitter.py           # TTSEmitter
│           └── workflow/
│               └── converter.py         # WorkflowDefinition → Pipeline 변환
│
├── tests/
│   ├── unit/
│   │   ├── test_pipeline.py
│   │   ├── test_stages/
│   │   │   ├── test_s01_input.py
│   │   │   ├── test_s02_context.py
│   │   │   └── ...                      # 각 Stage별 테스트
│   │   ├── test_tools/
│   │   └── test_memory/
│   ├── integration/
│   │   ├── test_api_integration.py      # 실제 API 테스트
│   │   ├── test_mcp_integration.py
│   │   ├── test_full_pipeline.py
│   │   └── test_mock_pipeline.py        # Mock API 테스트
│   └── fixtures/
│       ├── mock_provider.py             # MockProvider + ReplayProvider fixtures
│       └── sample_configs.py
│
└── examples/
    ├── 01_minimal.py                    # 최소 파이프라인
    ├── 02_chat.py                       # 대화형
    ├── 03_tool_use.py                   # 도구 사용
    ├── 04_custom_stage.py               # 커스텀 Stage
    ├── 05_custom_strategy.py            # 커스텀 Strategy (Level 2)
    ├── 06_mcp_server.py                 # MCP 연동
    ├── 07_streaming.py                  # 스트리밍 출력
    ├── 08_extended_thinking.py          # Extended Thinking
    ├── 09_multi_agent.py                # Multi-Agent 오케스트레이션
    ├── 10_prompt_caching.py             # Prompt Caching 전략
    ├── 11_pipeline_ui_metadata.py       # Pipeline UI 메타데이터
    └── 12_geny_migration.py             # Geny 이식 예제
```

---

## 10. Geny 호환성 매핑

### 10.1 기존 컴포넌트 → geny-executor 매핑

| Geny 기존 | geny-executor 대응 |
|-----------|-------------------|
| `ClaudeProcess` (subprocess) | `s06_api/providers.py:AnthropicProvider` |
| `StreamParser` (JSON line) | `s06_api/streaming.py:StreamHandler` |
| `ClaudeCLIChatModel` (LangChain) | `s06_api/stage.py:APIStage` |
| `AutonomousState` (TypedDict) | `core/state.py:PipelineState` |
| `AutonomousGraph` (30 nodes) | `Pipeline` (16 stages) |
| `BaseNode` (워크플로우) | `core/stage.py:Stage` + `core/strategy.py:Strategy` |
| `WorkflowExecutor` (컴파일러) | `extensions/workflow/converter.py` |
| `SessionMemoryManager` | `memory/provider.py:MemoryProvider` |
| `ToolPreset` + `ToolLoader` | `tools/registry.py:ToolRegistry` |
| `MCPConfig` + `.mcp.json` | `tools/mcp/manager.py:MCPManager` |
| `SessionFreshness` | `session/freshness.py:FreshnessPolicy` |
| `ExecutionSummary` | `core/result.py:PipelineResult` |
| `CompletionSignal` | `s09_parse/signals.py:CompletionSignalDetector` |
| `SessionLogger` (SSE) | `events/bus.py:EventBus` |
| `context_guard` (LangGraph node) | `s04_guard/guards.py:TokenBudgetGuard` |
| `post_model` (LangGraph node) | `Stage.on_exit()` + `s07_token/stage.py` |
| `memory_inject` (LangGraph node) | `s02_context/retrievers.py:MemoryRetriever` |
| `classify_difficulty` (LangGraph node) | 커스텀 Strategy in Stage 3 or 4 |
| `vtuber_respond` (LangGraph node) | `s14_emit/emitters.py:VTuberEmitter` |

### 10.2 기존 30+ Workflow Node → 16 Stage 흡수 매핑

```
기존 노드                          → Stage (Strategy)
─────────────────────────────────────────────────────────
memory_inject                      → S2  (MemoryRetriever)
relevance_gate                     → S4  (RelevanceGuard)
classify_difficulty                → S3  (AdaptivePromptBuilder) or S4 (커스텀 Guard)
context_guard                      → S4  (TokenBudgetGuard)
direct_answer, answer, llm_call    → S6  (AnthropicProvider)
post_model                         → S7  (TokenTracker) + S9 (CompletionSignalDetector)
create_todos, execute_todo         → S10 (ToolExecutor) — task를 tool로 모델링
review, final_review               → S11 (EvaluatorOrchestrator) + S12
iteration_gate, check_progress     → S13 (LoopController)
vtuber_respond, vtuber_classify    → S14 (VTuberEmitter)
transcript_record                  → S14 (EmitterChain) 또는 EventBus handler
memory_reflect                     → S15 (ReflectiveStrategy)
final_answer, final_synthesis      → S16 (ResultFormatter)
```

---

## 11. Pipeline UI Design

### 11.1 개념

기존 Geny의 캔버스(자유 노드 배치) → **Pipeline UI(고정 슬롯 + 교체)**

```
┌── Pipeline UI ─────────────────────────────────────────────────┐
│                                                                │
│  ┌──[1]──┐  ┌──[2]──┐  ┌──[3]──┐  ┌──[4]──┐  ┌──[5]──┐     │
│  │ Input │→│Context│→│System │→│ Guard │→│ Cache │ ... │
│  │       │  │       │  │       │  │       │  │       │     │
│  │ ▼impl │  │ ▼impl │  │ ▼impl │  │ ▼impl │  │ ▼impl │     │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘     │
│                                                                │
│  각 Stage 클릭 시:                                              │
│  ┌─ Stage 2: Context ──────────────────────────────────────┐  │
│  │                                                         │  │
│  │  Context Strategy:  [▼ ProgressiveDisclosure       ]    │  │
│  │  History Compactor: [▼ SummaryCompactor             ]    │  │
│  │  Memory Retriever:  [▼ VectorMemoryRetriever        ]    │  │
│  │                                                         │  │
│  │  ── Strategy Config ─────────────────────────────────   │  │
│  │  max_results: [5]                                       │  │
│  │  similarity_threshold: [0.7]                            │  │
│  │  include_categories: [general, knowledge]               │  │
│  │                                                         │  │
│  │  [Bypass this stage]  [Reset to default]                │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 11.2 Pipeline.describe() API

```python
@dataclass
class StageDescription:
    name: str
    order: int
    category: str           # ingress, pre_flight, execution, decision, egress
    is_active: bool         # bypass 여부
    strategies: List[StrategyInfo]

@dataclass
class StrategyInfo:
    slot_name: str          # "context_strategy", "history_compactor"
    current_impl: str       # "ProgressiveDisclosureStrategy"
    available_impls: List[str]  # 선택 가능한 다른 구현체들
    config: Dict[str, Any]  # 현재 설정값

@dataclass 
class PipelineDescription:
    stages: List[StageDescription]
    
    # UI가 이 데이터를 소비하여 파이프라인 시각화 렌더링
```

---

## 12. Dependencies

```toml
[project]
name = "geny-executor"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.52.0",       # Anthropic SDK (Messages API)
    "mcp>=1.0.0",              # MCP 프로토콜 클라이언트
    "pydantic>=2.0",           # 데이터 검증 & 설정
]

[project.optional-dependencies]
memory = [
    "numpy>=1.24",
    "faiss-cpu>=1.7",
]
all = [
    "geny-executor[memory]",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
]
```

---

## 13. Implementation Phases

### Phase 1: Core Engine + Minimal Pipeline (Week 1~2)

```
목표: Pipeline 엔진 + 최소 동작 (Input → API → Parse → Yield)

구현:
  core/           — pipeline, stage, strategy, state, config, result, errors
  stages/s01_input/   — DefaultValidator, DefaultNormalizer
  stages/s06_api/     — AnthropicProvider, MockProvider, ExponentialBackoffRetry
  stages/s09_parse/   — DefaultParser, RegexDetector
  stages/s16_yield/   — DefaultFormatter

검증: examples/01_minimal.py 동작
```

### Phase 2: Agent Loop + Tool System (Week 2~3)

```
목표: 도구 사용 에이전트 루프 완성

구현:
  stages/s04_guard/   — TokenBudgetGuard, CostBudgetGuard, IterationGuard
  stages/s07_token/   — DefaultTracker, AnthropicPricingCalculator
  stages/s10_tool/    — SequentialExecutor, ParallelExecutor, RegistryRouter
  stages/s13_loop/    — StandardLoopController
  tools/              — base, registry, presets
  tools/builtin/      — bash, file_ops
  tools/mcp/          — MCPManager, MCPToolAdapter

검증: examples/03_tool_use.py, examples/06_mcp_server.py 동작
```

### Phase 3: Context + Memory + Cache (Week 3~4)

```
목표: 대화 지속성, 메모리, 캐싱

구현:
  stages/s02_context/ — 모든 ContextStrategy, Compactor, Retriever
  stages/s03_system/  — ComposablePromptBuilder, blocks
  stages/s05_cache/   — 모든 CacheStrategy, CacheAnalyzer
  stages/s15_memory/  — 모든 MemoryUpdateStrategy, Persistence
  memory/             — FileMemory, VectorMemory, ConversationStore
  session/            — SessionManager, FreshnessPolicy

검증: examples/02_chat.py, examples/10_prompt_caching.py 동작
```

### Phase 4: Think + Agent + Evaluate (Week 4~5)

```
목표: Extended Thinking, Multi-Agent, 평가 시스템

구현:
  stages/s08_think/     — 모든 ThinkingProcessor
  stages/s11_agent/     — 모든 AgentOrchestrator, SubPipelineFactory
  stages/s12_evaluate/  — 모든 EvaluationStrategy, QualityScorer
  core/builder.py       — PipelineBuilder
  core/presets.py       — PipelinePresets

검증: examples/08_extended_thinking.py, examples/09_multi_agent.py 동작
```

### Phase 5: Emit + Events + Geny Extensions (Week 5~6)

```
목표: 출력 시스템, 이벤트, Geny 호환

구현:
  stages/s14_emit/      — 모든 Emitter (Text, Stream, VTuber, TTS)
  events/               — EventBus, types, handlers
  extensions/           — VTuber, TTS, Workflow converter

검증: examples/07_streaming.py, examples/12_geny_migration.py 동작
```

### Phase 6: Testing + Documentation + Pipeline UI metadata (Week 6~7)

```
목표: 프로덕션 준비

구현:
  tests/unit/           — 각 Stage + Strategy 단위 테스트
  tests/integration/    — Mock + 실제 API 통합 테스트
  Pipeline.describe()   — UI 메타데이터 API
  examples/             — 전체 예제 완성

검증: 90%+ coverage, 모든 예제 동작
```

---

## 14. Key Design Decisions Summary

| 결정 | 근거 |
|------|------|
| **16단계** | Claude Code 11 + Think/Cache/Agent/Evaluate/Memory |
| **Dual Abstraction** | Stage(Level 1) + Strategy(Level 2)로 두 레벨 모두 교체 가능 |
| **Extended Thinking = 별도 Stage** | 추론 과정이 독립적 관리/저장/분석이 필요한 1급 시민 |
| **Multi-Agent = 파이프라인 내부** | 단일 파이프라인이 모든 harness 관장, 위임 시에만 별도 세션 |
| **Prompt Caching = 별도 Stage** | 비용 최적화의 핵심, cache breakpoint 전략이 독립적으로 발전 |
| **Pipeline UI** | 캔버스가 아닌 고정 슬롯 + 교체 형식 |
| **Batch = 후순위** | 단일 파이프라인 완성 우선, 라이브러리 레벨에서 추후 지원 |
| **Mock + Real API 테스트** | MockProvider/RecordingProvider/ReplayProvider로 양쪽 지원 |

---

*이 계획서는 2차 이터레이션입니다. 방향성 합의 후 구현을 시작합니다.*
