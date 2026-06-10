# geny-executor 아키텍처 심층 검토 — Environment Management 철학 대비

- **날짜**: 2026-06-09
- **기준 버전**: geny-executor 2.1.4 (HEAD `c455643`), Geny `874d324`, GAPT (2.1.0 고정)
- **방법**: 14개 병렬 audit agent (9 subsystem 매핑 + 5 관점별 비판 검증), ~640 tool calls.
  모든 주장은 file:line 단위로 코드에서 직접 검증됨. README/PROGRESS 류 문서는 1차 출처로
  사용하지 않음 (코드가 기준).

---

## 0. 한 줄 결론

**골격은 진짜고, 의미는 아직 구호다.** 21-stage layout, manifest 직렬화/복원, slot/strategy
레지스트리, introspection, mutation/snapshot — "Environment 로 모든 것을 제어한다"는 철학의
**뼈대는 실제로 구현되어 있고 품질도 높다.** 그러나 세 개의 단층에서 철학과 코드가 갈라진다:

1. **Config fidelity 가 3계층으로 갈라짐** — stage/artifact/strategy *선택*은 완벽 작동,
   stage-level *config* 는 21개 중 9개 stage 만 소비, strategy-level *config* 는 가장
   중요한 전략들에서 **조용히 전부 버려짐**. 이것 때문에 Geny prod manifest 가 지금
   의미적으로 깨진 채 돌고 있다 (§2.1).
2. **~10개의 config 채널이 우선순위 정의 없이 공존** — `attach_runtime(llm_client=)` 이
   여전히 manifest 를 무조건 이기고 (#866 풋건 구조 잔존), state 편집은 매 run 스톰핑되고,
   `_config` 편집은 영구 지속. 그 결과 두 host 모두 private attribute 를 직접 만지는
   핵을 운영 중.
3. **Host 보상 레이어 ~3,800줄** — Geny ~2,500줄 + GAPT ~1,300줄. 전부 "라이브러리에
   없는 것"을 메우는 코드다: monkey-patch 모듈 2개, 이벤트 taxonomy 역공학, manifest
   빌더 사본 3개, 세션 rehydration 700줄. "adapter layer 금지" 룰의 위반은 host 의
   기강 문제가 아니라 라이브러리 공백의 정확한 지도다.

---

## 1. 철학 5개 조항별 판정

### 1-1. "Environment manifest = single source of truth" — **부분 충족 (구조적 구멍 존재)**

manifest 는 provider-location 수준에서는 잘 검증된다 (`_validate_manifest_provider_locations`
가 legacy 위치 거부, strict 모드의 stage-config schema 검증, attach_runtime 의 시작 후
hard-error). 그러나 실제 런타임에서 manifest 는 **약 10개 채널 중 하나의 목소리**일 뿐이며,
양방향으로 우회된다:

**위에서 덮어쓰는 채널 (manifest 를 이김):**
| 채널 | 수명 | 비고 |
|---|---|---|
| `attach_runtime(llm_client=)` | 세션 영구 | manifest provider 무조건 무시. #866 의 원인. 여전히 docstring 에 미기재 |
| `PipelineMutator` | 영구 (change-log 됨) | 정당한 escape hatch — 그러나 lock 미작동, restore 가 drift 를 조용히 스킵 |
| 직접 slot 대입 (`pipeline._stages[..].slot.strategy = X`) | 영구 | Geny install_* 헬퍼들이 사용. snapshot 복원 불가 상태를 만듦 |
| host 의 pre-build manifest rewrite | per-build | Geny `_force_required_stages_active`, GAPT ManifestOverrides dual-write |
| process env sniffing | 호출별 | `--bare` 결정이 부모 process env 의 `ANTHROPIC_API_KEY` 를 봄 (자식은 못 보는 변수) |

**아래에서 무시하는 채널 (manifest 가 선언해도 무효):**
| 선언 | 실제 |
|---|---|
| `strategy_configs` (s06 router/retry, s14 evaluator, s16 budget, s10 executor, s11 reviewer, s17 emitter) | base `Strategy.configure` 가 `pass` — **전부 버려짐** (§2.1) |
| 12개 stage 의 `config` dict | base `Stage.update_config` 가 `pass` — 장식 |
| `host_selections` (hooks/skills/permissions) | 라이브러리 내 `HostSelections.resolve` 호출처 0 — Geny 가 재구현 |
| `tools.adhoc` / `tools.scope` | 직렬화만 되고 소비자 없음 |
| CLI backend 의 temperature/top_p/max_tokens | `capabilities.drops` 에 선언만 — 소비자 0, 이벤트 0, argv 미반영 |
| schema 통과하는 inert 필드들 | s04 `fail_fast`/`max_chain_length`, s05 `cache_prefix`, s06 `timeout_ms`, s16 `max_turns`(multi_dim 컨트롤러에선 무효) |

> **판정**: 선언이 "받아들여지고, 저장되고, 검증 통과하고, 아무것도 안 하는" decoy 필드가
> 광범위함. 이는 v2.1.4 스트리밍 사건과 같은 *masked degradation* 계열 — 운영자는 환경을
> 편집했고 초록 체크를 봤는데 동작이 안 바뀐다.

### 1-2. "Stage-by-stage control" — **골격 real / 의미 aspirational**

- ✅ 21-slot layout, artifact loader, StrategySlot/SlotChain, manifest round-trip,
  session-less introspection 은 균일하고 UI-renderable (Geny Environment Builder 가
  adapter 없이 소비 중).
- ❌ 위 1-1 의 3-tier fidelity 문제.
- ❌ 라이브 pipeline 의 typed introspection 부재 → Geny 의 per-session heatmap 엔드포인트가
  **이중으로 깨진 채** (잘못된 kwarg 이름 → TypeError → fallback, 존재하지 않는 필드 getattr)
  fiction 을 렌더링 중인데 아무도 모름 (`agent_controller.py:1713-1737`).
- ❌ required-stage 강제가 라이브러리에 없음 — s06 inactive 인 manifest 가 strict 에서도
  통과해 "LLM 호출 없는 파이프라인"이 조용히 빌드됨. Geny 가 write-time 에 host-side 로 강제.
- ❌ `AdaptiveModelRouter`: thinking 켜면 무조건 Opus 승격, 모든 튜닝 knob 이
  constructor-only (manifest 도달 불가). 게다가 `_estimate_chars` 가 s05 의 block 변환 후
  `len(블록리스트)` 를 세서 size 휴리스틱이 망가져 있음.

### 1-3. "Hosts are thin consumers" — **미충족 (~3,800줄 보상 레이어)**

| Host | 보상 코드 | 원인이 된 라이브러리 공백 |
|---|---|---|
| Geny | `llm_patches.py` 479줄 (monkey-patch) | CLI-tool 이벤트/에러 envelope 가 event bus 에 없음 |
| Geny | `default_manifest.py` 728줄 + `stage_manifest.py` 440줄 + `backend_resolver.py` | preset→manifest factory 부재, "기본 백엔드" 질의 부재 |
| Geny | 이벤트→로거 브리지 ~600줄 ×2 (invoke/astream 중복, 이미 drift) | 공표된 이벤트 계약/멀티구독 tap 부재 → 50ms 폴링 구조 강제 |
| Geny | `queue_runtime_refresh` ~220줄 (private setter 우회) | between-turn runtime 갱신 API 부재 (`attach_runtime` 1회성 게이트) |
| GAPT | `executor_patches.py` (private 3개 교체: `_call_streaming`, `StreamJsonAccumulator.feed`, `CLIProcessRunner._spawn`) | 동일 (이벤트 egress + spawn 주입 seam 부재) |
| GAPT | 세션 persist/rehydrate ~700줄 (자체 이벤트 테이블에서 메시지 역산) | SessionStore/from_checkpoint 계약 부재 |
| GAPT | `pipeline._config.model.*` 직접 변조 + 수제 baseline/revert | per-run override API 부재 |
| GAPT | manifest 모델 설정 dual-write ("両쪽 다 써야 안전") | 모델 설정의 manifest 내 'home' 이 모호 + `from_dict` 가 잘못된 키를 조용히 버림 |

**결정적 부작용**: GAPT 는 `_call_streaming` fork 때문에 **2.1.0 에 고정**되어 있고,
2.1.1–2.1.4 의 vendor-drift 수정 4개가 전혀 전달되지 않았다. monkey-patch 가 업그레이드를
봉쇄하는 악순환.

### 1-4. "Robustness at the boundary" — **사후 대응만 존재, 구조적 방어 없음**

| 경계 | 현재 방어 | 평가 |
|---|---|---|
| Anthropic Messages | 정적 prefix 테이블 + `_retry_kwargs_after_deprecation` (이미 본 2개 클래스만) | 사후약방문. 다음 drift (max_tokens 개명 등)는 또 prod 에서 터짐 |
| Claude Code CLI | shape-tolerant 파서 + silent fallback | **2.1.4 마스킹 채널이 아직 열려있음** (§2.2) |
| OpenAI / Google | **0** — param 테이블 없음, retry 없음. Google 은 `str(e)` substring 으로 에러 분류 ('400' 포함 500 → BAD_REQUEST 오분류) | 다음 사건 후보 1순위. OpenAI 의 `max_tokens→max_completion_tokens` drift 는 이미 현실 |
| Embedding | **계약 자체가 없음** — 단일 generic 예외, env-var credential, circuit breaker 없음 | 라이브 401-spam 사건의 구조적 원인 |
| MCP | fail-fast FSM + phase-label 에러 — **5개 중 최상** | embedding 경계의 본보기로 쓸 것 |

추가로: **어느 경계도 version handshake 가 없다.** `claude --version` 을 읽는 코드가 0줄.
4개의 2.1.x 사건이 전부 version-skew 였는데, 사후 진단에 필요한 그 한 가지 사실을 어디에도
기록하지 않는다.

### 1-5. "Policy via config, not hardcode" — **위반**

- `permission/` 은 **allow-by-default** 이고 rule 미바인딩 시 통째로 스킵 — 선언된
  deny-by-default 와 반대.
- s11 reviewer 의 보안 설정 (allowed_hosts, destructive_tools, secret 패턴) 이
  constructor-only — 환경 편집으로 도달 불가, 사실상 hardcode.
- in-process hook 이 subprocess 보안 게이트 (`GENY_ALLOW_HOOKS`) 뒤에 묶여 있어 GAPT 가
  자기 policy engine 을 돌리려고 **보안 env var 를 위조**해야 함.

---

## 2. Critical — 지금 살아있는 문제 (검증 완료)

### 2.1 Geny prod 의 worker loop 의미 손상 (이번 검토에서 발견된 라이브 버그)

직접 검증함:
- Geny worker manifest: s14 `strategies={"strategy": "evaluation_chain"}` +
  `strategy_configs={"strategy": {"evaluators": ["binary_classify", "signal_based"], ...}}`
  (`Geny/backend/service/executor/default_manifest.py:445-455`)
- `EvaluationChain` 에 `configure()` 없음 → base no-op → evaluators 리스트 소실
- 빈 체인의 `evaluate()` → `decision="complete"` 무조건 반환
  (`s14_evaluate/artifact/default/strategies.py:260-268`)
- s16 은 upstream "complete" 를 terminal 로 처리

**결과: worker 세션이 [CONTINUE] 신호와 무관하게 1 iteration 에 종료될 수 있다.**
s16 의 `multi_dim_budget` dimensions 도 같은 경로로 빈 리스트가 된다. manifest 주석의
"strategy_configs 편집만으로 dimension 추가 — 코드 변경 불필요"는 현재 거짓.

### 2.2 v2.1.4 마스킹 채널이 아직 열려있음

`cli_unknown` / `cli_malformed` 신호는 파서가 생성하지만 **소비자가 0** (grep 검증).
s06 `_call_streaming` 은 `message_complete` 와 `text_delta` 만 전달. 다음 CLI wire 변경은
정확히 2.1.4 와 같은 방식으로 — 경고 없이, fallback envelope 에 가려진 채 — 몇 주간
조용히 저하된다. 탐지 비용은 이미 지불하고 있는데 아무에게도 알리지 않는 구조.

### 2.3 테스트가 자기 가정만 pin — fake CLI 가 아직도 옛 wire form 을 emit

- 모든 wire fixture 가 테스트 본문에서 발명한 dict (`json.dumps(invented)`) — 기록된
  실물 transcript 0개.
- `tests/_fixtures/fake_claude.py` 는 **pre-2.1.x delta form 만 emit** (`stream_event` 줄 0개).
  즉 **2.1.4 가 고친 바로 그 사건이 재발해도 3,276개 테스트 전부 통과한다.**
- `RUN_LIVE` canary 게이트는 정의만 있고 호출 테스트 0개.

### 2.4 Pipeline teardown 의 부재 — Geny 가 세션 종료마다 MCP subprocess 누수

`disconnect_all` 호출처는 빌드 실패 unwind 한 곳뿐. `Pipeline.aclose()` 가 없고,
Geny `cleanup()` 은 MemoryProvider 만 닫고 `self._pipeline = None` — mcp_servers 선언된
세션을 멈출 때마다 stdio MCP 자식 프로세스가 고아가 된다 (`agent_session.py:3806-3841`).

### 2.5 OpenAI streaming 호출 전부 $0 집계

`OpenAIClient._call_streaming` 이 `stream_options={"include_usage": True}` 를 요청하지
않음 → usage chunk 가 안 옴 → `TokenUsage()` 0 → s07 가격 $0. usage 수확 분기는 코드에
있는데 (작성자가 기대했는데) 요청 플래그만 빠진 1줄짜리 잠복 버그. CostBudgetGuard 와
두 host 의 비용 표시를 전부 무력화.

### 2.6 Embedding 401-spam (현재 prod 진행형) 의 구조 원인

embedding key 가 CredentialBundle 밖의 평행 채널 (LTMConfig → env var ladder) — bundle
docstring 의 "no other credential channel exists" 는 거짓. 에러 분류/회로차단기 없이
**매 turn 재시도 + traceback 로깅**. 같은 패키지의 MCP 는 이미 NEEDS_AUTH FSM 으로 동일
문제를 풀어놨음 — 미완 경계이지 설계 선택이 아님.

### 2.7 `attach_runtime(llm_client=)` — #866 풋건 구조 잔존

`_resolve_llm_client` 가 attached client 를 무조건 우선. 재발 방지책이 **Geny 안의 주석
한 줄** ("Intentionally NOT passing llm_client") 뿐. attach_runtime docstring 은 모든
kwarg 를 설명하면서 가장 위험한 llm_client 만 누락.

### 2.8 sub-agent provider 상속이 dead contract

`parent_state_shared['primary_provider']` 를 읽는 코드는 있는데 **쓰는 코드가 어느 repo 에도
없음** (grep 검증). `descriptor.provider=None` 은 항상 host 전역 휴리스틱으로 fall-through —
부모가 claude_code_cli 에 고정돼 있어도 sub-agent 는 다른 backend 로 갈 수 있다.
\#866 의 미스라우팅 클래스가 한 단계 아래에서 보장된 셈.

---

## 3. 구조적 발견 (중요도 순)

### 3.1 Config 채널 우선순위 모델의 부재

모델 선택만 해도 5+ 표면 (manifest top-level model block / manifest pipeline.model /
per-stage model_override / PipelineConfig stomp / mutator / state 직접 편집) 이 3가지
다른 수명 (영구 / 매 run 리셋 / 한 번 sticky) 으로 공존. GAPT 운영자의 실제 증언:
"opus 로 바꾼 줄 알았는데 manifest 모델이 계속 돌았다".

**제안된 단일 funnel**: per-run overrides (신규 public API) > session overrides/mutator >
attach_runtime 의 runtime-object (llm_client 강등) > manifest (설정당 home 1개) > 기본값.
\+ run 시작 시 `config.resolved` 이벤트로 각 필드의 값·출처 보고.

### 3.2 이벤트 시스템이 사실상의 host 계약인데 비공표

- 두 개의 분리된 채널 (EventBus vs `state.add_event`) — `pipeline.on()` 구독자는
  tool/api/text 이벤트를 영영 못 봄.
- taxonomy 가 untyped string — GAPT 가 이름을 추측하다 **전체 텍스트 100% 유실 버그**와
  **$0 비용 버그**를 실제로 출하했고, Geny 는 600줄 mapping switch 를 두 벌 (이미 drift) 유지.
- 멀티구독/replay tap 없음 → Geny 의 50ms 폴링 구조가 강제된 것.
- correlation 필드 (session_id/run id) 없음.

### 3.3 Session/state lifecycle 의 무소유

- `run_stream(input, state=None)` 이 조용히 fresh state 를 만들고 그 state 를 **돌려주지도
  않음** (GAPT prod 기억상실 사건의 원인).
- "fresh state per turn" (Geny) 도 "long-lived state" (GAPT) 도 공식 지원 모델이 아니며
  둘 다 불완전: 재사용 state 의 `iteration`/`loop_decision`/`events` 는 turn 간 리셋
  계약이 없어 장수 세션이 MAX_ITERATIONS 에 걸리거나 이전 turn 의 'error' 가 다음 turn 의
  success 판정을 오염시킴. state docstring 의 "Resets to {} at the start of each run" 은
  코드에 존재하지 않는 거짓말.
- 라이브러리 자체의 `session/` 패키지는 두 host 모두 import 0 (단, geny-executor-web 은
  사용 — §3.6).

### 3.4 Vendor 경계의 비대칭 (1-4 표 참조)

추가 디테일: Anthropic 의 TOKEN_LIMIT 휴리스틱 (`'token' in msg`) 이 라이브러리 자신이
문서화한 drift 에러 메시지 (`thinking.adaptive.budget_tokens: Extra inputs...`) 를
TOKEN_LIMIT 으로 오분류 → 다음 drift 가 "컨텍스트 줄이세요" 로 오진단될 예정.
비-streaming CLI 파서 (`parse_json_output_to_response`) 는 실물과 다른 발명된 envelope 을
기대 — top-level `result` 문자열/`total_cost_usd` 를 안 읽음. streaming accumulator 와
같은 CLI 출력을 두 파서가 다르게 해석.

### 3.5 죽은 표면 (decoy) 목록 — "wire it or delete it"

`HostSelections.resolve` (호출 0) / `tools.adhoc`·`tools.scope` (소비 0) /
`capabilities.drops` (소비 0) / `supports_session_continuity=True` (producer 0) /
`lock_stage/unlock_stage` (엔진 미사용 — MutationLocked 은 prod 에서 발화 불가) /
hook taxonomy 16개 중 3개만 발화 / `security/` 모듈 (live path 연결 없음) /
`StreamingToolExecutor` (export 되고 자기 stage 의 schema 설명에 등장하지만 registry 미등록).

### 3.6 geny-executor-web — 방치된 제3 소비자

마지막 커밋 2026-04-19 (2.0.0 이전), `geny-executor>=0.20.0` + 존재하지 않는 extras
(`[postgres]`, `[memory]`) 에 의존, "16-stage" README, 손으로 미러링한
`_REQUIRED_ORDERS = {1,6,9,16}` 가 현재 21-stage 와 불일치 (s16 을 required 로 강제하고
s21_yield 비활성을 허용 — strict from_manifest 가 거부할 manifest 를 생산 가능).
한편 이 repo 의 존재가 "dead code" 판정 일부를 뒤집음: `session/`, `ToolScopeManager`,
`ToolSandbox`, `PipelinePresets` 의 소비자가 여기 있다. **운명 결정 필요** (§5).

### 3.7 기타 확인 사항

- ✅ 보안 spot-check 클린: `ProviderCredentials.__repr__` 의 key redaction, CLI argv 에
  key 비노출 (자식 env 로만), dist/ git 위생, CI pip-audit.
- ⚠️ cancellation 테스트 0 (run_stream 소비자 이탈, CLI subprocess 고아화) — 두 host 모두
  SSE 서버라 client disconnect 가 일상인데, 다음 "아무도 몰랐던 사건" 1순위 후보.
- ⚠️ CHANGELOG 에 2.1.0 항목 누락 (GAPT 가 pin 한 바로 그 버전).
- ⚠️ `py.typed` 출하하면서 mypy/pyright CI 없음; ruff 의 `force-exclude` 가 tests/ 를
  조용히 lint 제외 중.
- ⚠️ pyproject 가 openai/google-genai/psycopg/pgvector/numpy/ddgs 를 전부 필수 의존으로 —
  "lazy optional" 은 import 만 lazy 고 설치는 강제. extras 제거가 CHANGELOG 미기재 계약 파괴.

---

## 4. 개선 로드맵

### Tier 0 — 핫픽스 (각각 독립 PR, 즉시)

| # | 항목 | 크기 |
|---|---|---|
| 0-1 | `EvaluationChain`/`MultiDimensionalBudgetController` 에 `configure()` 구현 → **Geny prod loop 의미 복구** | S |
| 0-2 | `cli_unknown`/`cli_malformed` 텔레메트리: 첫 발생 시 rate-limited warning + `llm_client.unknown_wire_shape` 이벤트 + APIResponse.raw 에 카운트 | S |
| 0-3 | OpenAI `stream_options={"include_usage": True}` + streamed-usage conformance 테스트 (모든 provider) | S |
| 0-4 | `Pipeline.aclose()` (MCP disconnect + tool-provider shutdown) + Geny cleanup 에서 호출 | S |
| 0-5 | fake CLI 에 `stream_event` 시나리오 추가 + 실물 기록 golden fixture 디렉토리 (`tests/llm_client/golden/`, CLI 버전 스탬프 포함) | M |
| 0-6 | `attach_runtime(llm_client=)` 가드: manifest provider 와 다르면 경고 (strict: 에러), docstring 에 우선순위 명기 | S |
| 0-7 | TOKEN_LIMIT 휴리스틱 anchor 강화 (param-path 형태 메시지 제외) | S |
| 0-8 | sub-agent 상속 producer: `_init_state` 가 `SharedKeys.PRIMARY_PROVIDER` 를 기록 + `SubAgentBuildContext.parent_provider` typed 필드 | S |

### Tier 1 — 구조 (Big 3, leverage 순 / 각각 minor 버전)

1. **Streaming event contract** (host 코드 ~1,800줄 삭제 효과)
   - s06 이 전체 canonical chunk (tool_use, thinking_delta, input_json_delta,
     content_block_stop) 를 state 이벤트로 전달
   - StreamJsonAccumulator 가 CLI-dispatch 도구를 `tool.call_start/complete(source='cli')` 로,
     is_error envelope 을 구조화된 `exec.cli.*` APIError 로 발화
   - versioned EventTypes 카탈로그 (enum + payload schema) + session/run correlation
   - 멀티구독 cursor tap (`pipeline.events()`) — Geny 폴링과 양쪽 monkey-patch 모듈 제거
   - `ClaudeCodeCLIClient(runner_factory=...)` spawn 주입 seam — GAPT 의 sandbox patch 를 설정으로
2. **Manifest 를 진짜 source of truth 로** (~1,300줄 삭제 효과)
   - `build_manifest(preset, *, provider, model, tools) -> EnvironmentManifest` 공개 factory
   - `CredentialBundle.preferred_provider()` — "기본 backend" 질의의 라이브러리 소유
   - 공개 `validate_manifest() -> list[ManifestIssue]`: 미소비 필드/unknown strategy/
     misplaced key 를 strict 에서 에러로
   - strategy `configure()`+`config_schema()`+`get_config()` 의무 트리오 + strict 검증
   - 설정당 manifest home 1개 선언, `from_dict` 가 unknown/misplaced 키에 경고
   - required-stage 를 strict from_manifest 에서 강제 (Geny 의 host-side 강제 삭제)
3. **Session lifecycle 소유** (~800줄 삭제 효과)
   - `state=None` 인데 prior state 가 있으면 경고 (strict: 에러); 생성된 state 를 결과에 노출
   - `begin_turn()` per-turn 리셋 계약 (loop_decision/iteration/events)
   - SessionStore persister 계약 (messages 저장/로드, host 는 backend 만 공급) +
     `Pipeline.from_checkpoint`
   - `pipeline.refresh_runtime(**kwargs)` between-turn 합법 + run-in-progress lock 실구동
   - `run_stream(..., overrides=ModelOverrides)` per-run override (GAPT 의 baseline/revert 흡수)

### Tier 2 — 경계 대칭화 + 철학 완성

- retry-on-heal 의 일반화 (BaseClient 로 승격, 메시지 needle 대신 구조화된 param-path 파싱,
  OpenAI `max_tokens→max_completion_tokens` 부터) + `llm_client.drift_healed` 이벤트
- version handshake: `claude --version` 1회 캡처 → 로그/APIResponse.raw/APIError context;
  MCP protocolVersion 기록
- embedding 경계를 MCP 패턴으로: bundle 의 embedding 채널 + auth FSM + trip-once
  `memory.vector_disabled`
- `ProviderCredentials.auth_mode` (api_key/oauth/setup_token) — env sniffing 삭제, CLI 인증
  lifecycle 헬퍼 (credentials.json 읽기/만료 검증) 라이브러리로
- sub-agent 의 manifest 1급화 (`subagents` 섹션: agent_type → env_id|inline manifest)
- memory 섹션의 manifest 1급화 (MemoryProviderFactory 매핑)
- permission deny-by-default + manifest 표현 + ASK→s15 HITL 라우팅
- in-process hook 을 subprocess 게이트에서 분리
- decoy 표면 일괄 정리 (wire or delete) + config-liveness 테스트 (모든 ConfigField 가
  execute-path 에 영향을 줌을 자동 검증)
- RUN_LIVE canary (nightly): alias 모델 + Opus+thinking+temperature + streaming delta —
  drift 를 release 전 failing test 로
- cancellation 테스트 스위트 + mypy CI + extras 복원 + CHANGELOG 2.1.0 backfill

---

## 5. 함께 결정할 사항

1. **geny-executor-web 의 운명** — (a) 폐기 선언 (그러면 session/, ToolScope/Sandbox,
   PipelinePresets 의 삭제 추천이 유효해짐) vs (b) canonical introspection/mutation UI 로
   승격 (2.1.4 업그레이드 + 손미러 상수를 introspection API 로 교체). 이 결정이 dead-code
   정리 범위를 좌우한다.
2. **Tier 0 착수 순서** — 제안: 0-1 (Geny prod 의미 버그) 즉시, 0-2~0-4 동일 주, 나머지 순차.
3. **strict mode 의 의미 강화 호환성** — "unknown strategy/unconsumed config = 에러" 로
   바꾸면 기존 저장 manifest 일부가 깨질 수 있음. strict 의 단계적 강화 (warning 릴리스 →
   에러 릴리스) 인지, `validate_manifest()` 별도 API 로 시작인지.
4. **이벤트 계약의 버전 정책** — EventTypes 를 semver 표면에 포함할지 (포함하면 이름 변경이
   major), correlation 필드 추가가 기존 host switch 에 미치는 영향.
5. **GAPT 2.1.0 탈출 계획** — Tier 1-1 (이벤트 contract + runner_factory) 가 끝나야
   executor_patches.py 를 지우고 업그레이드 가능. 그 전에 부분 백포트가 필요한지.

---

## 부록 — 검증된 강점 (지킬 것)

- manifest provider-location fail-fast 검증, attach_runtime 시작 후 hard-error
- 이중 차원 에러 분류 (ErrorCategory × ExecutorErrorCode) 와 안정성 계약
- v2 snapshot 의 포괄성 (slot/config/artifact 출처/chain/tool_binding/model_override) +
  원자적 batch rollback
- MCP bring-up 의 부분 실패 unwind, fail-fast FSM (5개 경계 중 최상 — 본보기)
- session-less introspection 의 정직한 per-stage capability map (grep 으로 검증된
  tool_binding/model_override 소비 여부)
- stage 12–21 의 안전한 no-op 기본값 규율 (NullRequester/NoPersister/NoSummarizer)
- credential redaction, argv 비노출, env whitelist — 보안 spot-check 전부 클린
- 3,276개 테스트의 절대량과 2.1.1–2.1.4 의 신속한 사후 대응 자체
