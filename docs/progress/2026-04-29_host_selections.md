# 2026-04-29: HostSelections — 훅/스킬/권한의 host-등록 + env-선택 패턴

## What changed

### geny-executor 1.3.3
- `HostSelections` dataclass 신설 (hooks/skills/permissions 각각 `["*"]`/`[]`/이름 리스트)
- `EnvironmentManifest.host_selections` 필드 추가 + serialize/deserialize 지원
- `blank_manifest()`이 host_selections 기본값(`["*"]` × 3)을 명시적으로 세팅
- `HostSelections.resolve(selection, available)` 헬퍼: 와일드카드/empty/literal 3-mode 해결
- 프리-1.3.3 payload (host_selections 필드 없음)는 와일드카드 기본값으로 로드 → forward-compat
- `geny_executor.HostSelections` top-level export
- 9개 신규 테스트 + 24개 기존 테스트 모두 통과

### Geny frontend
- `EnvironmentManifest.host_selections?: HostSelections` 타입 추가
- `useEnvironmentDraftStore.patchHostSelections()` + `hostSelectionsDirty` flag
- `HostEnvSelectionPicker` — 재사용 picker (와일드카드/전체선택/전체해제 3-mode + 검색 + 배지)
- `HookEnvPicker` / `SkillEnvPicker` / `PermissionEnvPicker` — 3 영역별 wrapper
- `GlobalSettingsView` 갱신:
  - 사이드바 "호스트 단위 (공용)" 그룹 제거 → 훅/권한/스킬을 "환경 단위" 그룹으로 통합
  - 각 패널 상단: env-side picker / 하단: collapsible `HostRegistryEditor`로 기존 HooksTab/SkillsTab/PermissionsTab 감싸기
  - 사이드바 배지: 와일드카드 `★` / literal count

## Why

사용자 요구: "훅/권한/스킬 모두가 MCP처럼 — 등록은 host 공통, 사용 여부는 env마다 체크박스로". 현재 Library/Builder의 "호스트 단위 (공용)" 패널은 모든 env에 동일 적용되어 dev/prod 환경 분리가 불가능했음.

권한은 의도적으로 mockup: 사용자가 명시적으로 "현재는 사용하지 않을 기능"이라고 지정. UI shape는 manifest forward-compat을 위해 노출, runtime enforcement는 미래 작업.

기본값 `["*"]` (전체) 채택: 사용자가 "나중에 일부 제외하는 기본값으로 변경할 것"이라고 명시 — wildcard sentinel은 그 변경을 매니페스트 schema 변경 없이 지원.

## How validated

- executor: `tests/unit/test_manifest_v2.py` 32 pass (9 신규)
- executor: smoke script로 wildcard / empty / literal-with-stale 3-mode resolve 검증
- executor: pre-1.3.3 legacy payload 로드 시 wildcard 기본값 적용 확인
- frontend: 빌드/타입체크는 user 환경에서 (이 dev env에 node_modules 없음)

## Follow-ups

1. **Geny pin bump**: Geny가 `geny-executor>=1.3.1,<1.4.0`이므로 1.3.3 자동 호환, 단 venv 재설치 필요
2. **Runtime enforcement (hooks/skills)**: env load 시 `host_selections.resolve()`로 host 레지스트리 필터링 — 별도 작업 (현재 schema와 UI만 in-place)
3. **Permissions enforcement**: 미래 작업. UI 이미 forward-compat이므로 manifest schema 변경 없이 enforcement만 추가 가능
4. **사용자 예고된 default 변경**: "나중에 몇 개 제외한 default" 적용 시점에 wildcard → literal list 전환 — schema는 이미 지원
5. **i18n**: 신규 picker 텍스트 ("이 환경에서 사용할 …", "와일드카드", "호스트 등록소 편집") 현재 하드코딩, ko.ts에 이관 검토
