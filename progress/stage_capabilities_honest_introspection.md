# L1.B Hotfix — Honest per-stage capability flags (v0.13.1 → v0.13.2)

> **Completed**: 2026-04-17
> **Scope**: Fix a correctness gap in `StageIntrospection`: every stage
> was reporting `tool_binding_supported=True` and
> `model_override_supported=True` regardless of whether the runtime
> actually consumes those fields. Downstream UIs (geny-executor-web
> Environment Builder) rendered Tool/Model editors on every stage and
> then had to paper over it with "this stage does not support tool
> bindings" banners, which is a misleading surface.
> **Target tag**: `v0.13.2`

---

## Summary

`StageIntrospection.tool_binding_supported` and
`StageIntrospection.model_override_supported` now reflect the stage's
**actual runtime behaviour**, not the dataclass plumbing shape.

Runtime truth (verified by grepping `self.tool_binding` /
`self.model_override` reads across `src/geny_executor`):

- `s06_api` is the only stage that reads `self.model_override` (in
  `_build_request` — it overrides model name / max_tokens / sampling /
  thinking).
- `s10_tool` is the only stage that reads `self.tool_binding` (in
  `execute` — it enforces the per-stage allow/block list).
- Every other stage reads neither; the fields are persisted and
  restored by the manifest layer but silently ignored at run time.

Alternative artifacts (`s06_api/openai`, etc.) inherit the same
capability because they're still the LLM-call / tool-call stage.

## Changes

**`src/geny_executor/core/introspection.py`**

- Added `_STAGE_CAPABILITIES: Dict[str, Dict[str, bool]]` — a single
  source of truth keyed by stage module name, mirroring the existing
  `_STAGE_INTROSPECTION_KWARGS` pattern above it.
- Added `_stage_capabilities(stage_module)` helper — returns the
  `(tool_binding, model_override)` flags for a stage. Unknown stages
  default to both-False (safe under-promise vs misleading
  over-promise).
- `introspect_stage` now reads `caps = _stage_capabilities(...)` and
  plumbs them into the returned `StageIntrospection`.
- `StageIntrospection` dataclass defaults for the two flags changed
  from `True` → `False` so constructing one by hand without passing
  flags picks the safe default.

## Tests

**`tests/unit/test_introspection.py`**

- Removed the blanket "every stage must report both flags True"
  assertions (they were asserting the bug, not the contract).
- Added three pinning tests:
  - `test_capability_flags_api_stage_only_allows_model_override` —
    `s06_api` must have `model_override_supported=True` and
    `tool_binding_supported=False`.
  - `test_capability_flags_tool_stage_only_allows_tool_binding` —
    `s10_tool` must have `tool_binding_supported=True` and
    `model_override_supported=False`.
  - `test_capability_flags_default_false_elsewhere` — parametrised
    over the other 14 stages; both flags must be False. This is the
    load-bearing test — it prevents a future regression from
    quietly returning to "everyone claims everything".

## Compatibility

- **Manifest / snapshot on-disk format unchanged.** The stage-level
  `tool_binding` and `model_override` fields are still persisted and
  restored exactly as before. Only the introspection surface changed.
- **Runtime behaviour unchanged.** Nothing new is *enforced* — a user
  who manually sets `tool_binding` on s03_log still gets it
  round-tripped, the runtime just won't read it (same as before).
- **UI-facing contract tightened.** Callers that keyed UI affordances
  on `tool_binding_supported` / `model_override_supported` will now
  see the honest answer and can hide Tool / Model tabs on stages
  that don't consume them.

## Version bumps

- `pyproject.toml`: `0.13.1` → `0.13.2`
- `src/geny_executor/__init__.py`: `__version__ = "0.13.2"`

## Follow-up (not in this change)

- `geny-executor-web` v0.8.3 should bump its pin to `>=0.13.2` and
  gate the Environment Builder's Tool / Model / Chain tabs on the
  new honest flags (plus a non-empty `strategy_chains` check for
  the Chain tab). Tracked separately in the web repo.
