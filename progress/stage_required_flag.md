# L1.C Hotfix — Structurally required stages (v0.13.2 → v0.13.3)

> **Completed**: 2026-04-17
> **Scope**: Expose a stage-level `required` flag so UIs can distinguish
> truly optional plumbing from the four stages every pipeline must keep
> active. Downstream UIs (geny-executor-web Environment Builder) were
> allowing users to deactivate every stage — including Input / API /
> Parse / Yield — which produces manifests that cannot build a working
> pipeline.
> **Target tag**: `v0.13.3`

---

## Summary

`StageIntrospection` now carries a boolean `required` field. Four
stages are marked `required=True`; the other twelve remain optional.

Required set (mirrors the `minimal` PipelineBuilder preset —
Input → API → Parse → Yield — which is the smallest canonical
configuration):

- `s01_input` — turns the user prompt into the initial artifact. No
  input, no pipeline run.
- `s06_api` — the LLM call. Removing it leaves nothing to parse or
  emit.
- `s09_parse` — converts raw API output into the typed events every
  downstream stage (tool, loop, memory, yield) consumes.
- `s16_yield` — surfaces the final result to the caller. Without it
  the run produces no output.

Everything else (`s02_context`, `s03_system`, `s04_guard`,
`s05_cache`, `s07_token`, `s08_think`, `s10_tool`, `s11_agent`,
`s12_evaluate`, `s13_loop`, `s14_emit`, `s15_memory`) is optional and
may be toggled off.

## Changes

**`src/geny_executor/core/introspection.py`**

- Added `_STAGE_REQUIRED: Set[str]` — a single source of truth keyed
  by stage module name, mirroring the existing `_STAGE_CAPABILITIES`
  pattern.
- Added `_stage_required(stage_module)` helper.
- Added `required: bool = False` field to the `StageIntrospection`
  dataclass. Defaults to `False` so any stage not in `_STAGE_REQUIRED`
  is optional by construction.
- `StageIntrospection.to_dict()` serializes the new field.
- `introspect_stage` plumbs `required=_stage_required(module_name)`
  into the returned dataclass.

## Tests

**`tests/unit/test_introspection.py`**

- Added `test_required_flag_true_for_structurally_required_stages`
  parametrised over the 4 required stages — each must report
  `required=True`.
- Added `test_required_flag_false_for_optional_stages` parametrised
  over the other 12 stages — each must report `required=False`. This
  is the load-bearing test — it prevents a future regression from
  quietly marking every stage as required.
- Added `test_required_flag_serializes_in_to_dict`.

## Compatibility

- **Manifest / snapshot on-disk format unchanged.** The new field
  lives on `StageIntrospection` (a UI-facing projection), not on
  `EnvironmentManifest`. Existing manifests load and save identically.
- **Runtime behaviour unchanged.** `Pipeline.from_manifest` still
  filters by `.active` alone — no enforcement is introduced here.
  Enforcing `required=True` is a UI concern, not a runtime one.
- **UI-facing contract extended.** Callers that want to forbid
  toggling Input / API / Parse / Yield off can now read
  `StageIntrospection.required` and disable the Active control
  accordingly.

## Version bumps

- `pyproject.toml`: `0.13.2` → `0.13.3`
- `src/geny_executor/__init__.py`: `__version__ = "0.13.3"`

## Follow-up (not in this change)

- `geny-executor-web` v0.8.5 should bump its pin to `>=0.13.3`, read
  `required` in the Environment Builder's StageCard to disable the
  Active checkbox, and auto-correct manifests that saved a required
  stage as `active=false`. Tracked separately in the web repo.
