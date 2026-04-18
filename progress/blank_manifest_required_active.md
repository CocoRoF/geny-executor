# L1.C follow-up — Blank manifest defaults required stages active (v0.13.3 → v0.13.4)

> **Completed**: 2026-04-18
> **Scope**: `EnvironmentManifest.blank_manifest()` hardcoded
> `active=False` for all 16 stages, so every freshly-created blank
> environment was born broken — it couldn't run even once without the
> user manually flipping on Input / API / Parse / Yield. Fix the
> default so blank envs come out the gate runnable.
> **Target tag**: `v0.13.4`

---

## The bug

v0.13.3 added `StageIntrospection.required` and shipped the correct
set of structurally-required stages, but `blank_manifest()` did not
consume the flag. The relevant line:

```python
# src/geny_executor/core/environment.py:317 (pre-fix)
entry = StageManifestEntry(
    order=insp.order,
    name=insp.name,
    active=False,  # ← always inactive, including the required four
    ...
)
```

Downstream symptom, as reported in geny-executor-web: after running
`POST /api/environments` with `mode="blank"`, the returned manifest
had every stage inactive. The web Environment Builder's
`coerceRequiredStagesActive()` only ran on `loadTemplate()` for
existing envs — newly-created envs went straight from the backend
into the same load path, but the user saw their required stages
already toggled off and got no clear signal that creation had
silently produced a non-runnable template.

## Fix

Single line change, plus docstring:

```python
# src/geny_executor/core/environment.py (post-fix)
entry = StageManifestEntry(
    order=insp.order,
    name=insp.name,
    active=insp.required,  # required stages default on; others off
    ...
)
```

This makes blank mode structurally equivalent to the `minimal` preset:
s01_input + s06_api + s09_parse + s16_yield active, everything else
inactive. Users who want more stages flip them on; users who want a
truly bare skeleton get one that actually runs.

`from_snapshot()` is untouched — it already copies `active` from the
incoming snapshot, and both `from_session` and `from_preset` go
through `from_snapshot()` with snapshots that already have the
required stages on.

## Changes

**`src/geny_executor/core/environment.py`**

- `blank_manifest()`: changed `active=False` → `active=insp.required`.
- Docstring rewritten to describe the new default ("required four on,
  other twelve off") and cite `_STAGE_REQUIRED` as the source of truth.

**`tests/unit/test_manifest_v2.py`**

- Replaced `test_blank_manifest_returns_all_16_stages_inactive` with:
  - `test_blank_manifest_returns_16_stages_with_required_ones_active`
    — asserts the active set is exactly `{1, 6, 9, 16}`.
  - `test_blank_manifest_optional_stages_default_inactive` — asserts
    the inactive set is exactly the complement (`{2,3,4,5,7,8,10,
    11,12,13,14,15}`). Guards against a regression that marks every
    stage required.
- Updated `test_blank_manifest_builds_empty_pipeline_via_from_manifest`
  (renamed to `..._builds_minimal_pipeline_via_from_manifest`) — the
  new expectation is the four required stages, matching `minimal`.
- Updated `test_blank_manifest_activation_then_rebuild_succeeds`
  (renamed to `..._extra_activation_then_rebuild_succeeds`) — flips
  the optional cache stage on and asserts the rebuilt pipeline holds
  the required four plus stage 5.

## Compatibility

- **Existing envs on disk are not rewritten.** Only new blank-mode
  creations pick up the new defaults. Envs saved before v0.13.4 keep
  whatever `active` flags they were saved with — users can fix them
  in the Environment Builder (the web v0.8.5 `coerceRequiredStagesActive`
  already flips required stages on during `loadTemplate`).
- **Manifest format unchanged.** The on-disk shape is identical; only
  the default values one specific constructor writes changed.
- **Runtime behaviour unchanged.** `Pipeline.from_manifest` still
  filters by `.active` alone. This change is about making `blank_manifest`
  produce a sensible default, not about introducing runtime enforcement.

## Version bumps

- `pyproject.toml`: `0.13.3` → `0.13.4`
- `src/geny_executor/__init__.py`: `__version__ = "0.13.4"`

## Follow-up (separate PR in geny-executor-web)

- Bump pin: `geny-executor>=0.13.3` → `>=0.13.4`
- Bump web: `0.8.6` → `0.8.7`
- CHANGELOG entry describing the user-visible behaviour change
  (blank envs now arrive with Input/API/Parse/Yield already on).
