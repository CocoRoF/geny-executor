# L1 Hotfix — `EnvironmentManifest.blank_manifest` (v0.13.0 → v0.13.1)

> **Completed**: 2026-04-17
> **Scope**: Fix a usability gap in the v0.13.0 manifest contract: there is
> no session-less way to produce a well-formed 16-stage template, so the
> geny-executor-web Environment Builder fell back to
> `PipelineSnapshot(pipeline_name=name)` — which has **zero** stages —
> resulting in an empty left pane and a spurious `base_preset: "<name>"`
> value leaking through `from_snapshot`.
> **Target tag**: `v0.13.1`

---

## Summary

Add `EnvironmentManifest.blank_manifest(name, *, description, tags, model,
pipeline)` — a classmethod that builds a complete 16-row template with
every stage **inactive** but already seeded with its default artifact and
that artifact's default strategy selections and config.

Behaviour:

- Uses `introspect_all()` (session-less) to enumerate the 16 canonical
  stages in order — no live `Pipeline` required.
- Every entry has `active=False`, so `Pipeline.from_manifest` on the raw
  blank manifest registers **zero** stages (as intended — the user must
  opt stages in via the UI first).
- Unlike `from_snapshot`, `metadata.base_preset` is left as the empty
  string. A blank environment has no origin preset.
- Each entry carries the artifact's `current_impl` per strategy slot so a
  toggle-to-active flip produces a manifest that `from_manifest` can
  rehydrate without "missing required strategy" errors.

## Rationale

`geny-executor-web v0.8.0` shipped an Environment Builder whose "Create
blank" path was implemented as:

```python
EnvironmentManifest.from_snapshot(
    PipelineSnapshot(pipeline_name=name),
    name=name, description=description, tags=tags,
)
```

Two defects fell out of this:

1. An empty `PipelineSnapshot` carries **no** stages, so the manifest was
   produced with `stages=[]` — the builder's 16-row `StageList` had
   nothing to render.
2. `from_snapshot` assigns `metadata.base_preset = snapshot.pipeline_name`
   unconditionally, so the environment's name was silently shadowed onto
   the preset field.

Both defects are properties of the library's `from_snapshot` contract, so
the right fix is a library-side helper rather than open-coding stage
seeding in the web service.

## Tests

Added to `tests/unit/test_manifest_v2.py`:

- `test_blank_manifest_returns_all_16_stages_inactive`
- `test_blank_manifest_uses_default_artifact_per_stage`
- `test_blank_manifest_seeds_strategy_current_impls`
- `test_blank_manifest_metadata_has_no_base_preset`
- `test_blank_manifest_accepts_optional_tags_model_pipeline`
- `test_blank_manifest_roundtrips_through_json`
- `test_blank_manifest_builds_empty_pipeline_via_from_manifest`
- `test_blank_manifest_activation_then_rebuild_succeeds`

Full unit suite: `462 passed, 1 skipped`.

## Downstream

`geny-executor-web` switches `EnvironmentService.create_blank` to use
`EnvironmentManifest.blank_manifest(...)` when `base_preset` is not
supplied, and pins `geny-executor >= 0.13.1`.

---

### Version bump

- `pyproject.toml`: `0.13.0` → `0.13.1`
- `src/geny_executor/__init__.py`: `__version__ = "0.13.1"`
