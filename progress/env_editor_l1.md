# L1 — Environment Builder Library Extensions (v0.12.0 → v0.13.0)

> **Completed**: 2026-04-17
> **Scope**: Lift Environment from runtime-only snapshot to a first-class,
> editable template. Add artifact metadata, session-less introspection,
> manifest v2, and `Pipeline.from_manifest`.
> **Target tag**: `v0.13.0`

---

## Summary

v0.13.0 adds the library contracts that the Environment Builder UI in
geny-executor-web needs. Every change is **additive**: v0.12.0 consumers
(notably `geny-executor-web v0.7.1` and Geny) remain binary-compatible.
The library can now:

1. Describe every artifact on disk with metadata (`ArtifactInfo`,
   `describe_artifact`, `list_artifacts_with_meta`).
2. Report each stage's schema and chain layout without instantiating a
   live Pipeline (`introspect_stage`, `introspect_all`).
3. Serialize per-stage template state — tool bindings, model overrides,
   chain ordering — through both `PipelineSnapshot` and
   `EnvironmentManifest`.
4. Reconstruct a runnable Pipeline directly from a saved manifest
   (`Pipeline.from_manifest`).

Existing v1 manifests on disk load unchanged — `from_dict` silently
migrates them to v2 shape.

---

## Deliverables

### L1.A — Artifact metadata + stage provenance
- `ARTIFACT_META_ATTR = "ARTIFACT_META"` convention for optional module-level
  metadata (description/version/stability/requires).
- `ArtifactInfo` frozen dataclass with `provides_stage` auto-detected from
  `Stage is None` — surfaces strategy-only artifacts like
  `s12_evaluate/adaptive` cleanly.
- `describe_artifact` / `list_artifacts_with_meta` for UI catalogs.
- `create_stage()` stamps `_artifact_name` and `_stage_module` on the
  instance; `Stage.artifact_name` / `Stage.stage_module` expose these.
- `StageToolBinding.to_dict`/`from_dict` and `ModelConfig.to_dict`/
  `from_dict` round-trip both types for manifest inclusion.

Tests: `tests/unit/test_artifact_metadata.py` covers ArtifactInfo
defaults, meta-dict parsing, provenance, bad meta rejection, and
serialization round-trips.

### L1.B — Session-less introspection
- `core/introspection.py` (new) — `SlotIntrospection`, `ChainIntrospection`,
  `StageIntrospection` dataclasses with JSON-safe `to_dict()`.
- `introspect_stage(stage, artifact="default")` builds a stage via
  `create_stage`, inspects `get_strategy_slots()` / `get_strategy_chains()`,
  and returns a typed description.
- `introspect_all()` enumerates all 16 stages in order and falls back to
  `"default"` when an artifact raises `IntrospectionUnsupported`
  (strategy-only artifacts).
- Internal `_STAGE_INTROSPECTION_KWARGS` injects `MockProvider` /
  dummy-key credentials so API stages can be introspected without a real
  pipeline session.

Tests: `tests/unit/test_introspection.py` parametrizes across all 16
stages and exercises alias/order resolution, schema/chain parity,
OpenAI/Google dummy-key paths, strategy-only raising, and JSON safety.

### L1.C — PipelineSnapshot v2 + EnvironmentManifest v2 + silent v1 migration
- `StageSnapshot` gains `artifact` / `tool_binding` / `model_override` /
  `chain_order` with safe defaults. Existing v1 snapshots load unchanged.
- `PipelineSnapshot.version` default bumps to `"2.0"`.
- `PipelineMutator.snapshot()` captures the new per-stage state;
  `restore()` rehydrates it. Restore treats `None` tool_binding /
  model_override as *not captured* — v1 snapshots never wipe live
  overrides already set on the target pipeline.
- `MANIFEST_VERSION = "2.0"`, `StageManifestEntry` dataclass, and
  `_migrate_v1_to_v2` silent upgrade path on `EnvironmentManifest.from_dict`.
- `from_snapshot` / `to_snapshot` round-trip v2 fields; `stage_entries` /
  `set_stage_entries` helpers give callers a typed view.

Tests: `tests/unit/test_manifest_v2.py` covers v2 field defaults,
snapshot round-trip, v1 silent load, manifest migration, v2 idempotent
round-trip, stage-entries helpers, mutator snapshot/restore, v1
restore preserving live overrides, and JSON UTF-8 validity.

### L1.D — `Pipeline.from_manifest`
- `PipelineConfig.to_dict` / `from_dict` with nested `ModelConfig` and
  unknown-key tolerance.
- `Pipeline.from_manifest(manifest, *, api_key=None, strict=True)`
  classmethod:
  1. Builds `PipelineConfig` from `manifest.pipeline` + `manifest.model`
     (caller's `api_key` kwarg wins).
  2. Instantiates every `active` stage via `create_stage(name, artifact)`
     with injected credentials for `s06_api` default/openai/google.
  3. Runs `PipelineMutator.restore(manifest.to_snapshot())` to apply
     strategies, configs, chain ordering, tool bindings, and model
     overrides.
  4. `strict=True` — validate each stage config against its
     `ConfigSchema` and surface instantiation errors. `strict=False` —
     silently drop stages that fail to construct.

Tests: `tests/unit/test_pipeline_from_manifest.py` covers round-trip
fidelity, chain ordering preservation, artifact routing, strict-mode
error surfacing, non-strict graceful degradation, and v1 payload
migrate→build.

---

## New top-level exports (`geny_executor.__init__`)

- `ArtifactInfo`, `describe_artifact`, `list_artifacts_with_meta`.
- `ChainIntrospection`, `IntrospectionUnsupported`, `SlotIntrospection`,
  `StageIntrospection`, `introspect_all`, `introspect_stage`.
- `StageManifestEntry`.
- Existing `Pipeline`, `PipelineConfig`, `ModelConfig`, `PipelineMutator`,
  `PipelineSnapshot`, `StageSnapshot`, `EnvironmentManifest` surface
  unchanged in signature.

---

## Backward compatibility

- v1 manifest JSON loads via `EnvironmentManifest.from_dict` and silently
  migrates on save. Consumers that round-trip on disk upgrade their
  files automatically.
- v1 `PipelineSnapshot.from_dict` payloads load as declared (version
  string is preserved), with v2-only fields defaulting in memory.
- `PipelineMutator.restore` with a v1 snapshot no longer wipes live
  tool bindings / model overrides set on the target — the absence of
  the fields is read as "not captured," not "cleared."
- All existing tests in `tests/contract/test_stage_uniformity.py` pass
  unchanged.

---

## Acceptance checklist

- [x] `list_artifacts_with_meta("s06_api")` → `ArtifactInfo(default, openai, google)`.
- [x] `introspect_stage("s10_tool", "default")` → `StageIntrospection` with
  slot-level `impl_schemas`.
- [x] `introspect_all()` → 16 ordered entries.
- [x] v1 fixture JSON → `EnvironmentManifest.from_json(...)` → v2 structure.
- [x] Pipeline → manifest → `Pipeline.from_manifest(manifest)` round-trip
  preserves artifact / tool_binding / model_override / chain_order.
- [x] `tests/contract/test_stage_uniformity.py` still green.
- [x] `geny-executor-web v0.7.1` imports (`PipelinePresets`,
  `PipelineMutator`, etc.) unchanged.
