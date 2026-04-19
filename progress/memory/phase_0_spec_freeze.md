# Phase 0 — Spec Freeze (memory initiative)

> **Started**: 2026-04-19
> **Owner**: memory initiative
> **Target tag**: `v0.14.0-spec` (no runtime change yet — docs + red tests only)
> **Predecessor doc**: `geny-executor-web/docs/MEMORY_ARCHITECTURE.md`
> **Gate**: G0 — Spec YAML present, red pytest scenarios C1–C7 present.

---

## Summary

Phase 0 freezes the **measurement surface** that the rest of the
initiative will be evaluated against. No runtime code changes. The
deliverables are:

1. `docs/MEMORY_SPEC.yaml` — machine-verifiable rendering of
   §1.2 R-A through R-F (layers, retrieval order, capability matrix,
   21 config fields). Subsequent phases verify their `MemoryDescriptor`
   and `ConfigSchema` against this file.
2. `tests/completeness/` — pytest scenarios C1–C7 (§3.6). They start
   red (`pytest.skip` with the reason "phase 0 — pending Phase 1
   provider"); each phase gate flips a subset to green.
3. Progress tracker for the initiative.

## Changes

### `progress/memory/INDEX.md` (new)
- Master tracker for the 6 phases (0–5) plus Phase G.
- Records gate definitions and completeness criteria summary.

### `progress/memory/phase_0_spec_freeze.md` (this file, new)
- Phase 0 record.

### `docs/MEMORY_SPEC.yaml` (new)
- Single source of truth for the 4-axis memory model.
- Sections: `layers`, `capabilities`, `scopes`, `backends`,
  `retrieval` (6-layer composition order), `events` (typed schema
  from §3.5), `config_fields` (21 fields from `LTMConfig`),
  `requirements` (R-A through R-F mapped to executor entry points),
  `completeness_criteria` (C1–C7 with acceptance text).
- Future phases must keep `MemoryDescriptor.config_schema` superset
  of `config_fields`.

### `tests/completeness/` (new)
- `__init__.py`
- `conftest.py` — loads `MEMORY_SPEC.yaml`, exposes `spec` fixture,
  exposes `provider_kind` parametrize point that Phase 2 will fill in.
- `test_c1_six_layer_retrieval.py` through
  `test_c7_adapter_parity.py` — one file per criterion.
- All tests `pytest.skip` initially with reason
  "phase 0 — pending phase N implementation". This is the **red** state.
- A meta-test `test_spec_loads.py` is green from day 1 — it asserts
  the spec YAML parses, has the expected top-level sections, and that
  every `requirements.*` entry references a known capability/layer.

## Tests

- `tests/completeness/test_spec_loads.py` — green.
  - Loads `docs/MEMORY_SPEC.yaml`.
  - Asserts top-level keys: `version`, `layers`, `capabilities`,
    `scopes`, `backends`, `retrieval`, `events`, `config_fields`,
    `requirements`, `completeness_criteria`.
  - Asserts every entry in `requirements` references a layer that
    exists in `layers` and a capability that exists in `capabilities`.
  - Asserts `len(config_fields) == 21` (R-F gate).
  - Asserts `len(completeness_criteria) == 7` and each has
    `id`, `title`, `acceptance` keys.
- `tests/completeness/test_c[1-7]_*.py` — red (skipped). Each contains
  a TODO docstring describing what Phase 1+ must wire up to make it
  green, plus a single `pytest.skip(...)` call with the phase number
  responsible for activating it.

## Compatibility

- **No runtime change**: no source under `src/geny_executor/` is
  modified by Phase 0.
- **No public API touched**: `pyproject.toml` is *not* bumped here;
  the next bump (`0.14.0`) occurs at the end of Phase 1 when the
  `geny_executor.memory.provider` module ships.
- **Test runner**: `tests/completeness/` joins the existing
  `testpaths = ["tests"]` glob. The skip-by-default pattern keeps CI
  green. Meta-test `test_spec_loads.py` actively guards the spec.

## Version bumps

- None. Phase 0 ships only docs + skipped tests.

## Follow-up

- Phase 1 (`progress/memory/phase_1_interface.md`):
  - Land `geny_executor.memory.provider` (Protocol + handles).
  - Implement `EphemeralMemoryProvider`.
  - Flip `test_c1_six_layer_retrieval.py` from skip → real assertion.
- Phase 2 will incrementally activate C2, C3, C5, C6, C7.
- Phase 4 activates C4 (REST coverage).
- The "golden dataset" extraction (sample notes + vector corpus +
  transcripts) remains a Phase 0 *aspirational* item; landing it
  requires read access to a Geny installation. Tracked here, not
  blocking the gate — C7 contract tests will operate on synthesised
  fixtures until a real dataset is dropped under
  `tests/completeness/fixtures/geny_golden/`.
