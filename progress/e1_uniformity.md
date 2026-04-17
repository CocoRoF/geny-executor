# E1 — Stage Uniformity (v0.11.0 → v0.12.0)

> **Completed**: 2026-04-17
> **Scope**: Standardize the introspection, configuration, and runtime-override
> contract across all 16 pipeline stages.
> **Target tag**: `v0.12.0`

---

## Summary

All 16 stages now expose the same programmatic surface, making the pipeline
uniformly hot-configurable. The payoff is in E2–E5: a web stage editor
(E2) can generically render every stage, and the consumer API (E4) can
guarantee per-stage model/tool overrides independent of which stage it
reaches.

---

## Deliverables

### E1.1 — 14 non-chain stages adopted the `StrategySlot` pattern
Every slot-based stage (s01/02/03/05/06/07/08/09/10/11/12/13/15/16) returns a
`Dict[str, StrategySlot]` from `get_strategy_slots()`. `list_strategies()`
is now auto-generated from `describe()` on each slot.

### E1.2 — `SlotChain` for chain stages
`s04 Guard` and `s14 Emit` migrated from ad-hoc chain fields to the new
`SlotChain` sibling of `StrategySlot`. `add_guard()` / `add_emitter()`
remain as deprecated shims emitting `DeprecationWarning`. A second hook
— `get_strategy_chains()` — was added to `Stage` so `list_strategies()`
composes both surfaces.

`SlotChain` features: `append(impl_name, config)` / `remove(name)` /
`reorder(order)` with permutation validation / `clear()` / `describe()`.

### E1.3 — Builder args introspectable via `ConfigSchema`
Stage-level parameters that used to live only in builder kwargs now have
explicit `get_config_schema()` / `get_config()` / `update_config()` triads
on `s02 / s03 / s05 / s06 / s11 / s13 / s15` (and `s04`). This lets a UI
render inputs without hand-wiring each stage.

### E1.4 — `PipelineMutator` extensions
New mutation kinds: `UPDATE_STRATEGY_CONFIG`, `REPLACE_STAGE`,
`REORDER_CHAIN`, `ADD_TO_CHAIN`, `REMOVE_FROM_CHAIN`, `REGISTER_HOOK`,
`UNREGISTER_HOOK`, `BIND_TOOL`, `UNBIND_TOOL`, `SET_TOOL_SCOPE`,
`SET_STAGE_MODEL`.

Added `batch()` context manager (snapshot/restore-based atomic rollback)
and hook bridge (monkey-patches stage `on_enter`/`on_exit`/`on_error` to
also dispatch registered callbacks, recoverable via hook id).

### E1.5 — Per-stage Tool Binding
New: `tools/stage_binding.py` with `StageToolBinding(stage_order, allowed?,
blocked?)` and `ToolAccessDenied` exception. Semantics: `blocked` wins
over `allowed`; `allowed=None` means inherit-everything. `Stage.tool_binding`
is a lazy property so existing stages keep working unchanged. `s10 Tool`
enforces the binding pre-execute and rejects disallowed tool calls.
`ToolContext` now carries `stage_order` + `stage_name` so tools can log
the requesting stage.

### E1.6 — Per-stage Model Override
`Stage.model_override` (nullable `ModelConfig`) + `Stage.resolve_model(state)`
helper. `s06 API._build_request` now honors the override per-field
(`model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`,
`thinking_*`). `PipelineMutator.set_stage_model(order, model|None)`
round-trips; passing `None` reverts to the pipeline-wide config. The
other model-adjacent stages (s08, s11, s12, s15 default strategies)
don't call the model directly today — the override hook is exposed for
future model-backed strategies via `resolve_model`.

### E1.7 — Preset Plugin System
New `PresetRegistry` + `register_preset` decorator in `core/presets.py`.
Third-party packages can contribute pipeline presets through the
`geny_executor.presets` entry-point group, or programmatically via the
decorator. `PresetManager.list_all()` now surfaces plugin presets as
`preset_type="plugin"`, and `PresetManager.create(name, **kwargs)`
resolves built-ins first, then plugins. Auto-discovery is cached and
can be refreshed with `refresh_plugins()`.

### E1.8 — Stage Uniformity Contract tests
`tests/contract/test_stage_uniformity.py` pins the surface: for each of
the 16 stages, 11 parametrized assertions verify identity, slot/chain
shape, describe, tool binding, model override, and the config triad.
Two chain-specific assertions pin `s04 Guard → "guards"` and
`s14 Emit → "emitters"`. Module-level tests verify that stage orders
are a permutation of `1..16` and that names are unique.

---

## Contract (as enforced)

Every `Stage`:
1. `name`, `order`, `category` (category ∈ `{ingress, pre_flight, execution, decision, egress}`)
2. `get_strategy_slots() -> Dict[str, StrategySlot]`
3. `get_strategy_chains() -> Dict[str, SlotChain]`
4. At least one of `slots` or `chains` is non-empty.
5. `describe() -> StageDescription` with matching identity
6. `tool_binding -> StageToolBinding(stage_order=self.order)` (default: inherit)
7. `model_override -> Optional[ModelConfig]` (default: `None` = inherit)
8. `get_config_schema() -> Optional[ConfigSchema]`
9. `get_config() -> Dict[str, Any]`
10. `update_config({}) -> None` (no-op safe)

Chain stages additionally:
- `s04 Guard` exposes `"guards"` chain.
- `s14 Emit` exposes `"emitters"` chain.

---

## Affected files (net)

- `src/geny_executor/core/slot.py` — added `SlotChain`
- `src/geny_executor/core/stage.py` — chain hook, tool_binding, model_override, resolve_model
- `src/geny_executor/core/mutation.py` — 11 new MutationKinds, hook bridge, batch, bind/unbind, set_stage_model
- `src/geny_executor/core/presets.py` — `PresetRegistry`, `register_preset`, plugin discovery, `PresetManager.create/refresh_plugins`
- `src/geny_executor/tools/stage_binding.py` — new
- `src/geny_executor/tools/base.py` — `ToolContext.stage_order` / `stage_name`
- `src/geny_executor/stages/s04_guard/artifact/default/stage.py` — SlotChain rewrite
- `src/geny_executor/stages/s14_emit/artifact/default/stage.py` — SlotChain rewrite
- `src/geny_executor/stages/s14_emit/artifact/default/emitters.py` — optional callback + `bind_callback`
- `src/geny_executor/stages/s06_api/artifact/default/stage.py` — model_override wiring
- `src/geny_executor/stages/s10_tool/artifact/default/stage.py` — binding enforcement + stage-scoped ToolContext
- `src/geny_executor/stages/{s02,s03,s05,s06,s11,s13,s15}/.../stage.py` — config triad
- `src/geny_executor/__init__.py` — re-export SlotChain, PresetRegistry, register_preset
- `tests/contract/test_stage_uniformity.py` — new

---

## Version

`geny-executor` bumps from `v0.11.0` to `v0.12.0` — new minor release
carrying the E1 surface additions (SlotChain, StageToolBinding,
per-stage `model_override`, `PresetRegistry` + `register_preset`,
contract test suite). Fully backward-compatible: legacy `add_guard()` /
`add_emitter()` retained with `DeprecationWarning`. Consumers pinned to
`>=0.11.0` (geny-executor-web) or `>=0.10.0` (Geny) remain compatible.

The plan_evolution/ naming (v0.8/v0.9/v1.0) referred to a hypothetical
reset; this repository's released numbering continues monotonically.
E1 → v0.12.0; subsequent phases will bump the minor further (E2/E3 →
v0.13.x; E4/E5 → v1.0.0 as major stability declaration).

---

## Next

- **E2**: Web stage editor (geny-executor-web consumes the new contract).
- **E3**: Environment completeness — persisting the expanded contract
  end-to-end.
- **E4**: External consumer one-line API.
- **E5**: Verification / release hardening.
