# EnvironmentManifest

> Status: current for geny-executor 2.1.0.

A pipeline is fully described by an `EnvironmentManifest` — a serialisable, diffable JSON artifact that names every stage, picks one strategy per slot, and pins config values. Loading the manifest reconstructs the pipeline deterministically.

This is the **recommended way to build pipelines for production hosts**. The fluent `PipelineBuilder` is convenient for scripts and tests; manifests are convenient for environments that need versioning, A/B comparison, multi-tenant deploys, and config-driven editing UIs.

## Quick start

```python
from geny_executor import (
    Pipeline,
    CredentialBundle,
    ProviderCredentials,
    EnvironmentManifest,
)

manifest = EnvironmentManifest.load("./envs/research_agent.json")

credentials = CredentialBundle(by_provider={
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
})

pipeline = await Pipeline.from_manifest_async(
    manifest,
    credentials=credentials,
    adhoc_providers=[my_tool_provider, my_skill_provider],   # optional
)

result = await pipeline.run("Find the latest Python release version")
```

## Schema (high level)

```json
{
  "name": "research-agent",
  "metadata": {
    "owner": "ops@example.com",
    "tags": ["research", "production"]
  },
  "stages": [
    {
      "order": 1,
      "name": "input",
      "active": true,
      "artifact": "default",
      "config": {"max_chars": 100000},
      "strategies": {}
    },
    {
      "order": 6,
      "name": "api",
      "active": true,
      "artifact": "default",
      "config": {
        "provider": "claude_code_cli",
        "model": "sonnet",
        "max_tokens": 4096
      },
      "strategies": {
        "retry":  "exponential_backoff",
        "router": "passthrough"
      }
    },
    ...
  ],
  "tools": {
    "built_in": ["Read", "Glob", "Grep", "TodoWrite"],
    "external": ["web_search", "memory_write", "send_dm"],
    "mcp_servers": []
  },
  "max_iterations": 50,
  "cost_budget_usd": 5.0
}
```

### `stages[]`

One entry per stage. Order matches the canonical 1–21. Fields:

| Field | Type | Notes |
|---|---|---|
| `order` | `int` | Canonical stage number (1–21). |
| `name` | `str` | Stage identifier (`input`, `context`, `system`, ..., `yield`). |
| `active` | `bool` | When `false`, the stage is registered but `should_bypass()` returns `True` for every input. |
| `artifact` | `str` | Which artifact bundle to load. Always `"default"` unless a stage ships alternates (e.g. `s14_evaluate` ships `adaptive` + `default`). |
| `config` | `dict` | Stage-level config consumed by `update_config()`. Provider lives here for Stage 6. |
| `strategies` | `dict` | One key per `StrategySlot` the stage owns. Value is the strategy name. |

### `tools`

| Field | Notes |
|---|---|
| `built_in` | Names from `geny_executor.tools.built_in.BUILT_IN_TOOL_CLASSES`. `["*"]` registers every shipped tool. |
| `external` | Names resolved against `adhoc_providers` passed to `Pipeline.from_manifest_async`. The first provider that knows the name wins. |
| `mcp_servers` | `MCPServerConfig` entries for host-attached MCP servers (transport + url/command + env). |

## Strict load (recommended)

```python
manifest = EnvironmentManifest.load("./envs/x.json", strict=True)
```

Strict mode rejects:
- Stage 6 (`api`) missing `config["provider"]`
- Any stage carrying `strategies["provider"]` (legacy slot — moved to `config["provider"]` in 2.0.0)
- Unknown stage `name` / `order`
- Unknown built-in tool names in `tools.built_in`

The single-source-of-truth rule lives in `core/pipeline.py:_validate_manifest_provider_locations`. The strict-load contract is the executor's commitment that "configuration is artifact" — if it loads, it runs.

## `Pipeline.from_manifest_async` parameters

| Param | Notes |
|---|---|
| `manifest` | The loaded `EnvironmentManifest`. |
| `credentials` | A `CredentialBundle` — see [providers.md](providers.md). |
| `api_key` | Legacy fallback; auto-wrapped into a `CredentialBundle` if `credentials` not provided. |
| `adhoc_providers` | A `Sequence[AdhocToolProvider]` resolving `tools.external` names. Geny passes a `GenyToolProvider` here. |
| `subagent_registry` | A `SubagentRegistry` (Stage 12 sub-agent orchestration). Optional. |
| `strict` | When `True` (default), strict-load is applied. |

## Mutating a live manifest

A built pipeline tracks the manifest it came from. `core/mutation.py` exposes `PipelineMutator` which can:

```python
from geny_executor import PipelineMutator

mut = PipelineMutator(pipeline)
record = await mut.swap_strategy(stage_order=2, slot_name="loader", impl="vector_search")
# record.before / record.after / record.applied
```

If the target stage is currently executing, `MutationLocked` is raised; the host can retry on the next iteration boundary. After a successful mutation the manifest is updated in place so a subsequent `pipeline.snapshot()` reflects the new shape.

## Snapshot — manifest from a live pipeline

```python
snap = pipeline.snapshot()
new_manifest = snap.to_manifest()
new_manifest.save("./envs/research_agent.v2.json")
```

Useful for:
- exporting a hand-built pipeline so it can be reloaded later
- diffing two pipelines (`EnvironmentDiff`)
- A/B comparing strategy choices in a controlled environment

## Environment vs preset

- **Preset** (`PipelinePresets.chat()`, etc.) — fluent builder shortcut. Best for scripts, examples, tests.
- **Manifest** — JSON artifact loaded at runtime. Best for production hosts, UI-edited environments, multi-tenant deploys.

A preset can always be snapshotted to a manifest, and a manifest can always be loaded into a pipeline equivalent to the preset that produced it. The two paths are interoperable.

## Where Geny uses this

[Geny](https://github.com/CocoRoF/Geny) is a multi-agent platform that ships ~5 manifest templates (worker / vtuber / sub-worker / …) and lets operators clone + edit them via a web UI. Every running session resolves a manifest via `EnvironmentService.instantiate_pipeline(env_id, credentials, ...)` which calls `Pipeline.from_manifest_async` under the hood. The manifest is the only place provider / model / built-in / external-tool selection live — no Python code path can pick a provider behind the manifest's back.
