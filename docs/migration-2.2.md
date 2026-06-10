# Migrating hosts to 2.2.0

> Audience: maintainers of Geny, GAPT, and any other host carrying a
> compensation layer over geny-executor ≤ 2.1.x. The 2026-06-09
> environment-philosophy audit mapped ~3,800 lines of host code that
> exists only because the library lacked an owned API. 2.2.0 ships
> those APIs; this guide maps each compensation module to its
> replacement so it can be deleted.

Everything below is additive — 2.2.0 is a minor release and no 2.1.x
surface was removed. You can migrate one module at a time.

## At a glance

| Host compensation | 2.2.0 replacement |
|---|---|
| Geny `llm_patches.py` (~479 lines, monkey-patch) | `pipeline.events()` tap + `api.error` / `api.cli_tool_call` events |
| GAPT `executor_patches.py` (3 private forks) | `ClaudeCodeCLIClient(runner_factory=…)` + built-in chunk forwarding |
| Geny `default_manifest.py` / `stage_manifest.py` (~1,150 lines) | `build_manifest(preset, *, provider, …)` |
| Geny `backend_resolver.py` (and GAPT's equivalent) | `CredentialBundle.preferred_provider()` |
| GAPT `pipeline._config.model.*` baseline/revert dance | `run(..., overrides=ModelOverrides(...))` |
| Geny `queue_runtime_refresh` (~220 lines, private setters) | `Pipeline.refresh_runtime(**kwargs)` |
| Geny `_force_required_stages_active` pre-build rewrite | `validate_manifest()` + `from_manifest(strict=True)` |

## 1. `llm_patches.py` → events tap + `api.error` events

The patch existed because CLI-dispatched tool calls and error
envelopes never reached the event bus — the only way to see them was
to monkey-patch the accumulator. 2.2.0 publishes both on the unified
stream, named in the `EventTypes` catalogue ([events.md](events.md)).

Before (Geny, abridged):

```python
# llm_patches.py — reach inside the CLI client to see tool calls
_orig_feed = StreamJsonAccumulator.feed
def _patched_feed(self, line):
    obj = json.loads(line)
    if _looks_like_tool_use(obj):
        session_logger.tool_call(...)        # reverse-engineered shape
    return _orig_feed(self, line)
StreamJsonAccumulator.feed = _patched_feed
```

After:

```python
from geny_executor import EventTypes

async for event in pipeline.events(replay_from=0):
    if event.type == EventTypes.API_CLI_TOOL_CALL:
        session_logger.tool_call(event.data)          # source == "cli"
    elif event.type == EventTypes.API_TOOL_RESULT:
        session_logger.tool_result(event.data)
    elif event.type == EventTypes.API_ERROR:
        session_logger.error(event.data["code"], event.data["message"])
```

`pipeline.events()` is a multi-subscriber tap with cursor replay
(`event.seq`), so the 50 ms polling loop over `state.events` and both
copies of the 600-line event-mapping switch go with it. Payload field
documentation lives in `PAYLOADS` / [events.md](events.md) — no more
guessing names (the guesses are how GAPT shipped a 100%-text-loss bug
and a $0-cost bug).

## 2. `executor_patches.py` → `runner_factory` + chunk forwarding

GAPT forked three private internals, which pinned it to 2.1.0 and cut
it off from every 2.1.x vendor-drift fix. Each fork now has a
supported seam:

- **`CLIProcessRunner._spawn` fork** (docker sandbox) →
  `runner_factory` constructor kwarg. The version-handshake probe
  routes through the same factory, so the recorded CLI version matches
  the binary that actually runs.
- **`_call_streaming` fork** (forward tool_use/thinking chunks) →
  built in. Stage 6 forwards every canonical chunk as events:
  `text.delta`, `thinking.delta`, `api.tool_use`,
  `api.input_json_delta`, `api.content_block_stop`, `api.tool_result`.
- **`StreamJsonAccumulator.feed` fork** → same events, plus
  structured `api.error` carrying the stable `exec.*` code
  ([error_codes.md](error_codes.md)).

Before:

```python
# executor_patches.py
CLIProcessRunner._spawn = _sandboxed_spawn            # docker exec wrapper
APIStage._call_streaming = _forward_tool_chunks       # 2.1.0 internals fork
```

After:

```python
client = ClaudeCodeCLIClient(
    runner_factory=lambda **kw: SandboxedRunner(container=workspace_id, **kw),
)
# chunk forwarding needs no code — subscribe to the events instead
```

## 3. `default_manifest.py` → `build_manifest`

Before: a 728-line hand-built manifest dict per preset (plus a second
copy in `stage_manifest.py`), drifting against the library's stage
catalogue with every release.

After:

```python
from geny_executor import build_manifest

manifest = build_manifest(
    "worker_adaptive",                  # or "vtuber" / "default"
    provider="claude_code_cli",
    model="claude-sonnet-4-6",
    built_in_tools=["*"],
    mcp_servers=[{"name": "geny-bridge", "url": bridge_url}],
)
```

The result carries the canonical 21-stage layout, builds under
`Pipeline.from_manifest(strict=True)`, and round-trips
`to_dict`/`from_dict` unchanged. Unknown presets/providers and
malformed MCP entries fail at factory time, not at first run.

## 4. `backend_resolver` → `preferred_provider`

```python
provider = credentials.preferred_provider()
if provider is None:
    raise ConfigError("no LLM backend configured")
manifest = build_manifest("worker_adaptive", provider=provider)
```

The default order prefers the agentic CLI backend, then vendor APIs
by capability breadth; pass your own `order=` if your priorities
differ. `None` is returned (never a silent default) when nothing is
configured.

## 5. Private `_config` mutation → `ModelOverrides`

Before (GAPT's baseline/revert dance — overrides leaked into later
runs whenever the revert path was skipped):

```python
baseline = pipeline._config.model.model
pipeline._config.model.model = "claude-opus-4-7"
try:
    result = await pipeline.run(text, state=state)
finally:
    pipeline._config.model.model = baseline
```

After:

```python
from geny_executor import ModelOverrides

result = await pipeline.run(
    text, state=state,
    overrides=ModelOverrides(model="claude-opus-4-7"),
)
```

Lifetime is exactly one run — the next run reverts automatically, and
each applied field emits `config.override_applied` so the UI can show
why this run used a different model.

## 6. Refresh queue → `refresh_runtime`

Before: ~220 lines queueing private-setter writes between turns to
rotate credentials / swap tool contexts.

After:

```python
pipeline.refresh_runtime(llm_client=rotated_client)
```

Same kwargs and wiring as `attach_runtime`, legal at any turn
boundary; raises `RuntimeError` if a run is in flight (the mixed-
runtime hazard the old construction-time gate existed to prevent). A
refreshed `llm_client` bumps the client generation so long-lived
states re-resolve it on the next turn.

## 7. `_force_required_stages_active` → strict validation

Before: the host rewrote manifests pre-build to flip required stages
active, because strict load happily built an "agent" with Stage 6
inactive.

After: delete the rewrite. `validate_manifest(manifest)` reports
`stage.required_inactive` (and unknown strategies, dropped configs,
misplaced keys) as `error` findings; `Pipeline.from_manifest(...,
strict=True)` refuses to build on them. Call `validate_manifest` at
environment-save time to surface the findings in the editor instead
of at build:

```python
issues = validate_manifest(manifest)
errors = [i for i in issues if i.severity == "error"]
if errors:
    return JSONResponse({"issues": [i.to_dict() for i in errors]}, 422)
```

## Notes

- **Strict mode got stricter.** Manifests that loaded under
  `strict=True` on 2.1.x may now be rejected — every rejection names a
  declaration that was silently inert before (a strategy config no
  strategy consumed, a required stage that wasn't running). Run
  `validate_manifest` over your stored manifests before upgrading;
  `warning` findings log and never block, lenient (`strict=False`)
  builds keep working unchanged.
- **Deny posture is config-reachable, not yet the default.** The
  permission system honours `default_posture: "deny"` both when no
  rule matches and when zero rules are bound. The shipped default
  remains `allow` for 2.x back-compat; flip it per environment now —
  3.0 flips the default.
- **Teardown is now owed.** Migrated hosts must call
  `await pipeline.aclose()` when the owning session ends — it cancels
  pending HITL futures, closes `events()` taps, disconnects MCP
  servers (reaping stdio children), and shuts down tool providers.
  Before 2.2.0 there was no aggregate teardown and Geny leaked an MCP
  child process per stopped session.
- **Long-lived states get a turn contract.** If you reuse one
  `PipelineState` across turns (GAPT's model), per-turn fields
  (`iteration`, `loop_decision`, in-flight tool work) are reset by
  `begin_turn()`, called automatically when the reused state re-enters
  `run()` / `run_stream()`. Pass `state=` explicitly — a `state=None`
  call still creates a fresh state per turn.
