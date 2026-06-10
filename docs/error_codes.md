# geny-executor Error Codes

**Since:** 2.1.0 (table verified against the enum on 2026-06-10, 2.2.0 cycle)
**Source of truth:** `src/geny_executor/core/errors.py` (`ExecutorErrorCode` enum)

Every exception raised by geny-executor carries a stable string identifier
in the form `exec.<component>.<reason>`. Hosts use this code for:

- **Logging / Sentry grouping** — drop the free-form `str(exception)` from
  your dashboards and group on the code instead.
- **i18n** — map each code to a localized message template in your UI
  layer (see [Geny's example](https://github.com/CocoRoF/Geny/blob/main/frontend/src/lib/i18n/en.ts)).
- **Telemetry routing** — alert differently for `exec.api.*` (vendor
  outages) vs `exec.cli.*` (host config bugs).
- **Retry / fallback decisions** — recoverability is also exposed via
  `ErrorCategory.is_recoverable`, but `code` lets you fine-tune.

## Stability contract

- Once published in a release, a code's string value **never changes**.
- Renaming or repurposing a code is a **breaking change** — deprecate
  the old code, add a new one.
- Adding new codes is non-breaking and ships in minor versions.
- The `tests/error_codes/test_code_stability.py` regression locks the
  string values so accidental rename CI-fails before release.

## Where the code surfaces

Every `GenyExecutorError` subclass exposes the code as the `code`
attribute. The pipeline's structured events (`stage.error`,
`pipeline.error`, `api.retry`, and — since 2.2.0 — `api.error`) also
carry it (see [events.md](events.md) for the full event catalogue):

```json
{
  "type": "pipeline.error",
  "data": {
    "error": "Claude Code CLI is not authenticated …",
    "code": "exec.cli.auth_failed",
    "exception_type": "geny_executor.core.errors.APIError"
  }
}
```

## Code table

### `exec.api.*` — vendor API surface

These come from the SDK-driven providers (Anthropic, OpenAI, Google,
vLLM). The companion `ErrorCategory` on the `APIError` decides retry
behavior; the code is the stable identifier consumers branch on.

| Code | Recoverable? | Source | Description |
|------|---|---|---|
| `exec.api.auth.invalid_key` | ❌ no | `APIError(category=AUTH)` | API key missing / malformed / rejected by vendor. Action: paste a valid key in the host's LLM Backends settings. |
| `exec.api.auth.expired` | ❌ no | `APIError(category=AUTH)` *(future use)* | Vendor reports the credential is past its TTL. Action: re-issue / refresh. |
| `exec.api.rate_limited` | ✅ yes | `APIError(category=RATE_LIMITED)` | Vendor 429. The retry strategy backs off and retries automatically. Persisted-rate errors after `EXEC_API_RETRY_EXHAUSTED`. |
| `exec.api.timeout` | ✅ yes | `APIError(category=TIMEOUT)` | Request exceeded the per-call timeout. Retry with backoff. |
| `exec.api.network` | ✅ yes | `APIError(category=NETWORK)` | Connection reset / DNS / TLS / transport. Retry with backoff. |
| `exec.api.token_limit` | ❌ no | `APIError(category=TOKEN_LIMIT)` | Prompt + max_tokens exceeded the model's context window. Action: shrink context or pick a larger-window model. |
| `exec.api.bad_request` | ❌ no | `APIError(category=BAD_REQUEST)` | Vendor 4xx other than auth/rate-limit. Usually a schema bug in the host's request shape. |
| `exec.api.server_error` | ✅ yes | `APIError(category=SERVER_ERROR)` | Vendor 5xx. Retried by the executor. |
| `exec.api.terminal` | ❌ no | `APIError(category=TERMINAL)` | Vendor declared the request fatally unprocessable (e.g. policy block). Don't retry. |
| `exec.api.unknown` | ❌ no | `APIError(category=UNKNOWN)` | Catch-all for vendor errors the executor couldn't classify. Investigate the underlying cause. |
| `exec.api.no_client` | ❌ no | Stage 6 build error | `state.llm_client` is `None`. Host forgot to call `Pipeline.from_manifest(credentials=…)` or `attach_runtime(llm_client=…)`. |
| `exec.api.stream_incomplete` | ❌ no | Stage 6 streaming | The stream ended without a `message_complete` event. Usually a vendor SDK bug or an interrupted upstream connection. |
| `exec.api.retry_exhausted` | ❌ no | Stage 6 retry loop | Hit `max_retries` after a recoverable error category. Look at the chained cause for the original failure. |

### `exec.cli.*` — CLI-driven backends (currently `claude_code_cli`)

| Code | Recoverable? | Source | Description |
|------|---|---|---|
| `exec.cli.binary_not_found` | ❌ no | `APIError(category=CLI_NOT_FOUND)` | The CLI binary (e.g. `claude`) is not on `PATH` and `binary_path` was not set. Action: install the CLI or configure the binary path. |
| `exec.cli.auth_failed` | ❌ no | `APIError(category=CLI_AUTH_FAILED)` | The spawned CLI reported `authentication_failed`. Action: re-run the CLI's login command (e.g. `claude auth login`) or paste a valid `ANTHROPIC_API_KEY`. |
| `exec.cli.timeout` | ✅ yes | `APIError(category=CLI_TIMEOUT)` | The CLI did not return within the configured `timeout_s`. Retry. |
| `exec.cli.protocol_error` | ✅ yes | `APIError(category=CLI_PROTOCOL_ERROR)` | The CLI emitted malformed stream-json output or unrecognised envelope. Retry; report if it persists. |
| `exec.cli.permission_denied` | ❌ no | `APIError(category=CLI_PERMISSION_DENIED)` | The CLI's permission system blocked the call (e.g. `--dangerously-skip-permissions` was attempted as root). Action: configure `permissions.allow` in the spawned settings. |
| `exec.cli.exited` | ✅ yes | CLI subprocess non-zero exit | The CLI process exited with a non-zero return code outside the categorised cases above. Inspect the chained cause. |

### `exec.pipeline.*` / `exec.stage.*` — orchestration

| Code | Source | Description |
|------|---|---|
| `exec.pipeline.not_initialized` | `PipelineError` | The pipeline was used before `build()` / `from_manifest()` was called. |
| `exec.pipeline.invalid_manifest` | `PipelineError` *(future use)* | The manifest's schema/strict load rejected the configuration. |
| `exec.stage.failed` | `StageError` (default) | A stage raised an exception that was wrapped by the pipeline's stage runner. Inspect the chained cause for the original failure. |
| `exec.stage.guard_rejected` | `GuardRejectError` | A Stage 4 guard refused execution (budget / cost / iteration / permission). The `guard_name` field on the exception identifies which guard. |

### `exec.tool.*` — Stage 10 tool dispatch

These mirror the existing `ToolErrorCode` enum at the routing layer.
Host pipelines see them surface via `ToolError.code` on the
`tool_result` payload too.

| Code | Source | Description |
|------|---|---|
| `exec.tool.unknown` | `RegistryRouter.unknown_tool()` | The LLM emitted a `tool_use` for a name that isn't registered. Usually a hallucination or a stale registry. |
| `exec.tool.invalid_input` | `RegistryRouter.invalid_input()` | The tool's input schema validation failed. `details.field_path` says where. |
| `exec.tool.access_denied` | `RegistryRouter.access_denied()` | The session's tool binding disallows this tool. |
| `exec.tool.crashed` | `RegistryRouter.tool_crashed()` | The tool's `execute()` raised an unexpected exception. `details.exception_type` carries the class. |
| `exec.tool.transport` | `RegistryRouter.transport()` | MCP adapter / RPC transport failure. `details.server` identifies the server. |

### `exec.mutation.*` — runtime config mutation

| Code | Source | Description |
|------|---|---|
| `exec.mutation.invalid` | `MutationError` | Bad stage / slot / impl in the mutation request. |
| `exec.mutation.locked` | `MutationLocked` | The target stage is currently executing; try again after stage exit. |

### `exec.mcp.*` — MCP server lifecycle (host-attached servers)

| Code | Source | Description |
|------|---|---|
| `exec.mcp.connect_failed` | `MCPConnectionError(phase="connect")` | Could not reach the MCP server (transport / process spawn / handshake). |
| `exec.mcp.initialize_failed` | `MCPConnectionError(phase="initialize")` | The MCP server connected but the `initialize` handshake failed. |
| `exec.mcp.list_tools_failed` | `MCPConnectionError(phase="list_tools")` | `tools/list` errored after a successful initialize. |
| `exec.mcp.sdk_missing` | `MCPConnectionError(phase="sdk_missing")` | The MCP SDK is not installed in the host's environment. |

### `exec.unknown` — fallback

| Code | When | Description |
|------|---|---|
| `exec.unknown` | last resort | The exception is a non-`GenyExecutorError` (e.g. raw `RuntimeError` / `ValueError`) and no code could be inferred. Indicates a raise site that hasn't been migrated to the typed exception hierarchy yet — file an issue. |

## How to add a new code

1. Add the enum value to `ExecutorErrorCode` in `core/errors.py`.
   Lowercase, dot-separated, ≤4 segments.
2. Add a row to the table above under the right component.
3. If the code corresponds to a legacy `ErrorCategory`, extend
   `_CATEGORY_TO_CODE_DEFAULT` so existing call sites pick it up.
4. The stability regression test will auto-pick up the new code; no
   test change needed for *additions*.

## How to deprecate a code

Don't delete it. Instead:

1. Mark the enum value with a deprecation comment.
2. Add a new code and migrate raise sites incrementally.
3. Keep the deprecated value in the table with a "deprecated → new code"
   note for at least one minor-version cycle.
4. Only remove the enum value in a **major** version bump.

## Migration phases

Phase 1 (2.1.0) — critical-path raise sites in
`claude_code.py` and `s06_api/stage.py`; all `APIError(category=…)`
sites automatically inherit the right code via `from_category`.

Phase 2+ (planned) — refit stage/guard/mutation/MCP raise sites with
explicit codes, drain generic `RuntimeError`/`ValueError` to typed
exceptions where appropriate.
