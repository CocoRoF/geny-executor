# Changelog

All notable changes to `geny-executor` are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.22.0] — 2026-04-20

Tool / MCP integration hardening release. Bundles four breaking
changes discovered during the Geny ↔ executor cutover (see
`Geny/dev_docs/20260420_2/plan/` for the full context). The release
is intentionally packaged as one breaking bump so downstream Geny
can pin `geny-executor>=0.22.0,<0.23.0` and cut over in a single
PR rather than chasing four micro-upgrades.

### Added

- **`ToolError` / `ToolFailure` / `ToolErrorCode`** in
  `geny_executor.tools.errors`. Structured error model replacing ad-hoc
  string returns. Every host-side error now surfaces a stable payload
  `{error: {code, message, details}}` which the Anthropic tool_result
  bridge renders with a leading `ERROR <code>: <message>` header line.
  Codes: `UNKNOWN_TOOL`, `INVALID_INPUT`, `TOOL_CRASHED`,
  `ACCESS_DENIED`, `TRANSPORT`. (#22)
- **`validate_input(schema, payload)`** — jsonschema helper used by the
  default router and available for tool implementations. Converts
  jsonschema failures into `ToolFailure(code=INVALID_INPUT)`. (#22)
- **`MCPConnectionError(server_name, phase, cause)`** in
  `geny_executor.tools.mcp.errors` — a single structured exception for
  every phase of MCP server start-up (`connect`, `initialize`,
  `list_tools`, `sdk_missing`). (#24)
- **`Pipeline.from_manifest_async`** — async sibling of
  `from_manifest` that assembles stages, opens MCP servers with
  fail-fast semantics, registers adapters, and attaches
  `pipeline.mcp_manager` / `pipeline.tool_registry`. (#24)
- **`MCPManager.add_server(config, *, registry=None)`** /
  **`MCPManager.remove_server(name, *, registry=None)`** — runtime
  hot-swap of MCP servers that also keeps the registry in sync. (#24)
- **`AdhocToolProvider` Protocol** in
  `geny_executor.tools.providers` — runtime-checkable Protocol with
  `list_names()` / `get(name)` that lets hosts supply tools not
  expressible as `AdhocToolDefinition`. (#25)
- **`ToolsSnapshot.external: List[str]`** — manifest-level whitelist
  naming which provider-backed tools are active in a given
  environment. Legacy manifests (without the field) continue to load
  unchanged. (#25)
- **`Pipeline.from_manifest(..., adhoc_providers=(), tool_registry=None)`**
  and the matching async signature — walks `manifest.tools.external`,
  registers the first claiming provider per name into the supplied
  (or fresh) registry, attaches it to the pipeline. (#25)

### Changed (breaking)

- **Every MCP tool is now always namespaced `mcp__{server}__{tool}`**
  (previously the bare tool name). The prefix is mandatory; there is
  no opt-out. Host-side tool registries, logs, and downstream
  display code that matched on bare MCP tool names need to be
  updated. (#23)
- **MCP lifecycle is fail-fast.** Previously an MCP server that
  failed its `initialize` or `list_tools` step could persist in a
  "connected-but-no-op" state. v0.22.0 raises `MCPConnectionError`
  from `MCPManager.connect_all` at session-start time and rolls
  back every transiently-connected server before the exception
  propagates. Manifests that reference a broken MCP server will no
  longer load — the failure is now eager, not lazy. (#24)
- **`MCPServerConnection.call_tool`** return type expanded from
  `str` to `str | list[dict]`. Single-text-block responses still
  return `str`; multi-block and non-text responses return
  `list[dict]` preserving block `type`. `MCPToolAdapter.execute`
  passes both through to `ToolResult.content` unchanged. Direct
  callers of `call_tool` may now need an `isinstance` branch. (#24)
- **`ToolRegistry.register`** now emits a warning when a different
  tool instance is re-registered under an existing name. The
  previous silent overwrite hid double-registration bugs. (#23)
- **Default `RegistryRouter`** emits structured `ToolError` payloads
  for unknown tool, invalid input, tool crash, and access-denied
  flows. Callers that parsed the previous plain-string error
  content must switch to the structured shape. (#22)

### Dependencies

- Adds `jsonschema>=4.0` as a runtime dependency. (#22)

### Migration notes

- **MCP tool names**: any prompt, mapping, or log-scrape that
  referred to `read_file` now needs to reference
  `mcp__filesystem__read_file` (or the appropriate server prefix).
- **MCP manifests**: any environment that previously got away with
  a half-broken MCP server definition will now fail loudly at
  session start. Clean stale `mcp_servers` entries before deploy.
- **Tool error parsing**: host code that did
  `if result.content.startswith("Error:")` should switch to
  checking `result.is_error` and reading the structured
  `content["error"]["code"]`.
- **Unified tool surface (opt-in)**: hosts using the new
  `AdhocToolProvider` hook can point every environment — env_id
  or non-env_id — at a single `Pipeline.from_manifest_async(...)`
  call and drop any bespoke `ToolRegistry` plumbing. See the
  companion `Geny/dev_docs/20260420_2/plan/01_unified_tool_surface.md`.

### PRs in this release

- #22 — structured `ToolError` + jsonschema input validation.
- #23 — mandatory `mcp__{server}__{tool}` namespace.
- #24 — MCP fail-fast lifecycle + `Pipeline.from_manifest_async`.
- #25 — `AdhocToolProvider` Protocol + `tools.external` field.
