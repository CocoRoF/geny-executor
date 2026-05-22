# Hosting the Claude Code CLI as a provider

> Status: current for geny-executor 2.1.0.

The `claude_code_cli` provider routes Stage 6 through a spawned `claude` CLI subprocess. Unlike the SDK providers, the CLI runs the **entire agentic loop internally** — LLM ↔ tools ↔ LLM happens inside the spawned process. To make that loop usable by a host (Geny, CI runner, custom orchestrator), geny-executor exposes:

- a stable argv builder with version-compat fixes
- a per-session **MCP wrap** that surfaces the host's tool registry to the CLI's LLM as `mcp__<server>__<tool>`
- a `tool_use` strip from the assembled `APIResponse` so the host's Stage 10 doesn't try to re-dispatch tools the CLI already handled
- structured error codes (`exec.cli.*`) so authentication / permission / timeout failures group cleanly in the host's logs

## Minimum viable setup

```python
from geny_executor import CredentialBundle, ProviderCredentials, PipelineBuilder

bundle = CredentialBundle(by_provider={
    "claude_code_cli": ProviderCredentials(
        # Subscription OAuth — leave api_key empty
        api_key="",
        binary_path="/usr/local/bin/claude",
        extras={
            "bare_mode": True,         # auto-stripped on OAuth path (2.0.6+)
            "timeout_s": 600.0,
        },
    ),
})

pipeline = (
    PipelineBuilder("cli-agent", credentials=bundle)
    .with_provider("claude_code_cli")
    .with_model(model="sonnet")        # alias; CLI resolves to current Sonnet
    .build()
)
```

That spawns `claude --print --output-format json --bare ...` on every call. The CLI's built-in palette (Bash / Read / Write / Edit / Glob / Grep / WebFetch / …) is available; no MCP servers are attached.

## Per-session MCP wrap (host tool registry → CLI)

To make the CLI's LLM call **your** tools, attach an MCP config. The host runs a small MCP stdio bridge that proxies `tools/list` + `tools/call` to its own tool registry.

```python
mcp_config = {
    "mcpServers": {
        "geny": {
            "type": "stdio",
            "command": "/usr/bin/python3",
            "args": ["/app/scripts/geny_mcp_bridge.py"],
            "env": {
                "GENY_MCP_URL": "http://127.0.0.1:8000",
                "GENY_MCP_TOKEN": session_bearer_token,
                "GENY_MCP_SESSION_ID": session_id,
            },
        },
    },
}

bundle = CredentialBundle(by_provider={
    "claude_code_cli": ProviderCredentials(
        binary_path="/usr/local/bin/claude",
        extras={
            "mcp_config": mcp_config,
            "settings_path": '{"permissions":{"allow":["mcp__geny","Bash","Read","Write","Edit"]}}',
        },
    ),
})
```

With this attached:
- The argv builder emits `--mcp-config <json> --strict-mcp-config`. The strict flag scopes the MCP surface to only what the host provides (no user-level or project-level MCP servers leak in).
- The CLI normalises MCP tool names to `mcp__<server>__<tool>` — your `geny.send_direct_message_internal` tool surfaces to the LLM as `mcp__geny__send_direct_message_internal`.
- CLI built-ins stay available alongside the MCP surface unless you explicitly disable them with `extras["extra_args"] = ("--tools", "")`.

The bridge script is whatever you want — an MCP-spec stdio loop that forwards JSON-RPC to your tool dispatcher. Geny ships [one](https://github.com/CocoRoF/Geny/blob/main/backend/scripts/geny_mcp_bridge.py) as reference (~130 lines, stdlib only).

## `extras` catalog

| Key | Type | Default | Notes |
|---|---|---|---|
| `bare_mode` | `bool` | `True` | Maps to `--bare`. Auto-stripped on the OAuth path (no `ANTHROPIC_API_KEY` in env) since 2.0.6 — the same default works for both auth modes. |
| `workspace_root` | `str` | `None` | Subprocess `cwd`. Useful for sandboxing file-system tools. |
| `default_permission_mode` | `str` | `"default"` | One of `acceptEdits` / `auto` / `bypassPermissions` / `default` / `dontAsk` / `plan`. `bypassPermissions` is **blocked when running as root** by the CLI; use a `settings_path` permissions allow-list instead. |
| `max_budget_usd` | `float` | `None` | Maps to `--max-budget-usd`. |
| `settings_path` | `str` | `None` | File path *or* inline JSON. Inline JSON (`'{"permissions":{"allow":["mcp__geny","Bash"]}}'`) is the recommended way to pre-allow tools without a temp file. |
| `mcp_config` | `dict \| str` | `None` | Per-client static MCP config. Per-request `APIRequest.mcp_config` wins when both are set. |
| `allow_tools` | `Sequence[str]` | `()` | Emitted as `--allowedTools`. **Permission-pattern allowlist** (`Bash(git *)`), not a tool enablement filter. |
| `disallow_tools` | `Sequence[str]` | `()` | Emitted as `--disallowedTools`. |
| `extra_args` | `Sequence[str]` | `()` | Escape hatch — appended verbatim. Use for flags the executor doesn't model (e.g. `("--tools", "")` to fully disable CLI built-ins). |
| `timeout_s` | `float` | `300.0` | Subprocess wall-clock timeout. |

## What the argv builder does for you (2.0.6+)

For streaming requests on the OAuth subscription path (no `ANTHROPIC_API_KEY` in env), the argv looks like:

```
claude --print --verbose
  --input-format stream-json
  --output-format stream-json
  --include-partial-messages
  --model sonnet
  --system-prompt '<host-assembled>'
  --settings '{"permissions":{"allow":["mcp__geny",...]}}'
  --mcp-config '{"mcpServers":{...}}'
  --strict-mcp-config
```

Automatic compat handling:

| Behaviour | Why |
|---|---|
| `--verbose` injected after `--print` when output is `stream-json` | CLI ≥ 2.1.x requires it; otherwise exits 1. |
| `--bare` stripped when no `ANTHROPIC_API_KEY` in env | `--bare` disables OAuth + keychain reads; combining with no API key crashes every subscription user. |
| Auto-`--tools ""` **not** emitted when MCP is configured | Earlier versions disabled CLI built-ins; in practice hosts want both surfaces (built-ins + MCP). Disable explicitly via `extras["extra_args"]` if you really want MCP-only. |
| `--strict-mcp-config` emitted when MCP is configured | Scopes MCP to the host's bridge only — no user-level / project-level MCP servers leak in. |

## `tool_use` blocks are dropped from the response (2.0.6+)

Claude Code CLI 2.1.x runs the entire agentic loop internally. Each intermediate turn arrives as its own `"assistant"` envelope in the stream-json output, and the accumulator collects every block from every envelope. Earlier executor versions surfaced those `tool_use` blocks in the final `APIResponse.content`, which made Geny's Stage 10 try to re-dispatch tools the CLI already handled — producing instant `ERROR (0 ms)` ghost-failures.

2.0.6 strips `tool_use` blocks from `StreamJsonAccumulator.finalize` (and the non-streaming `parse_json_output_to_response`). `stop_reason` is preserved so callers can still tell when the CLI ended in a tool turn. Hosts that want the raw per-block records can still recover them from the per-line `feed()` event stream.

Per the Phase-I design contract:
> Stage 10 receives that assistant message, sees no `tool_use` blocks (they were executed inside the CLI), and naturally no-ops.

## Error codes

CLI failures surface as `APIError` with `exec.cli.*` codes:

| Code | When | Recoverable? |
|---|---|---|
| `exec.cli.binary_not_found` | `claude` not on PATH and `binary_path` not set | ❌ |
| `exec.cli.auth_failed` | CLI reports `authentication_failed` | ❌ — re-login |
| `exec.cli.timeout` | Subprocess hit `timeout_s` | ✅ |
| `exec.cli.protocol_error` | Malformed stream-json envelope | ✅ |
| `exec.cli.permission_denied` | CLI's permission system blocked the call | ❌ — fix `settings_path` allow-list |
| `exec.cli.exited` | Non-zero exit outside the categorised cases | ✅ |

Full table + the host-side i18n recipe lives in [error_codes.md](error_codes.md).

## Observability tip

The accumulator's per-line `feed()` events bubble through Stage 6's `_call_streaming` as `{"type": "tool_use", ...}` / `{"type": "tool_result", ...}` chunks. Hosts can tap these events to render CLI-internal tool calls (e.g. `Bash`, `Read`, `Write`) in their UI — Geny does exactly this with a context-variable-routed session logger (see [`Geny's llm_patches.py`](https://github.com/CocoRoF/Geny/blob/main/backend/service/llm_patches.py) for a reference implementation).

A future minor release will fold this observability into a first-class pipeline event so hosts don't need to monkey-patch the accumulator.
