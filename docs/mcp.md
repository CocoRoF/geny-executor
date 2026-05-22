# MCP Integration

> Status: current for geny-executor 2.1.0.

geny-executor supports the [Model Context Protocol](https://modelcontextprotocol.io/) at **two distinct boundaries**, and it's important to know which one you want:

| Boundary | Where the MCP server lives | Who serves it | Who consumes it | Use when |
|---|---|---|---|---|
| **Host-attached MCP** (`MCPManager`) | Process spawned + managed by the executor host | Filesystem / GitHub / Slack / any external MCP server | The pipeline's `ToolRegistry` (so Anthropic SDK / OpenAI / Google / vLLM all see the tools natively) | You want to expose third-party MCP tools to a host-managed LLM call. |
| **Per-session CLI MCP wrap** (`APIRequest.mcp_config`) | Process spawned by the spawned `claude_code_cli` subprocess via `--mcp-config` | Your host's tool bridge | The spawned CLI's LLM | You're running `claude_code_cli` and want **your host's** tool registry available to the CLI's internal agentic loop. See [claude_code_cli.md](claude_code_cli.md). |

These two are independent. A single session can use both.

## Host-attached MCP servers

### Connecting

```python
from geny_executor.tools.mcp import MCPManager
from geny_executor.tools import ToolRegistry

mcp = MCPManager()
await mcp.connect(
    "filesystem",
    command="npx",
    args=["-y", "@anthropic/mcp-filesystem", "/sandbox"],
)
await mcp.connect(
    "github",
    command="npx",
    args=["-y", "@anthropic/mcp-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."},
)

registry = ToolRegistry()
for tool in mcp.list_tools():
    registry.register(tool)
```

The connected servers are adapted into `Tool` instances and registered alongside your native tools. The pipeline's Stage 10 dispatches them through the same router as built-ins — no special-casing needed at the call site.

### Lifecycle

| Phase | What happens |
|---|---|
| `connect()` | Spawns the server process, sends `initialize`, awaits `tools/list`. |
| `list_tools()` | Returns the cached tool descriptors as adapter-wrapped `Tool` instances. |
| `disconnect(name)` | Sends shutdown, joins the process. |
| `close()` | Disconnects every server; idempotent. |

Failures surface as `MCPConnectionError(server_name, phase, cause)` — the `phase` field is one of `connect`, `initialize`, `list_tools`, `sdk_missing`. See `exec.mcp.*` codes in [error_codes.md](error_codes.md).

### Manifest-driven MCP servers

A manifest can declare MCP servers under `tools.mcp_servers[]`:

```json
{
  "tools": {
    "built_in": ["Read", "Glob"],
    "external": ["fs_read", "github_list_repos"],
    "mcp_servers": [
      {
        "name": "filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-filesystem", "/sandbox"],
        "env": {}
      }
    ]
  }
}
```

`Pipeline.from_manifest_async` instantiates `MCPManager`, connects each declared server, and registers the discovered tools before the pipeline starts.

## CLI-side MCP wrap (claude_code_cli only)

This is the **inverse direction**: you give the spawned `claude` subprocess an MCP server that talks back to **your** tool registry. The CLI's LLM then sees your tools as `mcp__<server>__<tool>` and can call them natively inside its own agentic loop.

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

# Attach as request-level config (per-session, dynamic):
request.mcp_config = mcp_config

# OR as client-level config (static):
extras = {"mcp_config": mcp_config}
```

The argv builder emits `--mcp-config <json>` and `--strict-mcp-config`. The strict flag scopes the CLI's MCP surface to **only** what you provide — no user-level or project-level MCP servers leak in.

The bridge script (`geny_mcp_bridge.py` in the example above) is just an MCP-spec stdio loop that forwards JSON-RPC to your tool dispatcher. The pattern is fully decoupled from any specific transport library — Geny's reference implementation is ~130 lines, stdlib only.

Full integration details + a bridge skeleton: [claude_code_cli.md](claude_code_cli.md).

## Choosing between the two

| You want… | Use |
|---|---|
| Host LLM (any provider) calling a 3rd-party MCP server (FS, GitHub, …) | Host-attached `MCPManager` |
| `claude_code_cli` spawned CLI's LLM calling **your** tools | Per-session CLI MCP wrap |
| Both at once for a single session | Both — they don't conflict |

The host-attached path is the standard MCP client story. The CLI wrap is geny-executor-specific machinery that makes the `claude_code_cli` provider useful as a Stage 6 backend in an agentic pipeline.

## Error handling

| Code | Phase | Action |
|---|---|---|
| `exec.mcp.sdk_missing` | startup | `pip install mcp` |
| `exec.mcp.connect_failed` | `connect()` | Check the server binary + args |
| `exec.mcp.initialize_failed` | `initialize` RPC | Server is alive but rejected the handshake — check protocol version |
| `exec.mcp.list_tools_failed` | `tools/list` RPC | Initialize succeeded but tool listing failed — server-specific bug |

See [error_codes.md](error_codes.md) for the full table.
