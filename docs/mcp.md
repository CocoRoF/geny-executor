# MCP Integration

> Status: current for geny-executor 2.68.0.

geny-executor supports the [Model Context Protocol](https://modelcontextprotocol.io/) at **one boundary**: the host connects the server, the pipeline registers its tools, and Stage 10 dispatches them like any other tool.

| Boundary | Where the MCP server lives | Who serves it | Who consumes it |
|---|---|---|---|
| **Host-attached MCP** (`MCPManager`) | Process spawned + managed by the executor host | Filesystem / GitHub / Slack / any external MCP server | The pipeline's `ToolRegistry` — so every provider sees the tools, identically |

There used to be a second boundary: a CLI MCP wrap that handed servers to the `claude_code_cli` subprocess through `--mcp-config`, because that backend ran its own agentic loop and host-side connections were invisible to it. That provider was removed in 2.68.0 and the wrap with it. No backend owns the loop now, so there is exactly one place an MCP server connects and exactly one place its tools are dispatched.

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

## One place, every provider

A manifest's `tools.mcp_servers` are connected host-side by `MCPManager` at `from_manifest_async` time and registered into the `ToolRegistry` — whichever provider Stage 6 names. `APIRequest.mcp_config` survives on the request type for third-party clients that take MCP servers on their own channel; no shipped client does.

## Error handling

| Code | Phase | Action |
|---|---|---|
| `exec.mcp.sdk_missing` | startup | `pip install mcp` |
| `exec.mcp.connect_failed` | `connect()` | Check the server binary + args |
| `exec.mcp.initialize_failed` | `initialize` RPC | Server is alive but rejected the handshake — check protocol version |
| `exec.mcp.list_tools_failed` | `tools/list` RPC | Initialize succeeded but tool listing failed — server-specific bug |

See [error_codes.md](error_codes.md) for the full table.
