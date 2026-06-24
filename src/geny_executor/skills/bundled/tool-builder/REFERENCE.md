# Sandbox Tool Builder — reference

## The execution contract

Every forged tool runs as:

```
<runtime> <entrypoint> [argv...]      # cwd = workdir (default /workspace), inside the sandbox
```

- The tool's **input** (the JSON object passed to the tool call) is written to
  the process **stdin** as a single JSON document.
- The process must write its **result** to **stdout** as a single JSON object.
- Anything on **stderr** is captured for diagnostics but is not the result.
- **Errors:** print a JSON object with an `"error"` field, OR exit non-zero. Both
  become a tool error the caller sees. A zero exit with non-JSON stdout is also
  reported as an error.

The tool runs **inside your sandbox** with the same filesystem you authored it
in. Code never executes on the host.

## forge_tool arguments

| arg | required | meaning |
|-----|----------|---------|
| `name` | yes | the tool name you'll call it by (must be unique among active tools) |
| `entrypoint` | yes | path to the script, relative to `workdir` (e.g. `tools/x/main.py`) |
| `description` | recommended | what it does — the caller (you, next turn) sees this |
| `input_schema` | recommended | JSON Schema for the request object; guides correct calls |
| `runtime` | default `python3` | interpreter/launcher: `python3`, `node`, `bash`, `ruby`, … |
| `argv` | optional | extra fixed args appended after the entrypoint |
| `timeout_s` | default `60` | per-call wall-clock limit |
| `workdir` | default `/workspace` | cwd for the run; entrypoint is resolved against it |
| `network_egress` | default `false` | allow the tool to reach the network |
| `read_only` | default `false` | mount the workspace read-only for the run |

## Non-Python runtimes

The runtime is just the launcher; the contract is identical (stdin JSON → stdout
JSON).

Node (`tools/slug/main.js`, `runtime: "node"`):
```javascript
let raw = "";
process.stdin.on("data", d => raw += d);
process.stdin.on("end", () => {
  const req = JSON.parse(raw || "{}");
  process.stdout.write(JSON.stringify({ slug: String(req.title||"").toLowerCase().replace(/\s+/g,"-") }));
});
```

Bash (`runtime: "bash"`) — read stdin, emit JSON (use `jq` if available).

## Designing input_schema

Make it precise so calls are correct the first time:
```json
{
  "type": "object",
  "properties": {
    "url":   {"type": "string", "description": "page to fetch"},
    "limit": {"type": "integer", "minimum": 1, "default": 20}
  },
  "required": ["url"]
}
```
Read your own values back inside the script with safe defaults
(`req.get("limit", 20)`), since callers may omit optionals.

## Multi-tool packs (+ skills)

A single workspace can hold **many** tools — `tools/a/main.py`,
`tools/b/main.js`, … — each forged separately. When several tools work together
(plus a how-to skill explaining the workflow), the host can persist them as **one
Sandbox Tool Pack**: `[isolated workspace snapshot] + [N tool specs] + [M
skills]`. That pack is the durable, reusable unit — re-enable it later, restore
it into a fresh workspace, or fork it.

When you build a pack, also author a short skill (use the **skillify** skill, or
`env(action="create_skill")`) that documents how the tools fit together, so a
future session knows how to use them.

## Iterating & replacing

- To change a forged tool: edit the script, re-test, then
  `env(action="disable_tool", args={"name": "..."})` followed by another
  `forge_tool`. (`forge_tool` refuses to clobber an active name.)
- Keep entrypoints stable (`tools/<name>/main.*`) so a saved pack restores to the
  same paths.

## Troubleshooting

| symptom | cause / fix |
|---------|-------------|
| `no sandbox is attached` | this session has no isolated workspace; you can't forge here |
| tool error with raw text | the script printed non-JSON on stdout — wrap output in `json.dumps` |
| `tool '<name>' is already active` | pick a new name, or `disable_tool` it first |
| works in shell, fails as a tool | shell test didn't pipe stdin as JSON — test with `echo '{...}' | <runtime> <entrypoint>` |
| times out | raise `timeout_s`, or make the script faster; long network calls need `network_egress: true` |
| can't reach network | set `network_egress: true` when forging |

## Why sandboxed

The tool's code is yours, freshly written — running it inside the sandbox keeps
the host safe and makes the tool **portable**: snapshot the workspace and the
exact same tool runs anywhere the pack is restored.
