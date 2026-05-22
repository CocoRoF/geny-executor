# Hooks — PRE/POST tool-use lifecycle

> Status: current for geny-executor 2.1.0.

geny-executor's Stage 10 (Tool) fires lifecycle hooks around every tool dispatch. Hosts use this to:

- veto a tool call before it runs (permission / audit policy)
- mutate the tool input or context just before dispatch
- inspect the tool result before re-prompting the LLM
- emit telemetry (durations, costs, audit log entries)
- recover from tool failures with a synthetic success / error

Hooks are wired through `HookRunner` and consumed by `RegistryRouter` inside Stage 10. They're the official extension point for cross-cutting tool-layer concerns — there's no need to monkey-patch the dispatcher.

## The three lifecycle events

| Event | When | Can mutate? | Can veto? |
|---|---|---|---|
| `PRE_TOOL_USE` | Just before `tool.execute()` is invoked | input + context | ✅ — raise `ToolFailure` |
| `POST_TOOL_USE` | After a successful execute | result content | ❌ — informational |
| `POST_TOOL_FAILURE` | After a failed execute (raised `ToolFailure` / unexpected exception) | error metadata | ❌ — informational |

The contract lives in `hooks/runner.py`. Hooks are coroutines; ordering follows insertion order; exceptions propagate (a misbehaving hook fails the whole tool call — by design, since hooks gate the dispatch).

## Wiring a hook

```python
from geny_executor.hooks import HookRunner, HookEvent
from geny_executor.tools.errors import ToolFailure, ToolErrorCode

async def audit_pre(event: HookEvent) -> None:
    print(f"[audit] {event.session_id} → {event.tool_name}({list(event.tool_input.keys())})")

async def enforce_no_shell(event: HookEvent) -> None:
    if event.tool_name == "Bash":
        raise ToolFailure(
            "Shell access disabled for this session",
            code=ToolErrorCode.ACCESS_DENIED,
        )

async def cost_telemetry(event: HookEvent) -> None:
    # POST event carries event.duration_ms / event.result
    metrics.histogram("tool.duration_ms", event.duration_ms,
                      tags={"tool": event.tool_name})

runner = HookRunner()
runner.on_pre_tool_use(audit_pre)
runner.on_pre_tool_use(enforce_no_shell)
runner.on_post_tool_use(cost_telemetry)
runner.on_post_tool_failure(cost_telemetry)
```

Attach the runner to the pipeline context so Stage 10 picks it up:

```python
pipeline.attach_runtime(hook_runner=runner)
```

After this, every tool dispatch fires through the runner. No stage-level config needed.

## `HookEvent` payload

| Field | Notes |
|---|---|
| `event_type` | `pre_tool_use` / `post_tool_use` / `post_tool_failure` |
| `tool_name` | Resolved tool name (post-MCP normalisation if applicable). |
| `tool_input` | Dict the LLM provided. Mutable on `pre_tool_use`. |
| `context` | `ToolContext` (session_id, working_dir, stage info, permission_mode, …). |
| `result` | `ToolResult` (only on `post_tool_use`). |
| `error` | `ToolError` (only on `post_tool_failure`). |
| `duration_ms` | Wall-clock from execute-start (only on the two POST events). |
| `session_id` | Owning session, when available. |

## Common patterns

### Permission-rule evaluation

Geny ships a permission matrix evaluated as a `PRE_TOOL_USE` hook. The matrix is per-session and reads from a typed rule schema; on a deny it raises `ToolFailure(code=ACCESS_DENIED)` which the router converts into a structured `ToolError` for the LLM. No tool needs to know it's gated.

### Auto-injecting session context

Some tools (e.g. `send_direct_message_internal`) want the caller's `session_id` but the LLM shouldn't have to thread it. A `PRE_TOOL_USE` hook can inject it:

```python
async def inject_session_id(event):
    sig = inspect.signature(event.tool.execute)
    if "session_id" in sig.parameters and "session_id" not in event.tool_input:
        event.tool_input["session_id"] = event.context.session_id
```

### Cost telemetry → host metrics

A `POST_TOOL_USE` hook reads `event.duration_ms` and `event.result.content` length, emits to Datadog / Prometheus / whatever. The pipeline's own event bus carries the same data — pick the surface that fits your observability stack.

### Sub-worker delegation tracking

Geny tracks which session triggered a `send_direct_message_internal` from inside a `POST_TOOL_USE` hook so the runtime can correlate VTuber ↔ Sub-Worker pairs without the LLM ever seeing the linked session id.

## When NOT to use a hook

| Don't | Do |
|---|---|
| Re-implement tool dispatch inside a hook | Define a custom `Tool` subclass |
| Hide LLM behaviour from the event bus | Subscribe to `tool.call_start` / `tool.call_complete` events directly |
| Mutate the tool registry from inside a hook | Use `pipeline.attach_runtime(...)` at session create time |

Hooks are for **cross-cutting decisions around the call site**, not for replacing the call itself. If you find yourself doing real dispatch work in a hook, you probably want a router or a wrapping `Tool`.
