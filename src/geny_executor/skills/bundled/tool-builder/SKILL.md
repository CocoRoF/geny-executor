---
name: Sandbox Tool Builder
description: How to build a NEW tool of your own by writing its code in your sandbox, testing it, and forging it into a live, callable tool with `env(action="forge_tool")`. Use when no existing tool does what you need and you can implement it as a small script. Multiple tools + skills can be saved together as one reusable Sandbox Tool Pack.
category: meta
effort: medium
when_to_use: When the task needs a capability none of your current tools provide, and you can write it as a short script (parse/transform/compute/call something). Build it once, then call it like any other tool — and optionally persist it for reuse across sessions.
version: 1.0.0
---

# Building your own sandboxed tool

When you lack a tool you need, **write one**. A sandboxed tool is just a script
in your workspace that reads a JSON request on stdin and writes a JSON response
on stdout. You author it, test it, then `forge_tool` registers it as a real tool
you can call from the next turn — and it always runs **inside your sandbox**, not
on the host.

You need a sandbox (an isolated workspace) attached to this session. If you
don't have one, `forge_tool` will tell you.

## The loop

1. **Write the script** into your workspace with your file-write tool, at a
   stable path like `tools/<name>/main.py`. The contract:
   - read the whole request from **stdin** (a JSON object),
   - write the result to **stdout** as a JSON object,
   - on failure, print `{"error": "..."}` (or exit non-zero) — that surfaces as
     a tool error.

   Minimal Python example (`tools/wordcount/main.py`):
   ```python
   import sys, json
   req = json.load(sys.stdin)
   text = req.get("text", "")
   print(json.dumps({"words": len(text.split()), "chars": len(text)}))
   ```

2. **Test it** with your shell tool, exactly how the tool will be invoked:
   ```
   echo '{"text":"a b c"}' | python3 tools/wordcount/main.py
   ```
   Iterate until the output JSON is right.

3. **Forge it** — register it live:
   ```
   env(action="forge_tool", args={
     "name": "wordcount",
     "description": "Count words and characters in text.",
     "entrypoint": "tools/wordcount/main.py",
     "runtime": "python3",
     "input_schema": {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
   })
   ```
   From the **next turn** you can call `wordcount({"text": "..."})` like any tool.

4. **Use it.** Call it by name. If it misbehaves, fix the script, test again, and
   re-`forge_tool` (disable the old one first if the name clashes).

## Persisting it (reuse across sessions)

`forge_tool` is for **this session**. To keep it, save a **Sandbox Tool Pack**:

```
env(action="save_pack", args={"name": "text-utils", "description": "word + slug tools"})
```

This snapshots your workspace (code + artifacts) together with **every tool you
forged + every skill you authored** this session, and stores it as one reusable
unit (default disabled until an owner enables it for an environment). Pass
`tools`/`skills` (lists of names) to include only some. A pack can bundle
**several tools + several skills** and later be enabled per-environment, restored
into a fresh workspace, or forked.

See **REFERENCE.md** for: the full stdin/stdout contract, non-Python runtimes,
multi-tool packs, `input_schema` design, network/read-only options, and
troubleshooting.
