---
name: Self-Modifying Environment
description: How to inspect and edit your OWN operating environment at runtime — system prompt, active tools, and skills — using the `env` tool. Use when you need to change your own capabilities (turn a tool on/off, rewrite your instructions, author a new skill) or save them for this session.
category: meta
effort: low
when_to_use: When you decide your current toolset / instructions / skills aren't right for the task and you want to change them yourself — e.g. enable a tool you lack, drop noisy tools, rewrite your system prompt, or capture a reusable procedure as a new skill. Read this before making non-trivial environment edits.
allowed_tools:
  - env
version: 1.0.0
---

# Editing your own environment

You can change your own operating environment with the **`env`** tool. Changes
are **session-scoped** (they affect only you, not other sessions or the shared
template) and take effect **from the next turn**. Every change is logged.

`env` takes an `action` and an `args` object. Always start with
`env(action="view")` to see what's active vs available before editing.

## Quick recipes

- **See your environment** — `env(action="view")`. Returns your prompt size,
  active vs available tools, active vs available skills, and change count.
- **Read your prompt** — `env(action="get_prompt")`.
- **Rewrite your prompt** — `env(action="set_prompt", args={"prompt": "<full new prompt>"})`.
  Replaces it entirely; read it first. Use `append_prompt` to add a section
  instead of replacing.
- **Add a section to your prompt** — `env(action="append_prompt", args={"text": "..."})`.
- **Turn a tool on** — `env(action="enable_tool", args={"name": "<tool>"})`.
  Only tools shown under `available_tools` in `view` can be enabled.
- **Turn a tool off** — `env(action="disable_tool", args={"name": "<tool>"})`.
- **Turn a skill on/off** — `env(action="enable_skill", args={"skill_id": "<id>"})`
  / `env(action="disable_skill", args={"skill_id": "<id>"})`.
- **Author a reusable skill** — `env(action="create_skill", args={"skill_id": "...",
  "description": "one line — when to use it", "body": "# Title\n\nstep-by-step
  instructions"})`. It is enabled immediately. Edit later with
  `env(action="edit_skill", args={"skill_id": "...", "body": "..."})`.
- **Review your changes** — `env(action="changelog")`.
- **Save** — `env(action="save")`. Persists your evolved environment for THIS
  session so it is restored next time. (If saving isn't configured the call
  reports that and your changes simply stay live for this session.)

## Principles

- **Edit deliberately, minimally.** Change one thing, verify on the next turn,
  then continue. Don't disable a tool you're mid-task with.
- **Stay within what's available.** You can only enable tools/skills the host
  made available (see `view`) and edit your prompt/skills — you cannot invent
  arbitrary tools.
- **Prefer enabling a tool over apologising for not having it.** If a task needs
  a capability that's in `available_tools`, enable it and proceed.
- **Capture repeated procedures as skills** (`create_skill`) so future turns get
  a concise, on-demand playbook instead of re-deriving it.

For the exact arguments + return shape of every action, load
`resource="REFERENCE.md"` — pull it only when you need the precise contract.
