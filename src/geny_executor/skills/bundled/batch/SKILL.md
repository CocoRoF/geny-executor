---
name: Batch Run
description: Apply the same prompt or operation to a list of items in turn, collecting results. Useful for "do X to each of these files / PRs / records" tasks.
category: workflow
effort: medium
when_to_use: When the user gives you a list and asks you to do the same thing to every entry — "review each of these PRs", "summarise each file in this dir", "rename every constant matching this pattern". Don't use for tasks that need per-item planning; use for genuinely uniform operations.
arguments:
  - items
  - operation
argument_hint: "items=<comma-separated list> operation=<verb phrase>"
examples:
  - 'items="src/auth.ts, src/db.ts, src/api.ts" operation="summarise the file"'
  - 'items="#101, #103, #107" operation="check if it''s ready to merge"'
  - 'items="users, orders, payments" operation="describe the table''s purpose"'
version: 1.0.0
---

# Batch — same operation, every item

You're applying one operation across a list. Be uniform: every
item should get the *same* depth and *same* output shape. The
caller's downstream pipeline almost certainly expects regular
output.

## Inputs

- `${items}` — the list. Comma-separated by default; if entries
  contain commas, the caller may have used newlines instead. Trim
  whitespace and drop empty entries.
- `${operation}` — what to do for each item. A short verb phrase
  ("summarise", "check status", "lint and report"). If missing,
  ask the user once — don't guess.

## Algorithm

1. Parse `items`. Confirm count back to the user one time at the
   start: "I'll run *operation* on N items: ..." (no need to list
   all if N is large; show first 3 + "...and M more").
2. For each item:
   - Run the operation.
   - Emit a fixed-shape result entry. The shape is **always**:
     ```
     ## <item>
     <findings — 1–3 sentences>
     **Status**: ok | warning | error | n/a
     ```
   - If a single item fails or is unanswerable, mark its **Status**
     as `error` / `n/a` with a one-line reason. Do *not* abort the
     whole batch.
3. After every item is processed, emit a one-paragraph summary:
   - How many ran cleanly?
   - Common patterns / common failures?
   - One follow-up suggestion (if any).

## Constraints

- **Don't think out loud per-item** — the loop should feel
  mechanical. Save reasoning for the final summary.
- **Don't re-order the list.** Whatever order the user provided is
  the canonical order; the user may rely on it for downstream
  processing.
- **Don't skip items silently.** Every input item must appear in
  the output (even if the entry is just "n/a — couldn't reach").

## When this is the wrong fit

If items differ enough that the same operation produces wildly
different outputs (e.g. "review this codebase" vs "review this
README"), bail out early: tell the user the items aren't uniform
and ask whether to split into per-type batches. A consistent
shape matters more than processing every item.
