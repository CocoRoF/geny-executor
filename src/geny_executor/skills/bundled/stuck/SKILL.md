---
name: Get Unstuck
description: Pause, surface the current plan + open questions, and suggest the next concrete step so the agent (or operator) can recover from a deadlock or analysis paralysis.
category: meta
effort: low
when_to_use: When the conversation has been spinning — repeated tool calls without progress, contradictory edits, the agent admitting it doesn't know what to do next, or the operator typing "you're stuck" / "let's try again". Don't use this as a first response — use it when the rope's already in a knot.
version: 1.0.0
---

# Stuck — recovery checklist

Stop. Don't make another tool call. Read this whole skill before
deciding what to do next.

## What "stuck" actually looks like

You probably got here because one of these is true:

- The same file keeps being edited and reverted, or the same tool
  keeps being called with slightly different inputs and the same
  result.
- You've explained a plan three different ways and the user is
  still unhappy.
- A test / build keeps failing and your fixes aren't moving the
  failure mode, just relocating it.
- You're about to open a PR / send a commit but you can't summarise
  what changed.

If none of those describe the moment, this skill probably isn't
what you needed.

## Recovery — five-step recipe

### 1. State the original goal in one sentence

What was the *user's* original ask? Not your interpretation, not
the third-derivative subgoal you ended up chasing. The literal
prompt or its plain-English rephrasing.

### 2. List what's actually been done

Not what you tried, not what you intended. What's *committed to
disk* or *visible in the conversation* right now. Bullet list of
≤ 5 items.

### 3. List the open question(s)

What single piece of information would unblock you? It's almost
always one of:

- A clarification from the user (which of two interpretations,
  what counts as "done", which file is canonical).
- A reading you skipped (a config you haven't opened, a log line
  you haven't grepped, a manifest you assumed about).
- A test / repro you don't have (you're pattern-matching on a bug
  report instead of running the code).

### 4. Propose one concrete next step

Pick the smallest action that produces *new information*. Not a
plan, not a refactor — one Read, one Bash command, one user
question. The goal is to break the symmetry of being stuck.

### 5. Surface a "stop here?" prompt

End with a single yes/no question to the user. Examples:

- "I think the issue is X. Should I try Y, or did you mean Z?"
- "Before I keep editing, can you confirm the file at PATH is the
  right one?"
- "I've made N changes; should I commit what's there now and
  continue from a clean state?"

Make the question answerable with one short reply. The point is
to put the user back in the driver's seat without dumping your
state on them.

## What to *avoid* in this skill

- Don't write code.
- Don't make tool calls (Read / Bash / Edit / etc.) inside this
  skill — that's how you got stuck.
- Don't apologise more than once.
- Don't propose three options. One next step, one question.

## After the user replies

Discard the rest of your prior plan. Treat the user's answer as the
new starting point. The whole point of this skill is to *reset*,
not to continue.
