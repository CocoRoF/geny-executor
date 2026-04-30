---
name: Simplify
description: Review code in the current working tree across three independent dimensions (reuse, quality, efficiency) and synthesise a small punch-list of high-value cleanups.
category: workflow
effort: high
when_to_use: When the user asks for a code review, smell check, or "what should I clean up before merging" pass. Best on a focused diff (one feature, one bug fix) — running this on a sprawling rebase will surface noise. Don't use for "review the whole repo" requests; ask the user to scope to a branch / PR / file set first.
allowed_tools:
  - Read
  - Grep
  - Glob
  - Bash
shell: bash
shell_timeout_s: 30
arguments:
  - target
argument_hint: "<file | dir | git-range>  (optional; defaults to current working tree)"
examples:
  - "Review the changes I just made"
  - "Simplify src/auth/"
  - "Pick the top 3 cleanups in the diff vs main"
version: 1.0.0
---

# Simplify — three-pass code review

Run three passes over the target. Keep each pass narrow: don't let
the reuse reviewer second-guess the quality reviewer, etc. After
all three passes, *synthesise* — pick the smallest set of changes
that improves the most dimensions.

## Step 0: Locate the target

The user passed `${target}`. Resolve it:

- If empty, default to the current diff vs the merge base of HEAD
  (or vs `main` / `master` when the merge base is hard to find):
  ```!
  git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1 2>/dev/null || git ls-files --modified
  ```
- If a path: read it directly (or, for a directory, glob for code
  files).
- If a git range (e.g. `HEAD~3..HEAD`, `feature-branch...main`):
  walk the diff.

If the target turns up empty, stop and ask the user what they'd
like reviewed.

## Step 1 — Reuse pass

Pass criterion: **what is being re-implemented that already exists?**

Look for:
- Utility functions duplicated across files. Same logic, different
  imports.
- Inline implementations that bypass an existing helper / library
  the project already depends on.
- Constants / enums redeclared per-call-site instead of imported
  once.
- Shape coercion / conversion code that an existing serializer
  already handles.

Output (this pass): bullet list of duplicated / re-implemented
things, each with the existing canonical location + the new
duplicate location.

## Step 2 — Quality pass

Pass criterion: **what would a reviewer flag in a PR comment?**

Look for:
- Names that don't match what they hold (`data`, `temp`, `info`).
- Functions doing more than one thing (and / or in the function
  signature is a tell).
- Error handling that swallows information (`except: pass`, raw
  rethrow with no context).
- Comments that no longer describe the code.
- Dead code (unreachable branches, unused imports / params, no
  callers).
- Magic numbers that lack a named constant.

Output (this pass): bullet list, each item naming the file:line +
the smell.

## Step 3 — Efficiency pass

Pass criterion: **what's doing too much work?**

Look for:
- O(n²) loops over data that's almost certainly small but might
  not stay small.
- Repeated I/O (file reads, network calls) inside loops.
- Sync calls inside async paths (or vice versa) that block the
  event loop.
- Caches that are missed because keys vary spuriously
  (e.g. dict iteration order, time fields in cache keys).
- Imports inside hot paths.

Output (this pass): bullet list with the locus + the cost.

## Step 4 — Synthesis

Now look at all three lists together. Output **a single punch-list**:

```
## Punch list

1. <verb> at <file:line> — covers reuse + quality
2. <verb> at <file:line> — covers efficiency
3. <verb> at <file:line> — covers reuse
```

Rules for the punch list:

- **Maximum 5 items.** This skill is about prioritisation, not
  exhaustive enumeration. If the three passes turned up 30 items,
  pick the 5 with the highest combined dimension count.
- Each item is a verb phrase (`extract X into Y`, `replace X with Y`,
  `delete dead branch in Z`).
- Skip cosmetic things if substantive things exist. No "rename
  variable" if there's a real bug nearby.

End with one sentence: should the user act on these now, or note
them for a follow-up PR? Default: "do the top 1-2, defer the rest."
