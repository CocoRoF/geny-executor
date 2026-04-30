---
name: Loop
description: Run the same task on a recurring cron schedule. Parses interval syntax (5m, 2h, 1d, "*/15 * * * *"), records the schedule, and acknowledges back to the user.
category: workflow
effort: low
when_to_use: When the user wants something to happen periodically without typing it again — "every 5 minutes check X", "every morning summarise Y", "every Friday at 17:00 run the Z report". Don't use for one-shot tasks; use for genuine recurrence.
allowed_tools:
  - Bash
shell: bash
shell_timeout_s: 5
arguments:
  - interval
  - task
argument_hint: '<interval> "<task description>"  e.g. 5m "check open PRs"'
examples:
  - 'loop every 30m to /verify-build'
  - "every Monday at 9am summarise unread issues"
  - "every hour, check disk usage and warn if >90%"
version: 1.0.0
---

# Loop — schedule a recurring task

You are recording a recurring instruction. The host is responsible
for the actual cron daemon — your job is to (1) parse what the user
asked for, (2) translate it into a canonical cron expression, and
(3) hand it off to the host's scheduler with a clear acknowledgement.

## Inputs

- `${interval}` — when to fire. Three syntaxes accepted:
  - **Compact**: `5m`, `30m`, `1h`, `2h`, `1d`, `12h`. Translate to
    cron.
  - **Cron expression**: `*/5 * * * *`, `0 9 * * 1` — pass through.
  - **Plain English**: "every weekday at 9am", "every 30 minutes",
    "every Monday morning". Convert to cron.
- `${task}` — what to do each tick. Free-form; quote it with
  double-quotes if it contains spaces.

If either is missing, ask once. Don't guess on `task` — a
mis-recorded recurring task is worse than none.

## Translate to cron

Standard 5-field POSIX cron: `minute hour day-of-month month day-of-week`.

Common compact translations:

| Compact | Cron expression | Meaning |
|---|---|---|
| `5m`  | `*/5 * * * *`  | every 5 minutes |
| `15m` | `*/15 * * * *` | every 15 minutes |
| `30m` | `*/30 * * * *` | every 30 minutes |
| `1h`  | `0 * * * *`    | every hour on the hour |
| `2h`  | `0 */2 * * *`  | every 2 hours |
| `4h`  | `0 */4 * * *`  | every 4 hours |
| `12h` | `0 */12 * * *` | every 12 hours |
| `1d`  | `0 0 * * *`    | every day at midnight |
| `1w`  | `0 0 * * 0`    | every Sunday at midnight |

For natural-language inputs, try to be conservative. "Every morning"
defaults to 09:00 unless the user said otherwise. "Every weekday at
5pm" → `0 17 * * 1-5`. If you're not sure, **echo back your
interpretation** and ask the user to confirm before scheduling.

## Hand off to the scheduler

This skill does **not** implement the cron daemon itself — the host
provides one (geny-executor's `cron` extra ships `croniter`-backed
scheduling; Geny exposes `ScheduleCron` / similar tools).

Your job:

1. Show the user the parsed schedule + task in one block:
   ```
   Schedule: <cron expression>  (every <human-readable description>)
   Task:     <task>
   ```
2. Hand the canonical schedule + task off to the host's scheduler.
   In a Geny-style host the next step is calling `ScheduleCron` /
   the equivalent tool with `(cron_expression, task)`. If no such
   tool is in the current tool roster, *don't fake it* — say so:
   "I parsed your request to <cron>. The current session doesn't
   have a scheduler tool wired; here's the canonical form to keep
   for when you set one up."
3. Confirm the schedule was registered with a single short
   sentence, including the cron expression so the user can edit it
   later if needed.

## Constraints

- **No schedule, no acknowledgement.** Don't acknowledge what you
  haven't actually scheduled. If the scheduler call fails, surface
  the failure, don't pretend the recurring task is live.
- **Don't run the task once "to test".** Loop is for recurring
  scheduling, not one-shot execution. If the user wants to verify
  the task works, they should run it directly first.
- **Don't accept "always" or "constantly".** A cron tick of 1
  minute is the practical floor. Push back gently if asked to
  schedule sub-minute work.
- **Don't lose the user's wording.** Show your interpretation
  back; many users have an exact target in mind ("every weekday
  morning" sometimes means 8am, sometimes 9am, sometimes 10am).
