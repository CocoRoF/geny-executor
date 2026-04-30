---
name: Debug Snapshot
description: Capture the host's environment, working directory, recent git activity, and pipeline-relevant state so you can diagnose "what's actually happening on the machine".
category: diagnostic
effort: medium
when_to_use: When the user reports something acting weird, a tool calls fail unexpectedly, or you need to "ground" your reasoning in the real machine state before suggesting fixes. Don't use this just to look around — use it when there's a concrete problem to triangulate.
allowed_tools:
  - Read
  - Glob
shell: bash
shell_timeout_s: 20
version: 1.0.0
---

# Debug — host + session diagnostic snapshot

The captured snapshot below is the ground truth. Read it carefully
before suggesting any fix — many "obvious" fixes are wrong because
the bug is one layer up (wrong PATH, wrong cwd, stale lock file,
unset env var, dirty worktree).

## Identity

- User: !`id -un`
- Groups: !`id -Gn`
- Hostname: !`hostname`
- Uptime: !`uptime`

## Filesystem context

- cwd: !`pwd`
- cwd writable?: !`[ -w "$(pwd)" ] && echo yes || echo no`
- Disk free (cwd): !`df -h "$(pwd)" 2>/dev/null | tail -n 1 || echo "(df failed)"`
- Inode pressure: !`df -i "$(pwd)" 2>/dev/null | tail -n 1 || echo "(df failed)"`

## Recently-touched files (last 5 minutes, current tree)

```!
find . -maxdepth 3 -type f -mmin -5 -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/__pycache__/*" 2>/dev/null | head -n 20
```

## Git activity (last 10 commits, current branch)

```!
git log --oneline --decorate -n 10 2>/dev/null || echo "(not a git repo)"
```

## Open ports / running listeners (TCP, IPv4)

```!
{ ss -tln 2>/dev/null || netstat -tln 2>/dev/null || lsof -iTCP -sTCP:LISTEN -n 2>/dev/null; } | head -n 30
```

## Resource snapshot

- Free memory: !`free -h 2>/dev/null | head -n 2 || echo "(free unavailable)"`
- Load average: !`cat /proc/loadavg 2>/dev/null || uptime | sed 's/.*load average: //'`
- Open file limit: !`ulimit -n`

## Selected env vars

```!
env | grep -E '^(PATH|PYTHONPATH|NODE_ENV|VIRTUAL_ENV|HOME|USER|LANG|TZ|GENY_|ANTHROPIC_|OPENAI_)' | sed 's/=\(sk-\|ghp_\|gho_\)[A-Za-z0-9_-]*/=<redacted>/'
```

## How to interpret

The data above is the input — your job is the output:

1. **Spot the anomaly.** What's different from a clean dev box?
   Stale lock file? Stuck process on an unexpected port? cwd not
   writable? Dirty worktree masking a real fix?
2. **Tie it to the user's reported symptom.** If they said "build
   fails", point at the disk-full / inode-exhausted / permission
   mismatch. If they said "tests hang", point at the listener /
   load / process state.
3. **Recommend at most one or two surgical fixes.** Don't shotgun.
   Diagnostic skills earn trust by being precise.

If the snapshot looks clean and you can't tie anything to the
symptom, say so plainly and ask for a more specific reproducer.
