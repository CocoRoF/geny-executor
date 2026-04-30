---
name: Verify Project Setup
description: Check the host's runtime versions, required files, and recommended configs against what the project expects.
category: diagnostic
effort: medium
when_to_use: When the user is debugging an environment problem ("why doesn't this work on my machine?"), onboarding to a project, or validating CI parity. Run this *before* any nontrivial install / setup advice — it captures ground truth.
allowed_tools:
  - Read
  - Glob
shell: bash
shell_timeout_s: 15
version: 1.0.0
---

# Verify — host + project setup snapshot

The output below is *captured at skill execution time*. Read it,
note any mismatches, then explain to the user what's healthy, what's
suspicious, and what to fix next. Do **not** re-run these commands
yourself — the data is already here.

## Operating system + shell

- OS: !`uname -s`
- Kernel: !`uname -r`
- Architecture: !`uname -m`
- Shell: !`echo "$SHELL"`
- Working dir: !`pwd`

## Runtime versions

```!
for cmd in node npm pnpm bun yarn python python3 pip uv ruff git gh docker make; do
  if command -v "$cmd" >/dev/null 2>&1; then
    ver="$("$cmd" --version 2>&1 | head -n 1)"
    printf "  %-8s %s\n" "$cmd" "$ver"
  fi
done
```

## Project files (top-level)

- package.json: !`[ -f package.json ] && echo present || echo absent`
- pyproject.toml: !`[ -f pyproject.toml ] && echo present || echo absent`
- requirements.txt: !`[ -f requirements.txt ] && echo present || echo absent`
- Dockerfile: !`[ -f Dockerfile ] && echo present || echo absent`
- .env.example: !`[ -f .env.example ] && echo present || echo absent`
- README.md: !`[ -f README.md ] && echo present || echo absent`
- LICENSE: !`[ -f LICENSE ] && echo present || echo absent`

## Git state

- Repo root: !`git rev-parse --show-toplevel 2>/dev/null || echo "(not a git repo)"`
- Branch: !`git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(not a git repo)"`
- HEAD: !`git rev-parse --short HEAD 2>/dev/null || echo "(not a git repo)"`
- Dirty?: !`git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null && echo clean || echo dirty`

## Environment hints

- Node version manager: !`command -v fnm >/dev/null && echo "fnm" || (command -v nvm >/dev/null && echo "nvm" || echo "(none detected)")`
- Python in PATH points to: !`command -v python || command -v python3 || echo "(none)"`
- Active virtualenv: !`echo "${VIRTUAL_ENV:-(none)}"`

## How to interpret

Scan the captured output above. Report findings as:

1. **Healthy** — what's present and at a reasonable version.
2. **Suspicious** — anything that's missing-but-expected (e.g. no
   git repo when one is expected, no package.json in a node project),
   or a version that's notably old (Node < 18, Python < 3.10, etc.).
3. **Recommended next steps** — at most three concrete actions, in
   priority order. Skip if everything looks fine.

Keep the report scannable. Use bullets, not paragraphs.
