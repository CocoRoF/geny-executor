---
name: Skillify
description: Interview the user about a workflow they keep repeating, then generate a reusable SKILL.md and write it to disk so the next session starts with the skill already loaded.
category: meta
effort: medium
when_to_use: When the user says "we keep doing this same flow" or "can we make this a slash command" or "every time I onboard, I run these same five commands". The trigger is recognising a *repeated* workflow worth capturing — not one-off requests.
allowed_tools:
  - Read
  - Write
  - Bash
shell: bash
shell_timeout_s: 10
version: 1.0.0
---

# Skillify — capture a repeated workflow as a skill

You are going to interview the user, distill what they keep doing
into a SKILL.md, and write it to the right place on disk. The output
lives at `~/.geny/skills/<id>/SKILL.md` (user-scope) or
`<project>/.geny/skills/<id>/SKILL.md` (project-scope) depending on
how broadly the workflow applies.

## Step 1 — Sniff the context

Before asking anything, glance at the recent conversation. The
workflow being captured is *probably* something the user just did
(or just asked to redo). If you can already infer the shape, lead
with a one-sentence summary and ask the user to confirm.

```!
# Where would the project-scope skill go?
[ -d .geny/skills ] || mkdir -p .geny/skills
echo ".geny/skills/ ready under $(pwd)"
```

## Step 2 — Interview

Ask, in this order, one question at a time. Wait for an answer
before the next question. **Don't ask all at once** — operators get
overwhelmed and answer poorly.

1. **What should this skill do, in one sentence?** This becomes the
   `description` frontmatter field. Push back gently if the answer
   is vague — "produces a report" is too broad; "produces a markdown
   summary of changed files between two git refs" is right.

2. **What's a short id?** kebab-case, ≤ 32 chars, must match
   `[a-z0-9][a-z0-9_-]*`. Suggest one based on the description if
   the user shrugs.

3. **What arguments does it take?** Ask for argument names and a
   one-line hint. If none, that's fine — many skills are
   parameterless.

4. **Which tools does it need?** Pick from the host's actual tool
   roster (Read / Write / Bash / Edit / Grep / Glob, plus any
   custom). Default to `[Read, Bash]` if the user is unsure.

5. **When should the model use it?** This is the `when_to_use`
   field — extra discovery copy beyond the description. Examples:
   "When the user asks for a release-note draft."

6. **User scope or project scope?** User scope if it applies
   regardless of which repo they're in (e.g. "PR draft generator",
   "verify"). Project scope if it depends on this codebase's
   layout (e.g. "run the project's specific test sweep").

## Step 3 — Show the draft

Echo the proposed SKILL.md contents to the user. Use a fenced YAML
+ markdown block. Ask: "Does this look right? I can adjust before
writing."

The shape should be:

```yaml
---
name: <Human-readable name>
description: <One-line summary>
when_to_use: <Optional: longer discovery copy>
arguments: [<arg1>, <arg2>]   # only if any
argument_hint: "<hint>"        # only if arguments present
allowed_tools: [<tool1>, ...]
category: <utility | workflow | diagnostic | meta>
effort: <low | medium | high>
version: 1.0.0
---
```

Followed by the markdown body, structured as:

- One-paragraph intro that re-states the goal.
- Numbered steps for the model to follow.
- A short "What to avoid" or "Constraints" section.

## Step 4 — Write to disk

Once the user approves, write the file.

User-scope path: `${HOME}/.geny/skills/<id>/SKILL.md`
Project-scope path: `<cwd>/.geny/skills/<id>/SKILL.md`

Create the parent directory first; the Write tool needs it. After
writing, confirm with:

```
Wrote skill: /<full path>
Reload skills (or restart the session) to make /<id> available.
```

## Step 5 — Don't loop forever

If, after asking three questions, the user is bored or the answers
are all "I dunno", **stop**. Skillify is for capturing *concrete*
repeated workflows. If there's no repeated workflow, you're
manufacturing a skill nobody will use. Tell the user honestly:
"This doesn't sound like a recurring flow yet — let's revisit once
you've done it a couple more times."

## Constraints

- Don't generate more than one skill per invocation. If the user
  describes two workflows, ask which to capture first.
- Don't include shell blocks (` ```! ` / `` !`...` ``) unless the
  user explicitly mentioned a step that needs them. Inline shell is
  optional, not default.
- Don't ship a draft that uses `execution_mode: fork` unless the
  user explicitly wants a separate sub-agent run.
- Validate: after writing, run a quick `Read` on the file to
  confirm it landed correctly.
