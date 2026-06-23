# `env` tool — full action reference

The `env` tool dispatches on `action` with an `args` object. All edits are
session-scoped and take effect on the next turn. Read actions return data;
write actions return a short status string (and set an error flag on failure).

## Read actions

### `view`
- args: none
- returns: object — `prompt_chars`, `prompt_editable`, `active_tools[]`,
  `available_tools[]`, `active_skills[]`, `available_skills[]`, `changes`,
  `persistable`.

### `get_prompt`
- args: none
- returns: your full current system prompt (string).

### `changelog`
- args: `{ "limit": <int?> }` — keep only the last N entries.
- returns: list of `{seq, action, target, detail, ok}`.

## Prompt edits

### `set_prompt`
- args: `{ "prompt": "<full new system prompt>" }`
- Replaces the entire system prompt. Requires a mutable prompt builder
  (`prompt_editable: true` in `view`); otherwise returns an error.

### `append_prompt`
- args: `{ "text": "<section to append>" }`
- Adds a section after the current prompt (keeps the base).

## Tool toggles

### `enable_tool`
- args: `{ "name": "<tool name>" }` — must be in `available_tools`.
### `disable_tool`
- args: `{ "name": "<tool name>" }` — must be in `active_tools`. The `env` tool
  itself cannot be disabled.

## Skill toggles

### `enable_skill` / `disable_skill`
- args: `{ "skill_id": "<id>" }` — must be in `available_skills` /
  `active_skills`. An enabled skill surfaces as a callable tool next turn.

## Skill authoring

### `create_skill`
- args: `{ "skill_id": "kebab-id", "description": "one line — when to use it",
  "body": "# Title\n\nprocedural instructions", "allowed_tools": ["..."]? }`
- Creates an inline skill and enables it immediately. `allowed_tools` is
  optional. Fails if the id already exists (use `edit_skill`).

### `edit_skill`
- args: `{ "skill_id": "...", "description": "..."?, "body": "..."?,
  "allowed_tools": ["..."]? }`
- Updates only the provided fields. If the skill is active, its surfaced tool is
  refreshed to the new body.

## Persistence

### `save`
- args: none
- Persists the overlay (prompt + active tools + active skills + authored skills
  + changelog) for THIS session via the host's persistence callback, so it is
  restored on resume. Returns an error string if no persistence is configured
  (changes still remain live for the current session).
