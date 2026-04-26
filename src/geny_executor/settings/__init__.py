"""Hierarchical settings.json loader (PR-B.3.1).

Mirrors claude-code-main's settings.json shape: user → project →
local cascade with deep-merged JSON dicts. Each "section" is a
dict under a top-level key; hosts register pydantic-style schemas
to validate sections at access time.

Example layout::

    ~/.geny/settings.json              (user)
    .geny/settings.json                (project)
    .geny/settings.local.json          (local; gitignored)

::

    {
      "permissions": { "mode": "acceptEdits", "rules": [...] },
      "hooks": { "enabled": true, "entries": {...} },
      "skills": { "user_skills_enabled": true },
      "model": { "default": "claude-haiku-4-5-20251001" },
      "telemetry": { "enabled": false },
      "notifications": { "endpoints": [...] },
      "preset": { "default": "worker_adaptive" }   # service-specific section
    }
"""

from geny_executor.settings.loader import (
    SettingsLoader,
    get_default_loader,
    reset_default_loader,
)
from geny_executor.settings.section_registry import (
    get_section_schema,
    list_section_names,
    register_section,
)

__all__ = [
    "SettingsLoader",
    "get_default_loader",
    "get_section_schema",
    "list_section_names",
    "register_section",
    "reset_default_loader",
]
