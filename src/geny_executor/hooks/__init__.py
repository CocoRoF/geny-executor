"""Subprocess hooks — user-configurable external observers / gates.

Cycle 20260424 executor uplift:
- Phase 1 Week 2 (PR #51): event taxonomy (``HookEvent``,
  ``HookEventPayload``, ``HookOutcome``).
- Phase 5 Week 9 (this PR): subprocess runner + configuration types
  + YAML loader. Stage 4 / Stage 10 wiring follows in the next PR.

Distinct from ``geny_executor.events.EventBus`` which is the
in-process pub/sub channel for observability:

- ``events.EventBus`` — in-process Python callbacks, used for UI
  updates / metrics / audit. Cannot block pipeline execution.
- ``hooks`` (this package) — external programs (shell scripts, Python,
  anything that speaks JSON) that can *block*, *deny*, or *modify*
  tool execution. Power + responsibility; opt-in via the
  ``GENY_ALLOW_HOOKS`` env var **and** ``HookConfig.enabled``.

See ``executor_uplift/09_design_extension_interface.md`` §3 and
``executor_uplift/12_detailed_plan.md`` §5.
"""

from geny_executor.hooks.config import (
    DEFAULT_TIMEOUT_MS,
    HOOKS_OPT_IN_ENV,
    HookConfig,
    HookConfigEntry,
    hooks_opt_in_from_env,
    load_hooks_config,
    parse_hook_config,
)
from geny_executor.hooks.events import (
    HookEvent,
    HookEventPayload,
    HookOutcome,
)
from geny_executor.hooks.runner import HookRunner

__all__ = [
    # events
    "HookEvent",
    "HookEventPayload",
    "HookOutcome",
    # config
    "HookConfig",
    "HookConfigEntry",
    "DEFAULT_TIMEOUT_MS",
    "HOOKS_OPT_IN_ENV",
    "hooks_opt_in_from_env",
    "load_hooks_config",
    "parse_hook_config",
    # runner
    "HookRunner",
]
