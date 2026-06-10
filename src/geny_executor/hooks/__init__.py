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
- ``hooks`` (this package) — gates that can *block*, *deny*, or
  *modify* tool execution. Two layers with separate opt-ins (split in
  2.2.0, audit 2026-06-09 §1-5):

  - subprocess hooks (external programs speaking JSON) — opt-in via
    the ``GENY_ALLOW_HOOKS`` env var **and** ``HookConfig.enabled``;
  - in-process handlers (``HookRunner.register_in_process``) — gated
    by ``HookConfig.enabled`` alone. The env var scopes subprocess
    *spawning*, not Python callback dispatch.

Taxonomy honesty: only the events listed in ``FIRED_EVENTS`` are
emitted by the engine today — the rest of :class:`HookEvent` is
reserved schema. Check the set before binding handlers.

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
    FIRED_EVENTS,
    HookEvent,
    HookEventPayload,
    HookOutcome,
)
from geny_executor.hooks.runner import HookRunner

__all__ = [
    # events
    "FIRED_EVENTS",
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
