"""Subprocess hooks — user-configurable external observers / gates.

Cycle 20260424 executor uplift — Phase 1 Week 2 Checkpoint 3.

This package defines the *taxonomy* and *payload* of events that can
drive subprocess hooks (invoked via JSON stdin/stdout).  Distinct from
``geny_executor.events.EventBus`` which is the in-process pub/sub
channel for observability:

- ``events.EventBus`` — in-process Python callbacks, used for UI
  updates / metrics / audit. Cannot block pipeline execution.
- ``hooks`` (this package) — external programs (shell scripts, Python,
  anything that speaks JSON) that can *block*, *deny*, or *modify*
  tool execution. Power + responsibility; opt-in via environment var.

The runner implementation lands in a later checkpoint (Phase 5). This
checkpoint defines the event taxonomy and payload schema so Stage 4 /
Stage 10 code can import them now.

See ``executor_uplift/09_design_extension_interface.md`` §3.
"""

from geny_executor.hooks.events import (
    HookEvent,
    HookEventPayload,
    HookOutcome,
)

__all__ = [
    "HookEvent",
    "HookEventPayload",
    "HookOutcome",
]
