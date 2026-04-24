"""Central catalogue of well-known keys for ``state.shared``.

Cycle 20260424 executor uplift — Phase 1 Week 2 Checkpoint 3.

``state.shared`` is a ``Dict[str, Any]`` — free-form by design so stages
and host plugins can stash arbitrary data. Without a convention two
unrelated components may collide on the same key.

This module provides:
- Canonical string constants for keys produced / consumed by executor
  core and officially-blessed host plugins (e.g. Geny's creature
  state). Use these instead of literal strings.
- Namespacing helper for third-party plugins: ``plugin_key("namespace",
  "key") → "plugin.namespace.key"``.

All constants are deliberately stable strings — renaming one is a
breaking change and requires a major bump.

See ``executor_uplift/09_design_extension_interface.md`` §4.
"""

from __future__ import annotations

from typing import Final


class SharedKeys:
    """Well-known keys for ``PipelineState.shared``."""

    # ── Executor core -------------------------------------------------

    TOOL_CALL_ID: Final = "executor.current_tool_call_id"
    """Identifier of the tool_use block currently being processed."""

    SKILL_CTX: Final = "executor.current_skill_ctx"
    """Context for an in-flight Skill invocation (Phase 3)."""

    PERMISSION_CACHE: Final = "executor.permission_cache"
    """Map of ``(tool_name, input_hash) → PermissionDecision`` to avoid
    re-checking the matrix for identical inputs in the same turn."""

    TOOL_REVIEW_FLAGS: Final = "executor.tool_review_flags"
    """List of annotations emitted by Stage 11 Tool Review (Phase 9)."""

    TASKS_NEW_THIS_TURN: Final = "executor.tasks_new_this_turn"
    """Tasks spawned by Stage 12 Agent in this iteration (Phase 9)."""

    TASKS_BY_STATUS: Final = "executor.tasks_by_status"
    """Dict keyed by TaskStatus for cross-stage inspection (Phase 9)."""

    HITL_REQUEST: Final = "executor.hitl_request"
    """Present when Stage 15 HITL should block for approval (Phase 9)."""

    HITL_DECISION: Final = "executor.hitl_decision"
    """Set by Pipeline.resume(token, decision) — consumed by Stage 16
    Loop to decide continue / error."""

    TURN_SUMMARY: Final = "executor.turn_summary"
    """SummaryRecord written by Stage 19 Summarize (Phase 9)."""

    LAST_CHECKPOINT_ID: Final = "executor.last_checkpoint_id"
    """Most recent persist id from Stage 20 Persist (Phase 9)."""

    # ── Memory -------------------------------------------------------

    MEMORY_CONTEXT_CHUNKS: Final = "memory.context_chunks"
    """Retrieved chunks injected by Stage 2 Context's retriever."""

    MEMORY_NEEDS_REFLECTION: Final = "memory.needs_reflection"
    """Boolean flag for deferred reflection — legacy Geny path."""

    # ── Geny (host plugin) -------------------------------------------

    GENY_CREATURE_STATE: Final = "geny.creature_state"
    """Tamagotchi creature state snapshot hydrated at turn start."""

    GENY_MUTATION_BUFFER: Final = "geny.mutation_buffer"
    """Pending creature-state mutations accumulated by game tools."""

    GENY_CREATURE_ROLE: Final = "geny.creature_role"
    """Creature role classifier bound per-session (vtuber etc.)."""

    # ── Helpers ------------------------------------------------------

    @staticmethod
    def plugin_key(namespace: str, key: str) -> str:
        """Build a namespaced key for third-party plugins.

        Example::

            SharedKeys.plugin_key("myplugin", "state") == "plugin.myplugin.state"

        Plugin authors should always use this helper — direct literal
        strings in user code risk collisions with a future executor
        release that claims the same name.
        """
        if not namespace or not namespace.isidentifier():
            raise ValueError(f"plugin namespace must be a valid identifier: {namespace!r}")
        if not key:
            raise ValueError("plugin key cannot be empty")
        return f"plugin.{namespace}.{key}"
