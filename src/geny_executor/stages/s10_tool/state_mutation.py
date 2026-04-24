"""Apply ``ToolResult.state_mutations`` onto ``PipelineState.shared``.

Cycle 20260424 executor uplift — Phase 3 Week 6.

The Tool ABC has carried a ``state_mutations`` attribute since the
Phase 1 foundation: a dict of proposed updates to ``state.shared`` that
the orchestrator applies on the tool's behalf. Until now the field was
declared but no stage consumed it — this module provides the
application side so tools like ``TodoWrite`` can finally propagate
their view into cross-stage state.

Rules:

* Only applied on **successful** tool calls. ``is_error=True`` results
  skip mutation — a failing tool must not leave its half-written state
  behind.
* Keys must use one of the documented namespaces (``executor.``,
  ``memory.``, ``geny.``, or ``plugin.<ns>.``). Unknown namespaces are
  logged and dropped — we don't want a misspelled key silently
  polluting ``state.shared``.
* Values are shallow-copied into ``state.shared``; callers are
  responsible for avoiding shared-reference aliasing hazards in
  mutable containers they hand over.
* Application is idempotent within a turn — two consecutive tool
  results writing the same key end up with the second value, same as
  plain dict assignment.

See ``executor_uplift/06_design_tool_system.md`` §6 and
``executor_uplift/09_design_extension_interface.md`` §4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from geny_executor.tools.base import ToolResult

logger = logging.getLogger(__name__)

# Allowed top-level prefixes for ``state.shared`` keys. Mirrors the
# SharedKeys namespace catalogue. Extending this list without adding a
# corresponding entry to SharedKeys is a code smell — the key should
# have a canonical constant.
_ALLOWED_NAMESPACES = ("executor.", "memory.", "geny.", "plugin.")


def _is_allowed(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in _ALLOWED_NAMESPACES)


def apply_state_mutations(
    result: ToolResult,
    shared: MutableMapping[str, Any],
    *,
    tool_name: str = "?",
) -> Dict[str, Any]:
    """Merge ``result.state_mutations`` into ``shared`` in place.

    Arguments:
        result: The raw ``ToolResult`` from a tool invocation.
        shared: The ``PipelineState.shared`` dict (or any mutable
            mapping). Updates are written in place.
        tool_name: Name of the tool, used in warning messages.

    Returns:
        A dict of the mutations that were actually applied (empty if
        the tool failed or proposed no changes). Useful for emitting
        a single ``tool.state_mutation`` event per call.
    """
    if result.is_error:
        return {}

    proposed = result.state_mutations or {}
    if not proposed:
        return {}

    applied: Dict[str, Any] = {}
    for key, value in proposed.items():
        if not isinstance(key, str):
            logger.warning(
                "tool %s proposed state mutation with non-string key %r; skipping",
                tool_name,
                key,
            )
            continue
        if not _is_allowed(key):
            logger.warning(
                "tool %s proposed state mutation with unknown namespace: %r. "
                "Valid prefixes: %s (see SharedKeys catalogue)",
                tool_name,
                key,
                _ALLOWED_NAMESPACES,
            )
            continue
        shared[key] = value
        applied[key] = value
    return applied


def apply_mutations_for_results(
    results: Iterable[tuple[str, ToolResult]],
    shared: MutableMapping[str, Any],
    *,
    on_apply: Optional[Mapping[str, Any]] = None,  # type: ignore[assignment]
) -> Dict[str, Any]:
    """Apply mutations for a batch of ``(tool_name, result)`` pairs.

    Kept separate from the per-result helper so executors can choose
    whether to apply inline (one call per tool) or batch-apply at the
    end. Returns the union of every applied mutation.
    """
    final: Dict[str, Any] = {}
    for tool_name, result in results:
        applied = apply_state_mutations(result, shared, tool_name=tool_name)
        final.update(applied)
    return final
