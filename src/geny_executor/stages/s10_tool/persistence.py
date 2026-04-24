"""Tool result persistence — send large payloads to disk, short summary to LLM.

Cycle 20260424 executor uplift — Phase 2 Week 4 Checkpoint 2.

When a tool returns a payload larger than
``ToolCapabilities.max_result_chars``, ``maybe_persist_large_result`` writes
the full body to ``{storage_path}/tool-results/{tool_use_id}.json`` and
returns a new ``ToolResult`` with a short ``display_text`` pointing at the
persisted path. The LLM then sees the summary, while the host retains the
full payload for audit / replay / Stage 15 memory.

Fallback policy (fail-open to preserve correctness):

* If ``storage_path`` is not set, the result is returned untouched. The
  orchestrator will still inline the full payload this turn; the caller
  is responsible for attaching storage before running in production.
* If writing the file fails for any reason, the error is logged via the
  standard ``logging`` module and the original result is returned — the
  turn must not die because the on-disk sink is unavailable.
* ``max_result_chars=0`` disables the cap (matches the ``ToolCapabilities``
  contract: ``0`` = infinite).

See ``executor_uplift/06_design_tool_system.md`` §6 and
``executor_uplift/12_detailed_plan.md`` §2 (E-2.4).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from typing import Any, Dict, Optional

from geny_executor.tools.base import ToolCapabilities, ToolContext, ToolResult

_LOG = logging.getLogger(__name__)

# Subdirectory under ``ToolContext.storage_path`` where full tool result
# bodies are written. Kept public so hosts (Geny, Stage 15 memory) can
# enumerate or garbage-collect the sink.
TOOL_RESULTS_DIRNAME = "tool-results"


def _render_content(content: Any) -> str:
    """Render a tool result ``content`` payload to a string for size check.

    Mirrors the branching that ``ToolResult.to_api_format`` uses so the
    threshold we compare against matches the bytes the LLM would see.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _summarise(body: str, *, limit: int, path: str) -> str:
    """Short LLM-facing replacement for an oversized body.

    Shows the first ~480 characters of the persisted body so the model
    still has a peek at the content, followed by a pointer to the file
    and the total length. Callers may override by pre-setting
    ``display_text`` on the returned ``ToolResult``.
    """
    peek = body[:480].rstrip()
    if len(body) > 480:
        peek = peek + "…"
    return (
        f"[tool result truncated: {len(body)} chars > {limit} limit]\n"
        f"Full body persisted to: {path}\n"
        f"Preview:\n{peek}"
    )


def maybe_persist_large_result(
    result: ToolResult,
    *,
    tool_use_id: str,
    tool_name: str,
    capabilities: ToolCapabilities,
    context: ToolContext,
) -> ToolResult:
    """Return a possibly-rewritten ``ToolResult`` with large bodies on disk.

    Arguments:
        result: The raw tool output.
        tool_use_id: Stable call identifier used to name the persisted file.
        tool_name: Tool name (embedded in the persisted envelope for audit).
        capabilities: Resolved capabilities for this invocation. The
            ``max_result_chars`` field gates persistence.
        context: Tool execution context; its ``storage_path`` determines
            the sink directory.

    Returns:
        A new ``ToolResult`` when persistence happened, or the input
        ``result`` unchanged. ``display_text`` and ``persist_full`` on the
        returned value reflect the persisted-path + summary. When the
        tool already set ``display_text``, it is preserved — tools may
        choose their own summary, and the helper only needs to store the
        full body and populate ``persist_full``.
    """
    # Respect explicit opt-outs before even rendering the body.
    if capabilities.max_result_chars == 0:
        return result
    if capabilities.max_result_chars < 0:
        return result
    if result.persist_full:
        # Tool has already persisted externally; nothing to do.
        return result

    rendered = _render_content(result.content)
    if len(rendered) <= capabilities.max_result_chars:
        return result

    storage_path = context.storage_path
    if not storage_path:
        # No sink configured — log once per call so ops can spot misconfig,
        # but return the original result so correctness is preserved.
        _LOG.warning(
            "tool %s returned %d chars (> %d) but no storage_path; inlining the full payload",
            tool_name,
            len(rendered),
            capabilities.max_result_chars,
        )
        return result

    try:
        target_dir = os.path.join(storage_path, TOOL_RESULTS_DIRNAME)
        os.makedirs(target_dir, exist_ok=True)
        # Defensive: tool_use_ids from the API are safe; still sanitize.
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_use_id)
        target_path = os.path.join(target_dir, f"{safe_id}.json")

        envelope: Dict[str, Any] = {
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "is_error": result.is_error,
            "content": result.content,
            "metadata": result.metadata,
            "artifacts": result.artifacts,
        }
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(envelope, fh, ensure_ascii=False, default=str, indent=2)
    except OSError as exc:
        _LOG.warning(
            "failed to persist tool result for %s (%s): %s — inlining payload",
            tool_name,
            tool_use_id,
            exc,
        )
        return result

    summary: Optional[str] = result.display_text
    if summary is None:
        summary = _summarise(rendered, limit=capabilities.max_result_chars, path=target_path)

    return replace(
        result,
        display_text=summary,
        persist_full=target_path,
    )
