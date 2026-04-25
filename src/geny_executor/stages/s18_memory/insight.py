"""Insight record schema for Stage 15 structured reflection (S7.9).

Stage 15's legacy :class:`ReflectiveStrategy` flags
``state.metadata["needs_reflection"] = True`` and leaves the actual
extraction to a downstream consumer. The structured reflection path
keeps the *contract* explicit by re-using
:class:`geny_executor.memory.provider.Insight` (and its
:class:`Importance` grade) as the canonical record shape — kind, content,
importance, evidence/tags — instead of inventing a parallel schema.

This module exposes:

* :data:`PENDING_INSIGHTS_KEY` / :data:`INSIGHTS_KEY` — the agreed-upon
  ``state.metadata`` keys used by callers (host code, sub-agents) to
  queue insights and by :class:`StructuredReflectiveStrategy` to drain
  them into the running collection.
* :func:`record_insight` — convenience for stages and hosts to append an
  insight without touching ``metadata`` directly.
* :func:`coerce_insight` — normalises whatever a caller dropped into the
  pending queue (dict, dataclass, ``Insight``) into a real
  :class:`Insight`.

The strategy itself lives in ``artifact/default/strategies.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance, Insight

PENDING_INSIGHTS_KEY = "memory.pending_insights"
INSIGHTS_KEY = "memory.insights"


def _normalise_importance(value: Union[Importance, str, None]) -> Importance:
    if value is None:
        return Importance.MEDIUM
    if isinstance(value, Importance):
        return value
    try:
        return Importance(str(value).lower())
    except ValueError as exc:
        raise ValueError(
            f"unknown importance value: {value!r} "
            f"(expected one of: {[m.value for m in Importance]})"
        ) from exc


def coerce_insight(value: Any) -> Insight:
    """Normalise any caller-supplied payload into an :class:`Insight`.

    Accepts:
      * an :class:`Insight` instance — returned unchanged.
      * a mapping with at least ``title`` and ``content`` keys.

    Raises ``TypeError`` for anything else and ``ValueError`` for
    missing required fields. The strategy uses this helper so callers
    can queue plain dicts (the common case for cross-language hosts)
    without losing schema validation.
    """
    if isinstance(value, Insight):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"insight payload must be Insight or Mapping, got {type(value).__name__}")
    title = value.get("title")
    content = value.get("content")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("insight payload requires a non-empty 'title'")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("insight payload requires a non-empty 'content'")
    tags_raw = value.get("tags") or []
    if not isinstance(tags_raw, (list, tuple)):
        raise ValueError("insight 'tags' must be a list/tuple of strings")
    tags = [str(t) for t in tags_raw]
    category = str(value.get("category") or "general")
    importance = _normalise_importance(value.get("importance"))
    return Insight(
        title=title.strip(),
        content=content.strip(),
        category=category,
        tags=tags,
        importance=importance,
    )


def record_insight(
    state: PipelineState,
    *,
    title: str,
    content: str,
    importance: Union[Importance, str] = Importance.MEDIUM,
    category: str = "general",
    tags: Optional[Sequence[str]] = None,
) -> Insight:
    """Queue a single insight for the next Stage 15 run.

    Returns the :class:`Insight` instance that was appended. Callers
    that already have an ``Insight`` can skip this helper and append to
    ``state.metadata[PENDING_INSIGHTS_KEY]`` directly.
    """
    insight = coerce_insight(
        {
            "title": title,
            "content": content,
            "importance": importance,
            "category": category,
            "tags": list(tags or []),
        }
    )
    queue: List[Any] = state.metadata.setdefault(PENDING_INSIGHTS_KEY, [])
    queue.append(insight)
    return insight


def drain_pending_insights(state: PipelineState) -> Iterable[Insight]:
    """Pop and normalise every queued insight, then clear the queue.

    Order is preserved. Items that fail :func:`coerce_insight` validation
    raise immediately; the strategy is expected to wrap this call in its
    own error-event emission so a single bad entry does not leave a
    half-drained queue behind.
    """
    queue = state.metadata.get(PENDING_INSIGHTS_KEY)
    if not queue:
        return []
    insights: List[Insight] = []
    try:
        for raw in list(queue):
            insights.append(coerce_insight(raw))
    finally:
        # Always clear, even if coercion partially failed: the queue
        # owner (Stage 15 strategy) emits the error event and we don't
        # want to retry a bad payload on every subsequent run.
        state.metadata[PENDING_INSIGHTS_KEY] = []
    return insights


def list_recorded_insights(state: PipelineState) -> List[Insight]:
    """Return the running collection of insights persisted on the state."""
    return list(state.metadata.get(INSIGHTS_KEY, []))


def insights_to_dicts(insights: Iterable[Insight]) -> List[Dict[str, Any]]:
    """Project insights into JSON-ready dicts for serialization."""
    out: List[Dict[str, Any]] = []
    for ins in insights:
        out.append(
            {
                "title": ins.title,
                "content": ins.content,
                "category": ins.category,
                "tags": list(ins.tags),
                "importance": ins.importance.value,
            }
        )
    return out


__all__ = [
    "PENDING_INSIGHTS_KEY",
    "INSIGHTS_KEY",
    "coerce_insight",
    "record_insight",
    "drain_pending_insights",
    "list_recorded_insights",
    "insights_to_dicts",
    "Insight",
    "Importance",
]
