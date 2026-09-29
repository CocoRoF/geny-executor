"""Shared compaction runner — one place that runs a compactor, emits a
uniform event, and persists the snapshot to a memory provider.

Both the Stage 2 proactive trigger (context near 80%) and the Stage 4
reactive token-budget guard (context near the hard ceiling) compact the
SAME ``state.messages`` with the SAME compactor instance. Centralising
"compact + record" here guarantees they log and persist identically, and
that the snapshot is never written twice (a host wrapper that records its
own snapshot sets ``compactor.persists_own_compaction = True``).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens

logger = logging.getLogger(__name__)

#: Stage-18's STM-recording watermark (index into ``state.messages``).
#: Duplicated here (not imported from s18) so the core has no stage dep;
#: the string is the contract.
_STATE_LAST_RECORDED = "memory.last_recorded_idx"


#: Messages a compaction removed before Stage 18 had recorded them. The
#: recorders take these first, then ``state.messages[watermark:]``.
UNRECORDED_KEY = "memory.unrecorded_before_compaction"


def reconcile_recorded_index(before: List[Any], after: List[Any], metadata: dict) -> None:
    """Translate Stage-18's STM watermark across a compaction (audit D3).

    Stage 18 records ``state.messages[last_idx:]`` as STM turns and sets
    ``last_idx = len(messages)``. Compaction shrinks ``state.messages``,
    so a watermark of 60 against a now-15-long list makes
    ``messages[60:]`` empty forever — every subsequent turn silently
    stops being recorded until the list regrows past 60.

    Compactors keep a SUFFIX of the real messages (the same dict objects,
    by identity) and prepend synthetic summary messages. We find that
    kept suffix by identity and remap the watermark so already-recorded
    messages stay recorded and the genuinely-new tail still gets picked
    up next turn. Pure index arithmetic on object identity — no message
    is mutated.

    Messages removed before they were recorded — on a host that records at
    the end of the turn, the turn's own request and its first tool calls,
    whenever the loop ran long enough to compact — go to
    :data:`UNRECORDED_KEY`. They used to be skipped: the watermark jumped
    over them and they were never written anywhere. The same happened to
    a turn with no watermark yet, whose recorder then wrote the summary
    and its acknowledgement into memory as if they had been said.
    """
    raw = metadata.get(_STATE_LAST_RECORDED)
    old_idx = raw if isinstance(raw, int) and raw > 0 else 0

    # Longest suffix of ``after`` whose objects are the trailing objects
    # of ``before`` (by identity) is the kept region.
    before_ids = [id(m) for m in before]
    after_ids = [id(m) for m in after]
    kept = 0
    bi, ai = len(before_ids) - 1, len(after_ids) - 1
    while bi >= 0 and ai >= 0 and before_ids[bi] == after_ids[ai]:
        kept += 1
        bi -= 1
        ai -= 1
    start = len(before) - kept  # first before-index that survived
    n_synthetic = len(after) - kept  # summary messages prepended

    if start == 0 and n_synthetic == 0:
        return  # nothing was removed or added

    # The replay's stable-prefix length (a Stage 5 cache hint) counted
    # messages that are gone now.
    metadata.pop("cache.stable_prefix_messages", None)

    if old_idx < start:
        pending = metadata.get(UNRECORDED_KEY)
        stash = list(pending) if isinstance(pending, list) else []
        stash.extend(before[old_idx:start])
        metadata[UNRECORDED_KEY] = stash

    if old_idx <= start:
        # Recorded boundary sits in the removed region: everything kept is
        # unrecorded, the summary messages in front of it are not to be.
        new_idx = n_synthetic
    else:
        # Boundary lands inside the kept suffix: shift by the prefix delta.
        new_idx = n_synthetic + (old_idx - start)

    metadata[_STATE_LAST_RECORDED] = max(0, min(new_idx, len(after)))


def unrecorded_messages(state: Any) -> List[Any]:
    """Every message STM does not have yet, in order.

    The ones a compaction removed before they were recorded come first
    (:data:`UNRECORDED_KEY`), then
    ``state.messages`` past the watermark.
    """
    pending = state.metadata.get(UNRECORDED_KEY)
    removed = [m for m in pending if isinstance(m, dict)] if isinstance(pending, list) else []
    last = int(state.metadata.get(_STATE_LAST_RECORDED, 0) or 0)
    return removed + list(state.messages[last:])


def mark_recorded(state: Any) -> None:
    """Everything :func:`unrecorded_messages` returned is now in STM."""
    state.metadata[_STATE_LAST_RECORDED] = len(state.messages)
    state.metadata.pop(UNRECORDED_KEY, None)


def _compactor_name(compactor: Any) -> str:
    return str(getattr(compactor, "name", None) or type(compactor).__name__)


def _summary_text(state: PipelineState) -> str:
    """Best-effort extraction of the summary a compactor placed at the head."""
    msgs = state.messages or []
    if not msgs:
        return ""
    head = msgs[0]
    if not isinstance(head, dict):
        return ""
    content = head.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


async def run_compaction(
    state: PipelineState,
    compactor: Any,
    *,
    trigger: str,
    provider: Optional[Any] = None,
) -> dict:
    """Run ``compactor`` against ``state`` and return a result summary dict.

    Emits a ``context.compacted`` event carrying ``trigger`` ("proactive"
    from Stage 2, "guard" from Stage 4) and, when a provider exposes
    ``record_compaction`` and the compactor does not persist its own
    snapshot, records the snapshot to the provider's "compactions"
    category. Never raises — compaction is best-effort relief, not a
    correctness gate; failures are logged as events and swallowed.
    """
    before_list = list(state.messages or [])
    before_msgs = len(before_list)
    before_tokens = estimate_prompt_tokens(state)

    # Deterministic pre-pass BEFORE the (expensive, lossy) compactor: dedup
    # repeated tool outputs, strip stale base64 images, trim oversized stale
    # results. In-place, count/order-preserving, pair-safe — so the watermark
    # reconcile below and the compactor itself see a normal message list. The
    # token-estimate memo keys on list length + head/tail ids, which this
    # pass preserves, so it must be dropped for after_tokens to be honest.
    try:
        from geny_executor.core.context_prune import prune_messages

        prune_metrics = prune_messages(state.messages or [])
        if any(prune_metrics.get(k) for k in ("deduped", "images_stripped", "trimmed")):
            state.shared.pop("_prompt_tokens_memo", None)
            state.add_event("context.pruned", dict(prune_metrics))
    except Exception:  # noqa: BLE001 — relief, never a gate
        logger.debug("deterministic prune pass failed", exc_info=True)

    try:
        await compactor.compact(state)
    except Exception as exc:  # noqa: BLE001 — best effort
        state.add_event(
            "context.compaction_failed",
            {"compactor": _compactor_name(compactor), "trigger": trigger, "error": str(exc)},
        )
        logger.warning("Compaction (%s) failed: %s", trigger, exc)
        return {"ok": False, "before_messages": before_msgs, "after_messages": before_msgs}

    # Keep Stage-18's STM watermark valid across the shrink (audit D3).
    reconcile_recorded_index(before_list, list(state.messages or []), state.metadata)

    after_msgs = len(state.messages or [])
    after_tokens = estimate_prompt_tokens(state)
    replaced = max(0, before_msgs - after_msgs)
    saved_tokens = max(0, before_tokens - after_tokens)

    state.add_event(
        "context.compacted",
        {
            "strategy": _compactor_name(compactor),
            "trigger": trigger,
            "messages_before": before_msgs,
            "messages_after": after_msgs,
            "saved_tokens_estimate": saved_tokens,
        },
    )

    # Persist the snapshot unless the compactor already does it itself.
    if (
        replaced > 0
        and provider is not None
        and not getattr(compactor, "persists_own_compaction", False)
        and hasattr(provider, "record_compaction")
    ):
        try:
            await provider.record_compaction(
                _summary_text(state),
                replaced_count=replaced,
                strategy=_compactor_name(compactor),
                saved_tokens=saved_tokens,
                session_id=getattr(state, "session_id", "") or "",
                trigger=trigger,
            )
        except Exception as exc:  # noqa: BLE001 — best effort
            state.add_event(
                "context.compaction_record_failed",
                {"compactor": _compactor_name(compactor), "error": str(exc)},
            )
            logger.debug("record_compaction failed: %s", exc)

    return {
        "ok": True,
        "before_messages": before_msgs,
        "after_messages": after_msgs,
        "replaced": replaced,
        "saved_tokens_estimate": saved_tokens,
    }
