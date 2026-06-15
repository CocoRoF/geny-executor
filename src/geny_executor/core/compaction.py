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
from typing import Any, Optional

from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens

logger = logging.getLogger(__name__)


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
    before_msgs = len(state.messages or [])
    before_tokens = estimate_prompt_tokens(state)

    try:
        await compactor.compact(state)
    except Exception as exc:  # noqa: BLE001 — best effort
        state.add_event(
            "context.compaction_failed",
            {"compactor": _compactor_name(compactor), "trigger": trigger, "error": str(exc)},
        )
        logger.warning("Compaction (%s) failed: %s", trigger, exc)
        return {"ok": False, "before_messages": before_msgs, "after_messages": before_msgs}

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
