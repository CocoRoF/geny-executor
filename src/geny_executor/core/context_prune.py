"""Deterministic tool-output pruning — the no-LLM pass that runs BEFORE
summary compaction.

LLM summarization is expensive and lossy; a large share of context pressure
is mechanical redundancy that a deterministic pass can reclaim first:

* **Duplicate tool outputs** — the same file read five times keeps five full
  copies in history. Only the NEWEST copy carries information; older ones are
  rewritten to a one-line back-reference (hash-matched, exact).
* **Stale image payloads** — a base64 screenshot in an old tool result rides
  every subsequent request forever (and survives summary compaction whenever
  the compactor keeps that message verbatim).
* **Oversized stale outputs** — a 10k-char terminal dump from twenty turns ago
  is almost never load-bearing; the head plus an explicit trim marker is.

Invariants (why this is safe to run on any history):

* Message COUNT and ORDER never change — STM watermarks and compaction
  bookkeeping stay valid.
* ``tool_use`` blocks and every ``tool_use_id`` are untouched — pruning can
  never orphan a tool call (pair repair stays a no-op).
* Blocks are rewritten in place, never removed; only ``tool_result`` content
  and image blocks shrink.
* The newest ``protect_last`` messages are never touched — recent context is
  exactly what the model is working from.

Everything here is pure function + stdlib; no model, no network.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

__all__ = ["prune_messages", "PruneMetrics"]

#: Messages within this tail window are never modified.
DEFAULT_PROTECT_LAST = 6
#: Tool-result text shorter than this is never considered for dedup — tiny
#: results ("ok", exit codes) legitimately repeat and cost nothing.
DEFAULT_MIN_DUP_CHARS = 512
#: Stale tool-result text longer than this is trimmed to its head.
DEFAULT_TRIM_OVER_CHARS = 4000
#: How much head text a trimmed output keeps.
DEFAULT_TRIM_KEEP_CHARS = 600

_DUP_NOTE = "[duplicate tool output — identical to a more recent result]"
_IMAGE_NOTE = "[image removed during context compaction]"


class PruneMetrics(Dict[str, int]):
    """Plain dict of counters; a class only so callers can type it."""


def _iter_tool_results(message: Dict[str, Any]):
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            yield block


def _result_text_parts(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The text sub-blocks of a tool_result whose content is a block list."""
    content = block.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content
            if isinstance(b, dict) and b.get("type") == "text"]


def _content_hash(block: Dict[str, Any]) -> str:
    """Stable digest of a tool_result's content (str or block list)."""
    content = block.get("content")
    try:
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True,
                             default=str)
    except Exception:  # noqa: BLE001 — unhashable exotic content: skip dedup
        return ""
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()


def _content_size(block: Dict[str, Any]) -> int:
    content = block.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content
                   if isinstance(b, dict) and b.get("type") == "text")
    return 0


def _strip_images_in(blocks: List[Any]) -> int:
    """Replace base64 image blocks with a small text marker, recursing into
    tool_result content lists. Returns the number of images removed."""
    removed = 0
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        if (b.get("type") == "image"
                and isinstance(b.get("source"), dict)
                and b["source"].get("type") == "base64"):
            blocks[i] = {"type": "text", "text": _IMAGE_NOTE}
            removed += 1
        elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            removed += _strip_images_in(b["content"])
    return removed


def _trim_result(block: Dict[str, Any], keep: int) -> int:
    """Trim an oversized tool_result to its head + explicit marker. Returns
    chars removed."""
    content = block.get("content")
    if isinstance(content, str):
        dropped = len(content) - keep
        block["content"] = (content[:keep]
                            + f"\n… [{dropped} chars trimmed during context compaction]")
        return dropped
    if isinstance(content, list):
        dropped = 0
        budget = keep
        for sub in _result_text_parts(block):
            text = sub.get("text", "")
            if budget <= 0:
                dropped += len(text)
                sub["text"] = ""
            elif len(text) > budget:
                dropped += len(text) - budget
                sub["text"] = text[:budget]
                budget = 0
            else:
                budget -= len(text)
        if dropped > 0:
            parts = _result_text_parts(block)
            if parts:
                parts[-1]["text"] += (
                    f"\n… [{dropped} chars trimmed during context compaction]")
        return dropped
    return 0


def prune_messages(
    messages: List[Dict[str, Any]],
    *,
    protect_last: int = DEFAULT_PROTECT_LAST,
    min_dup_chars: int = DEFAULT_MIN_DUP_CHARS,
    trim_over_chars: int = DEFAULT_TRIM_OVER_CHARS,
    trim_keep_chars: int = DEFAULT_TRIM_KEEP_CHARS,
) -> PruneMetrics:
    """Run the deterministic prune over *messages* in place.

    Returns metrics: ``deduped`` (results back-referenced), ``images_stripped``,
    ``trimmed`` (oversized results shortened), ``chars_saved`` (text chars
    removed, images excluded).
    """
    metrics = PruneMetrics(deduped=0, images_stripped=0, trimmed=0,
                           chars_saved=0)
    if not messages:
        return metrics
    cutoff = max(0, len(messages) - max(0, protect_last))

    # Pass 1 — hashes of every PROTECTED/newer occurrence win. Walk from the
    # end so the newest copy of any repeated output is the one kept.
    seen: set = set()
    for mi in range(len(messages) - 1, -1, -1):
        msg = messages[mi]
        if not isinstance(msg, dict):
            continue
        protected = mi >= cutoff
        for block in _iter_tool_results(msg):
            if _content_size(block) < min_dup_chars:
                continue
            digest = _content_hash(block)
            if not digest:
                continue
            if digest in seen and not protected:
                old = _content_size(block)
                block["content"] = _DUP_NOTE
                metrics["deduped"] += 1
                metrics["chars_saved"] += max(0, old - len(_DUP_NOTE))
            else:
                seen.add(digest)

    # Pass 2 — stale-region cleanup: image stripping + oversize trimming.
    for mi in range(cutoff):
        msg = messages[mi]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            metrics["images_stripped"] += _strip_images_in(content)
        for block in _iter_tool_results(msg):
            size = _content_size(block)
            if size > trim_over_chars:
                saved = _trim_result(block, trim_keep_chars)
                if saved > 0:
                    metrics["trimmed"] += 1
                    metrics["chars_saved"] += saved
    return metrics
