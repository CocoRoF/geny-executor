"""Deterministic prune pass — the no-LLM relief that runs before summary
compaction (dedup repeated tool outputs, strip stale base64 images, trim
oversized stale results), plus the Stage-3/Stage-5 cache-split fixes.

Effect-proving doctrine: each behavior asserts a MEASURED improvement
(estimated tokens / chars saved), not just that code ran.
"""

from __future__ import annotations

import pytest

from geny_executor.core.compaction import run_compaction
from geny_executor.core.context_prune import prune_messages
from geny_executor.core.message_repair import repair_dangling_tool_calls
from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_prompt_tokens
from geny_executor.stages.s02_context.interface import HistoryCompactor

BIG = "리듬게임 판정 데이터 " * 60  # ~1.3k chars, over the dup threshold


def _tool_turn(tool_id: str, result_content) -> list:
    """An assistant tool_use + its user tool_result pair."""
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": "read", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id,
             "content": result_content}]},
    ]


def _b64_image_block():
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": "A" * 5000}}


# ── dedup ─────────────────────────────────────────────────────────────


def test_dedup_keeps_newest_copy_and_protected_tail():
    msgs = []
    for i in range(4):                     # 4 identical big reads
        msgs += _tool_turn(f"t{i}", BIG)
    msgs += [{"role": "user", "content": "마지막 질문"}]

    metrics = prune_messages(msgs, protect_last=3)

    results = [b for m in msgs if isinstance(m.get("content"), list)
               for b in m["content"] if b.get("type") == "tool_result"]
    full = [b for b in results if b["content"] == BIG]
    ghost = [b for b in results if b["content"] != BIG]
    # newest copy (t3, inside protect window) keeps full content;
    # every older duplicate is rewritten to the back-reference
    assert len(full) == 1 and len(ghost) == 3
    assert all("duplicate tool output" in g["content"] for g in ghost)
    assert metrics["deduped"] == 3
    assert metrics["chars_saved"] > 3 * (len(BIG) - 100)


def test_small_results_never_deduped():
    msgs = _tool_turn("a", "ok") + _tool_turn("b", "ok") + \
        [{"role": "user", "content": "x"}] * 3
    metrics = prune_messages(msgs, protect_last=1)
    assert metrics["deduped"] == 0  # tiny repeats are legitimate & free


# ── image stripping ───────────────────────────────────────────────────


def test_stale_base64_images_stripped_recursively_tail_kept():
    old = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "s1",
         "content": [_b64_image_block(),
                     {"type": "text", "text": "screenshot taken"}]}]}]
    fresh = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "s2",
         "content": [_b64_image_block()]}]}]
    msgs = ([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "s1", "name": "shot", "input": {}}]}]
        + old
        + [{"role": "user", "content": "패딩"}] * 4
        + [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "s2", "name": "shot", "input": {}}]}]
        + fresh)

    s = PipelineState()
    s.messages = msgs
    before = estimate_prompt_tokens(s)
    metrics = prune_messages(msgs, protect_last=3)
    s.shared.pop("_prompt_tokens_memo", None)
    after = estimate_prompt_tokens(s)

    assert metrics["images_stripped"] == 1
    old_blocks = old[0]["content"][0]["content"]
    assert old_blocks[0]["type"] == "text" and "image removed" in old_blocks[0]["text"]
    assert old_blocks[1]["text"] == "screenshot taken"          # siblings kept
    assert fresh[0]["content"][0]["content"][0]["type"] == "image"  # tail kept
    # measured effect: the flat image estimate (~1600 tok) is reclaimed
    assert before - after > 1000


# ── oversize trimming ─────────────────────────────────────────────────


def test_oversized_stale_output_trimmed_with_marker():
    huge = "x" * 20_000
    msgs = _tool_turn("h1", huge) + [{"role": "user", "content": "패딩"}] * 4
    metrics = prune_messages(msgs, protect_last=2,
                             trim_over_chars=4000, trim_keep_chars=600)
    trimmed = msgs[1]["content"][0]["content"]
    assert trimmed.startswith("x" * 600)
    assert "chars trimmed during context compaction" in trimmed
    assert metrics["trimmed"] == 1 and metrics["chars_saved"] > 19_000
    # recent copy would have been protected
    msgs2 = _tool_turn("h2", huge)
    assert prune_messages(msgs2, protect_last=6)["trimmed"] == 0


# ── invariants ────────────────────────────────────────────────────────


def test_prune_preserves_count_order_and_tool_pairs():
    msgs = []
    for i in range(5):
        msgs += _tool_turn(f"p{i}", BIG if i % 2 == 0 else "짧은 결과")
    msgs[2]["content"].append(_b64_image_block())
    snapshot_roles = [m["role"] for m in msgs]
    ids_before = [b["id"] for m in msgs if isinstance(m.get("content"), list)
                  for b in m["content"] if b.get("type") == "tool_use"]

    prune_messages(msgs, protect_last=2)

    assert [m["role"] for m in msgs] == snapshot_roles          # count+order
    ids_after = [b["id"] for m in msgs if isinstance(m.get("content"), list)
                 for b in m["content"] if b.get("type") == "tool_use"]
    assert ids_after == ids_before                              # tool_use intact
    assert repair_dangling_tool_calls(msgs) == 0                # no orphans


# ── run_compaction integration ────────────────────────────────────────


class _NoopCompactor(HistoryCompactor):
    @property
    def name(self) -> str:
        return "noop"

    async def compact(self, state) -> None:  # noqa: D401 — test stub
        return None


@pytest.mark.asyncio
async def test_run_compaction_prunes_before_compactor_and_measures():
    """EFFECT PROOF at the integration seam: even with a compactor that does
    nothing, run_compaction's deterministic pre-pass alone reduces the
    estimate, rolls into saved_tokens_estimate, and emits context.pruned."""
    s = PipelineState()
    msgs = []
    for i in range(4):
        msgs += _tool_turn(f"r{i}", BIG)
    msgs += [{"role": "user", "content": "마지막"}] * 2
    s.messages = msgs

    before = estimate_prompt_tokens(s)
    result = await run_compaction(s, _NoopCompactor(), trigger="guard")
    assert result["ok"] is True

    events = {e[0] if isinstance(e, tuple) else e.get("type"): e
              for e in getattr(s, "events", [])} if hasattr(s, "events") else {}
    event_types = [getattr(e, "type", None) or (e[0] if isinstance(e, tuple) else e.get("type"))
                   for e in (s.events or [])]
    assert "context.pruned" in event_types
    compacted = [e for e in s.events
                 if (getattr(e, "type", None) or (e[0] if isinstance(e, tuple) else e.get("type")))
                 == "context.compacted"]
    assert compacted, "context.compacted must still be emitted"
    s.shared.pop("_prompt_tokens_memo", None)
    after = estimate_prompt_tokens(s)
    # measured effect: 3 of 4 duplicate reads reclaimed → >30% of the estimate
    assert after < before * 0.7, \
        f"pre-pass alone must reclaim real tokens (before={before}, after={after})"


# ── B1: cache-split edge fixes (catalog × volatile_placement="system") ─


class _FakeDeferredRegistry:
    """Registry with one deferred (non-core) tool → non-empty catalog."""
    version = 1

    def list_names(self):
        return ["hidden_tool"]

    def is_core(self, name):
        return False

    def get(self, name):
        class _T:
            description = "does hidden things"
            def to_schema(self):
                return {"name": "hidden_tool"}
        return _T()

    def to_api_format(self, exposed_only=True):
        return []


@pytest.mark.asyncio
async def test_system_mode_catalog_lands_in_stable_region():
    """volatile_placement='system' + deferred catalog: the (cache-stable)
    catalog must join the STABLE region so Stage 5 can still split the
    volatile tail out of the cached prefix. Before the fix the catalog was
    appended AFTER the joined string, breaking the split → the volatile
    clock/memory tail was cached → full system re-prefill every turn."""
    from geny_executor.stages.s03_system.artifact.default.builders import (
        ComposablePromptBuilder, DateTimeBlock, PersonaBlock,
    )
    from geny_executor.stages.s03_system.artifact.default.stage import SystemStage
    from geny_executor.stages.s05_cache.artifact.default.strategies import (
        AggressiveCacheStrategy,
    )

    stage = SystemStage(
        builder=ComposablePromptBuilder(
            blocks=[PersonaBlock("Persona."), DateTimeBlock()]),
        volatile_placement="system",
        tool_registry=_FakeDeferredRegistry(),
    )
    state = PipelineState()
    state.metadata["model_provider"] = "anthropic"
    await stage.execute("in", state)

    parts = state.shared["system_parts"]
    assert "Additional tools" in parts["stable_text"], \
        "catalog must be recorded as part of the STABLE region"
    assert "Current date" in parts["volatile_text"]
    # joined string still equals stable + sep + volatile → split holds
    assert state.system == f"{parts['stable_text']}\n\n{parts['volatile_text']}"

    AggressiveCacheStrategy().apply_cache_markers(state)
    assert isinstance(state.system, list) and len(state.system) == 2
    cached, tail = state.system
    assert cached.get("cache_control") == {"type": "ephemeral"}
    assert "Additional tools" in cached["text"]      # catalog IS cached
    assert "Current date" in tail["text"]            # volatile is NOT
    assert "cache_control" not in tail


def test_cache_system_tolerant_split_survives_appended_suffix():
    """Defense-in-depth: even when extra text lands between/after the parts
    (breaking exact equality), the split still finds the volatile tail by
    position, keeps byte-identity, and never caches the volatile bytes."""
    from geny_executor.stages.s05_cache.artifact.default.strategies import (
        AggressiveCacheStrategy,
    )
    stable, volatile = "Persona.", "Current date: 2026-07-23 12:00"
    system = f"{stable}\n\nEXTRA-STABLE-SUFFIX\n\n{volatile}"
    state = PipelineState()
    state.metadata["model_provider"] = "anthropic"
    state.messages = [{"role": "user", "content": "hi"}]
    state.system = system
    state.shared["system_parts"] = {"stable_text": stable,
                                    "volatile_text": volatile}

    AggressiveCacheStrategy().apply_cache_markers(state)

    assert isinstance(state.system, list) and len(state.system) == 2
    cached, tail = state.system
    assert cached["cache_control"] == {"type": "ephemeral"}
    assert volatile not in cached["text"]            # volatile never cached
    assert "EXTRA-STABLE-SUFFIX" in cached["text"]   # appended text cached
    assert cached["text"] + tail["text"] == system   # byte-identical
