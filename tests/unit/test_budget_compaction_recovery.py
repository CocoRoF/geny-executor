"""2.5.0 — token-budget guard → compaction → recheck recovery.

Before 2.5.0 the token-budget guard read session/turn-cumulative
``token_usage`` and hard-rejected on breach — a measure compaction could
never lower, so a long tool-loop turn died with no recovery and the
Stage 2 compactor was decoupled from the guard entirely.

These tests pin the new contract:

  * The guard measures the *projected* next-call context (system +
    messages + tools) via the shared estimator both stages use.
  * On pressure the guard returns ``action="compact"`` (recoverable),
    not ``action="reject"``.
  * GuardStage, given a compactor, compacts ``state.messages`` and
    re-checks once — passing if it now fits, hard-rejecting only if it
    still does not.
  * With no compactor wired the signal degrades to the pre-2.5.0 hard
    reject.
  * The shared ``run_compaction`` helper emits a uniform event and
    records to a provider unless the compactor self-persists.
  * The pipeline auto-wires the Context compactor into the Guard stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.core.compaction import run_compaction
from geny_executor.core.errors import GuardRejectError
from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import (
    estimate_message_tokens,
    estimate_prompt_tokens,
)
from geny_executor.stages.s02_context.interface import HistoryCompactor
from geny_executor.stages.s04_guard.artifact.default.guards import TokenBudgetGuard
from geny_executor.stages.s04_guard.artifact.default.stage import GuardStage


# ── helpers ─────────────────────────────────────────────────────────


class _KeepLastCompactor(HistoryCompactor):
    """Trivial compactor: keep the last N messages. Records call count."""

    def __init__(self, keep_last: int = 2, *, persists_own: bool = False):
        self._keep_last = keep_last
        self.persists_own_compaction = persists_own
        self.calls = 0

    @property
    def name(self) -> str:
        return "keep_last"

    async def compact(self, state: PipelineState) -> None:
        self.calls += 1
        if len(state.messages) > self._keep_last:
            state.messages = state.messages[-self._keep_last :]


class _RecordingProvider:
    def __init__(self):
        self.records = []

    async def record_compaction(self, summary, **kwargs):
        self.records.append({"summary": summary, **kwargs})
        return "compactions/note.md"


def _msgs(n: int, chars: int = 2000):
    return [{"role": "user", "content": "a" * chars} for _ in range(n)]


def _events(state: PipelineState, etype: str):
    return [e for e in state.events if e.get("type") == etype]


# ── estimator ───────────────────────────────────────────────────────


class TestEstimator:
    def test_sums_system_messages_tools(self):
        s = PipelineState()
        s.system = "x" * 4000  # ~1000 tok
        s.messages = [{"role": "user", "content": "y" * 8000}]  # ~2000 tok
        s.tools = [{"name": "t", "description": "z" * 4000}]  # ~1000+ tok
        est = estimate_prompt_tokens(s)
        assert est >= 4000  # 1000 + 2000 + ~1000+

    def test_image_block_is_flat_not_base64_length(self):
        huge_b64 = "A" * 200_000  # would be ~50k tokens if counted literally
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "source": {"data": huge_b64}},
                ],
            }
        ]
        est = estimate_message_tokens(msgs)
        # Flat image estimate (~1.6k) dominates, not the 50k base64 length.
        assert est < 5_000


# ── guard signal ────────────────────────────────────────────────────


class TestGuardSignal:
    def test_compact_action_on_projected_pressure(self):
        s = PipelineState()
        s.context_window_budget = 1000
        s.messages = [{"role": "user", "content": "a" * 4000}]  # ~1000 projected
        r = TokenBudgetGuard().check(s)
        assert not r.passed
        assert r.action == "compact"  # recoverable, not "reject"

    def test_passes_with_headroom(self):
        s = PipelineState()
        s.context_window_budget = 200_000
        s.messages = [{"role": "user", "content": "small"}]
        assert TokenBudgetGuard().check(s).passed

    def test_cumulative_usage_no_longer_trips_guard(self):
        # The old measure: huge cumulative token_usage, but a tiny actual
        # request. New guard looks at the request, so it passes.
        s = PipelineState()
        s.context_window_budget = 200_000
        s.token_usage.input_tokens = 500_000
        s.token_usage.output_tokens = 500_000
        s.messages = [{"role": "user", "content": "tiny"}]
        assert TokenBudgetGuard().check(s).passed


# ── guard stage recovery ────────────────────────────────────────────


class TestGuardStageRecovery:
    @pytest.mark.asyncio
    async def test_compact_then_recheck_passes(self):
        s = PipelineState()
        s.context_window_budget = 5000
        s.messages = _msgs(40)  # ~20000 projected, over
        comp = _KeepLastCompactor(keep_last=2)
        stage = GuardStage(guards=[TokenBudgetGuard(min_remaining_tokens=500)])
        stage.attach_budget_recovery(comp)

        out = await stage.execute("IN", s)

        assert out == "IN"
        assert comp.calls == 1
        assert len(s.messages) == 2
        # both the original check and the recheck were emitted
        checks = _events(s, "guard.check")
        assert len(checks) == 2
        assert checks[-1]["data"].get("recheck") is True
        assert _events(s, "guard.compacting")
        assert _events(s, "context.compacted")

    @pytest.mark.asyncio
    async def test_still_over_after_compaction_hard_rejects(self):
        s = PipelineState()
        s.context_window_budget = 5000
        s.system = "S" * 40_000  # ~10000 tok system — compaction can't help
        s.messages = _msgs(40)
        comp = _KeepLastCompactor(keep_last=2)
        stage = GuardStage(guards=[TokenBudgetGuard(min_remaining_tokens=500)])
        stage.attach_budget_recovery(comp)

        with pytest.raises(GuardRejectError) as exc:
            await stage.execute("IN", s)
        assert exc.value.guard_name == "token_budget"
        assert comp.calls == 1  # compacted once, then gave up

    @pytest.mark.asyncio
    async def test_no_compactor_degrades_to_reject(self):
        s = PipelineState()
        s.context_window_budget = 5000
        s.messages = _msgs(40)
        stage = GuardStage(guards=[TokenBudgetGuard(min_remaining_tokens=500)])
        # no attach_budget_recovery

        with pytest.raises(GuardRejectError):
            await stage.execute("IN", s)

    @pytest.mark.asyncio
    async def test_recovery_in_non_fail_fast_mode(self):
        s = PipelineState()
        s.context_window_budget = 5000
        s.messages = _msgs(40)
        comp = _KeepLastCompactor(keep_last=2)
        stage = GuardStage(
            guards=[TokenBudgetGuard(min_remaining_tokens=500)], fail_fast=False
        )
        stage.attach_budget_recovery(comp)

        out = await stage.execute("IN", s)
        assert out == "IN"
        assert comp.calls == 1


# ── run_compaction helper ───────────────────────────────────────────


class TestRunCompaction:
    @pytest.mark.asyncio
    async def test_emits_event_and_records(self):
        s = PipelineState(session_id="sess1")
        s.messages = _msgs(10)
        comp = _KeepLastCompactor(keep_last=2)
        prov = _RecordingProvider()

        out = await run_compaction(s, comp, trigger="guard", provider=prov)

        assert out["ok"] is True
        assert out["replaced"] == 8
        evt = _events(s, "context.compacted")[0]["data"]
        assert evt["trigger"] == "guard"
        assert evt["messages_before"] == 10
        assert evt["messages_after"] == 2
        assert len(prov.records) == 1
        assert prov.records[0]["replaced_count"] == 8
        assert prov.records[0]["session_id"] == "sess1"

    @pytest.mark.asyncio
    async def test_self_persisting_compactor_skips_provider(self):
        s = PipelineState()
        s.messages = _msgs(10)
        comp = _KeepLastCompactor(keep_last=2, persists_own=True)
        prov = _RecordingProvider()

        await run_compaction(s, comp, trigger="proactive", provider=prov)

        assert prov.records == []  # not double-recorded

    @pytest.mark.asyncio
    async def test_compactor_failure_is_swallowed(self):
        class _Boom(HistoryCompactor):
            @property
            def name(self):
                return "boom"

            async def compact(self, state):
                raise RuntimeError("kaboom")

        s = PipelineState()
        s.messages = _msgs(5)
        out = await run_compaction(s, _Boom(), trigger="guard")
        assert out["ok"] is False
        assert _events(s, "context.compaction_failed")


# ── file provider record_compaction ─────────────────────────────────


class TestFileProviderRecord:
    @pytest.mark.asyncio
    async def test_writes_compaction_note(self, tmp_path: Path):
        from geny_executor.memory.providers.file.provider import FileMemoryProvider

        prov = FileMemoryProvider(root=tmp_path, session_id="s1")
        await prov.initialize()

        fname = await prov.record_compaction(
            "A recap of the earlier conversation.",
            replaced_count=12,
            strategy="llm_summary",
            saved_tokens=3400,
            trigger="guard",
        )
        assert fname
        # The note landed in the compactions category folder.
        comp_dir = tmp_path / "memory" / "compactions"
        files = list(comp_dir.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert "recap of the earlier conversation" in text
        assert "replaced_count: 12" in text

    @pytest.mark.asyncio
    async def test_noop_when_nothing_to_record(self, tmp_path: Path):
        from geny_executor.memory.providers.file.provider import FileMemoryProvider

        prov = FileMemoryProvider(root=tmp_path, session_id="s1")
        await prov.initialize()
        assert await prov.record_compaction("", replaced_count=0) is None


# ── pipeline auto-wire ──────────────────────────────────────────────


class TestPipelineAutowire:
    def test_init_state_wires_guard_from_context(self):
        from geny_executor.core.pipeline import Pipeline
        from geny_executor.core.config import PipelineConfig
        from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
        from geny_executor.stages.s02_context.artifact.default.compactors import (
            SummaryCompactor,
        )

        pipe = Pipeline(PipelineConfig(name="t"))
        ctx = ContextStage(compactor=SummaryCompactor())
        guard = GuardStage(guards=[TokenBudgetGuard()])
        pipe.register_stage(ctx)
        pipe.register_stage(guard)

        pipe._init_state(None)

        assert guard._budget_compactor is ctx._compactor

    def test_explicit_recovery_not_clobbered(self):
        from geny_executor.core.pipeline import Pipeline
        from geny_executor.core.config import PipelineConfig
        from geny_executor.stages.s02_context.artifact.default.stage import ContextStage

        pipe = Pipeline(PipelineConfig(name="t"))
        ctx = ContextStage()
        guard = GuardStage(guards=[TokenBudgetGuard()])
        explicit = _KeepLastCompactor()
        guard.attach_budget_recovery(explicit)
        pipe.register_stage(ctx)
        pipe.register_stage(guard)

        pipe._init_state(None)

        assert guard._budget_compactor is explicit  # host wiring preserved


# ── slot registry ───────────────────────────────────────────────────


def test_llm_summary_registered_in_s02_slot():
    from geny_executor.stages.s02_context.artifact.default.stage import ContextStage

    stage = ContextStage()
    registry = stage._slots["compactor"].registry
    assert "llm_summary" in registry
