"""Unit tests for Pipeline.resume + PipelineResumeRequester (S9c.1)."""

from __future__ import annotations

import asyncio

import pytest

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl import (
    HITL_HISTORY_KEY,
    HITL_LAST_DECISION_KEY,
    HITL_REQUEST_KEY,
    AutoRejectTimeout,
    HITLDecision,
    HITLRequest,
    HITLStage,
    PipelineResumeRequester,
)


# ── Pipeline-side API ──────────────────────────────────────────────


class TestPipelineResumeBookkeeping:
    def test_list_pending_initially_empty(self):
        p = Pipeline()
        assert p.list_pending_hitl() == []

    @pytest.mark.asyncio
    async def test_pending_future_appears_in_list(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        try:
            assert "abc" in p.list_pending_hitl()
        finally:
            f.cancel()

    @pytest.mark.asyncio
    async def test_resolved_futures_excluded_from_list(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        f.set_result(HITLDecision.APPROVE)
        assert "abc" not in p.list_pending_hitl()

    def test_resume_unknown_token_raises(self):
        p = Pipeline()
        with pytest.raises(KeyError):
            p.resume("ghost", HITLDecision.APPROVE)

    @pytest.mark.asyncio
    async def test_resume_already_resolved_raises(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        f.set_result(HITLDecision.APPROVE)
        with pytest.raises(RuntimeError, match="already resolved"):
            p.resume("abc", HITLDecision.REJECT)

    @pytest.mark.asyncio
    async def test_resume_with_string_decision_coerced(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        p.resume("abc", "approve")
        assert (await f) == HITLDecision.APPROVE

    @pytest.mark.asyncio
    async def test_resume_with_enum_decision(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        p.resume("abc", HITLDecision.REJECT)
        assert (await f) == HITLDecision.REJECT

    def test_resume_unknown_decision_string_raises(self):
        p = Pipeline()
        loop = asyncio.new_event_loop()
        try:
            f = loop.create_future()
            p._pending_hitl["abc"] = f
            with pytest.raises(ValueError, match="unknown HITL decision"):
                p.resume("abc", "bogus")
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_cancel_pending_returns_true_for_unresolved(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        assert p.cancel_pending_hitl("abc") is True
        assert (await f) == HITLDecision.CANCEL

    def test_cancel_pending_returns_false_for_unknown(self):
        p = Pipeline()
        assert p.cancel_pending_hitl("ghost") is False

    @pytest.mark.asyncio
    async def test_cancel_pending_returns_false_for_resolved(self):
        p = Pipeline()
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        p._pending_hitl["abc"] = f
        f.set_result(HITLDecision.APPROVE)
        assert p.cancel_pending_hitl("abc") is False


# ── PipelineResumeRequester ───────────────────────────────────────


class TestPipelineResumeRequester:
    @pytest.mark.asyncio
    async def test_registers_future_then_returns_decision(self):
        p = Pipeline()
        requester = PipelineResumeRequester(p)
        request = HITLRequest(token="tok-1", reason="r")

        async def driver():
            # Wait until requester registers the token.
            for _ in range(100):
                if "tok-1" in p._pending_hitl:
                    break
                await asyncio.sleep(0.001)
            p.resume("tok-1", HITLDecision.APPROVE)

        result, _ = await asyncio.gather(
            requester.request(request, PipelineState()),
            driver(),
        )
        assert result == HITLDecision.APPROVE
        # Future entry cleaned up.
        assert "tok-1" not in p._pending_hitl

    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_registration(self):
        p = Pipeline()
        requester = PipelineResumeRequester(p)
        request = HITLRequest(token="tok-2", reason="r")

        task = asyncio.create_task(requester.request(request, PipelineState()))
        # Give the task a chance to register.
        for _ in range(50):
            if "tok-2" in p._pending_hitl:
                break
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "tok-2" not in p._pending_hitl

    @pytest.mark.asyncio
    async def test_external_cancel_via_pipeline(self):
        p = Pipeline()
        requester = PipelineResumeRequester(p)
        request = HITLRequest(token="tok-3", reason="r")

        async def driver():
            for _ in range(100):
                if "tok-3" in p._pending_hitl:
                    break
                await asyncio.sleep(0.001)
            assert p.cancel_pending_hitl("tok-3") is True

        result, _ = await asyncio.gather(
            requester.request(request, PipelineState()),
            driver(),
        )
        assert result == HITLDecision.CANCEL

    def test_name_and_description(self):
        p = Pipeline()
        r = PipelineResumeRequester(p)
        assert r.name == "pipeline_resume"
        assert "Pipeline.resume" in r.description


# ── End-to-end with HITLStage ─────────────────────────────────────


class TestHITLStageResumeIntegration:
    @pytest.mark.asyncio
    async def test_stage_pauses_then_resumes(self):
        p = Pipeline()
        stage = HITLStage(requester=PipelineResumeRequester(p))
        s = PipelineState(session_id="sess")
        request = HITLRequest(token="tok-e2e", reason="ship?")
        s.shared[HITL_REQUEST_KEY] = request

        async def driver():
            for _ in range(100):
                if "tok-e2e" in p._pending_hitl:
                    break
                await asyncio.sleep(0.001)
            p.resume("tok-e2e", HITLDecision.APPROVE)

        await asyncio.gather(
            stage.execute(input=None, state=s),
            driver(),
        )
        assert s.shared[HITL_LAST_DECISION_KEY] == "approve"
        assert len(s.shared[HITL_HISTORY_KEY]) == 1
        # Token registration cleaned up.
        assert p.list_pending_hitl() == []

    @pytest.mark.asyncio
    async def test_stage_timeout_with_resume_requester_uses_policy(self):
        """If no resume() call arrives in time, the timeout policy
        decides — same contract as any other slow requester."""
        p = Pipeline()
        stage = HITLStage(
            requester=PipelineResumeRequester(p),
            timeout=AutoRejectTimeout(timeout_seconds=0.05),
        )
        s = PipelineState(session_id="sess")
        s.shared[HITL_REQUEST_KEY] = HITLRequest(token="tok-timeout", reason="x")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "reject"
        # Token cleaned up even on timeout.
        assert "tok-timeout" not in p._pending_hitl
