"""Unit tests for Stage 15 HITL (S9b.3)."""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s15_hitl import (
    AutoApproveTimeout,
    AutoRejectTimeout,
    CallbackRequester,
    HITLDecision,
    HITLRequest,
    HITLStage,
    HITL_HISTORY_KEY,
    HITL_LAST_DECISION_KEY,
    HITL_REQUEST_KEY,
    IndefiniteTimeout,
    NullRequester,
    coerce_decision,
    coerce_request,
)


def _state() -> PipelineState:
    return PipelineState(session_id="s")


# ── Types & coercion ───────────────────────────────────────────────────


class TestTypes:
    def test_request_to_dict(self):
        r = HITLRequest(reason="why", severity="error", tool_call_id="t1")
        d = r.to_dict()
        assert d["reason"] == "why"
        assert d["severity"] == "error"
        assert d["tool_call_id"] == "t1"
        assert d["token"]

    def test_coerce_request_passthrough(self):
        r = HITLRequest(reason="x")
        assert coerce_request(r) is r

    def test_coerce_request_dict(self):
        r = coerce_request({"reason": "x", "severity": "error"})
        assert r.reason == "x"
        assert r.severity == "error"

    def test_coerce_request_normalises_bad_severity(self):
        r = coerce_request({"reason": "x", "severity": "BOGUS"})
        assert r.severity == "warn"

    def test_coerce_request_none(self):
        assert coerce_request(None) is None

    def test_coerce_decision(self):
        assert coerce_decision("approve") == HITLDecision.APPROVE
        assert coerce_decision(HITLDecision.REJECT) == HITLDecision.REJECT
        assert coerce_decision("nonsense") is None
        assert coerce_decision(None) is None


# ── Requesters ─────────────────────────────────────────────────────────


class TestRequesters:
    @pytest.mark.asyncio
    async def test_null_always_approves(self):
        r = NullRequester()
        d = await r.request(HITLRequest(), _state())
        assert d == HITLDecision.APPROVE

    @pytest.mark.asyncio
    async def test_callback_uses_supplied_callable(self):
        async def cb(req, state):
            return HITLDecision.REJECT

        r = CallbackRequester(callback=cb)
        d = await r.request(HITLRequest(), _state())
        assert d == HITLDecision.REJECT

    @pytest.mark.asyncio
    async def test_callback_no_callable_returns_none(self):
        r = CallbackRequester()
        assert await r.request(HITLRequest(), _state()) is None

    @pytest.mark.asyncio
    async def test_callback_configure(self):
        r = CallbackRequester()

        async def cb(req, state):
            return HITLDecision.APPROVE

        r.configure({"callback": cb})
        assert await r.request(HITLRequest(), _state()) == HITLDecision.APPROVE


# ── Timeout policies ──────────────────────────────────────────────────


class TestTimeoutPolicies:
    def test_indefinite_returns_none_seconds(self):
        p = IndefiniteTimeout()
        assert p.timeout_seconds is None

    def test_auto_approve_validation(self):
        with pytest.raises(ValueError):
            AutoApproveTimeout(timeout_seconds=0)

    def test_auto_approve_returns_approve(self):
        p = AutoApproveTimeout(timeout_seconds=1)
        assert p.on_timeout(HITLRequest(), _state()) == HITLDecision.APPROVE

    def test_auto_reject_returns_reject(self):
        p = AutoRejectTimeout(timeout_seconds=1)
        assert p.on_timeout(HITLRequest(), _state()) == HITLDecision.REJECT

    def test_auto_approve_configure(self):
        p = AutoApproveTimeout(timeout_seconds=1)
        p.configure({"timeout_seconds": 5})
        assert p.timeout_seconds == 5
        with pytest.raises(ValueError):
            p.configure({"timeout_seconds": -1})


# ── HITLStage ─────────────────────────────────────────────────────────


class TestHITLStage:
    def test_default_bypass_when_no_request(self):
        stage = HITLStage()
        assert stage.should_bypass(_state()) is True

    @pytest.mark.asyncio
    async def test_default_null_requester_approves(self):
        stage = HITLStage()
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="touch")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "approve"
        history = s.shared[HITL_HISTORY_KEY]
        assert len(history) == 1
        assert history[0]["decision"] == "approve"

    @pytest.mark.asyncio
    async def test_callback_decision_propagates(self):
        async def cb(req, state):
            return HITLDecision.REJECT

        stage = HITLStage(requester=CallbackRequester(callback=cb))
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="touch")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "reject"
        assert s.loop_decision == "complete"
        assert s.completion_signal == "HITL_REJECTED"

    @pytest.mark.asyncio
    async def test_cancel_sets_escalate(self):
        async def cb(req, state):
            return HITLDecision.CANCEL

        stage = HITLStage(requester=CallbackRequester(callback=cb))
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="risk")
        await stage.execute(input=None, state=s)
        assert s.loop_decision == "escalate"
        assert s.completion_signal == "HITL_CANCELLED"

    @pytest.mark.asyncio
    async def test_request_key_consumed(self):
        stage = HITLStage()
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="x")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_REQUEST_KEY] is None

    @pytest.mark.asyncio
    async def test_emits_request_and_decision_events(self):
        stage = HITLStage()
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="x")
        await stage.execute(input=None, state=s)
        types = [e["type"] for e in s.events]
        assert "hitl.request" in types
        assert "hitl.decision" in types

    @pytest.mark.asyncio
    async def test_timeout_uses_policy_verdict(self):
        async def slow(req, state):
            await asyncio.sleep(1.0)
            return HITLDecision.APPROVE

        stage = HITLStage(
            requester=CallbackRequester(callback=slow),
            timeout=AutoRejectTimeout(timeout_seconds=0.05),
        )
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="x")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "reject"
        evts = [e for e in s.events if e["type"] == "hitl.timeout"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_requester_exception_falls_through_to_cancel(self):
        async def boom(req, state):
            raise RuntimeError("kaboom")

        stage = HITLStage(requester=CallbackRequester(callback=boom))
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="x")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "cancel"
        errs = [e for e in s.events if e["type"] == "hitl.requester_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_no_decision_falls_through_to_timeout_verdict(self):
        async def noop(req, state):
            return None

        stage = HITLStage(
            requester=CallbackRequester(callback=noop),
            timeout=AutoApproveTimeout(timeout_seconds=1),
        )
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="x")
        await stage.execute(input=None, state=s)
        assert s.shared[HITL_LAST_DECISION_KEY] == "approve"
        evts = [e for e in s.events if e["type"] == "hitl.no_decision"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_dict_request_coerced(self):
        stage = HITLStage()
        s = _state()
        s.shared[HITL_REQUEST_KEY] = {"reason": "from dict"}
        await stage.execute(input=None, state=s)
        history = s.shared[HITL_HISTORY_KEY]
        assert history[0]["request"]["reason"] == "from dict"

    @pytest.mark.asyncio
    async def test_history_accumulates_across_runs(self):
        stage = HITLStage()
        s = _state()
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="r1")
        await stage.execute(input=None, state=s)
        s.shared[HITL_REQUEST_KEY] = HITLRequest(reason="r2")
        await stage.execute(input=None, state=s)
        assert len(s.shared[HITL_HISTORY_KEY]) == 2

    def test_slot_registries(self):
        stage = HITLStage()
        slots = stage.get_strategy_slots()
        assert set(slots["requester"].registry) == {"null", "callback"}
        assert set(slots["timeout"].registry) == {
            "indefinite",
            "auto_approve",
            "auto_reject",
        }

    def test_default_strategies(self):
        stage = HITLStage()
        slots = stage.get_strategy_slots()
        assert isinstance(slots["requester"].strategy, NullRequester)
        assert isinstance(slots["timeout"].strategy, IndefiniteTimeout)
