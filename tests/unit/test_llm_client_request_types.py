"""Tests for the extended APIRequest / APIResponse / TokenUsage (Phase A1).

Validates the new optional fields and additive aggregation semantics.
"""

from __future__ import annotations

import sys
import os
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


# ---------------------------------------------------------------------------
# APIRequest
# ---------------------------------------------------------------------------


def test_apirequest_has_response_format_and_session_hint() -> None:
    req = APIRequest(model="x", messages=[])
    assert req.response_format is None
    assert req.session_hint is None


def test_apirequest_response_format_json_schema() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    req = APIRequest(
        model="x",
        messages=[],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    assert req.response_format == {"type": "json_schema", "json_schema": schema}


def test_apirequest_session_hint_resume() -> None:
    req = APIRequest(
        model="x",
        messages=[],
        session_hint={"session_id": "abc-123", "resume": True},
    )
    assert req.session_hint == {"session_id": "abc-123", "resume": True}


def test_apirequest_asdict_round_trip() -> None:
    req = APIRequest(
        model="x",
        messages=[],
        response_format={"type": "json_object"},
        session_hint={"session_id": "s1"},
    )
    d = asdict(req)
    assert d["response_format"] == {"type": "json_object"}
    assert d["session_hint"] == {"session_id": "s1"}


# ---------------------------------------------------------------------------
# TokenUsage cost / duration
# ---------------------------------------------------------------------------


def test_tokenusage_defaults_cost_and_duration_none() -> None:
    usage = TokenUsage()
    assert usage.cost_usd is None
    assert usage.duration_ms is None


def test_tokenusage_set_cost_and_duration() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.0125, duration_ms=2400)
    assert usage.cost_usd == pytest.approx(0.0125)
    assert usage.duration_ms == 2400


def test_tokenusage_add_aggregates_cost() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=20, cost_usd=0.01, duration_ms=500)
    b = TokenUsage(input_tokens=30, output_tokens=40, cost_usd=0.02, duration_ms=750)
    c = a + b
    assert c.input_tokens == 40
    assert c.output_tokens == 60
    assert c.cost_usd == pytest.approx(0.03)
    assert c.duration_ms == 1250


def test_tokenusage_add_one_side_none_is_treated_as_zero() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=20, cost_usd=0.01)
    b = TokenUsage(input_tokens=30, output_tokens=40)  # cost None
    c = a + b
    assert c.cost_usd == pytest.approx(0.01)


def test_tokenusage_add_both_sides_none_stays_none() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=20)
    b = TokenUsage(input_tokens=30, output_tokens=40)
    c = a + b
    assert c.cost_usd is None
    assert c.duration_ms is None


def test_tokenusage_iadd_aggregates_cost() -> None:
    a = TokenUsage(input_tokens=10, output_tokens=20, cost_usd=0.01, duration_ms=100)
    b = TokenUsage(input_tokens=5, output_tokens=10, cost_usd=0.005, duration_ms=200)
    a += b
    assert a.input_tokens == 15
    assert a.cost_usd == pytest.approx(0.015)
    assert a.duration_ms == 300


# ---------------------------------------------------------------------------
# APIResponse.cost_usd proxy
# ---------------------------------------------------------------------------


def test_apiresponse_cost_usd_proxies_usage() -> None:
    resp = APIResponse(
        content=[ContentBlock(type="text", text="hi")],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5, cost_usd=0.001),
    )
    assert resp.cost_usd == pytest.approx(0.001)


def test_apiresponse_cost_usd_none_when_unset() -> None:
    resp = APIResponse(
        content=[ContentBlock(type="text", text="hi")],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    assert resp.cost_usd is None
