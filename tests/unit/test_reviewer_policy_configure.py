"""2.2.0 Wave 1 — s11 reviewer policy reachable via configure() (audit §1-5).

"Policy via config, not hardcode": the reviewers' security knobs
(allowed_hosts, destructive_tools, secret patterns, size thresholds)
were constructor-only — no environment edit could reach them, making
the declared policy surface de-facto hardcode. These tests pin that a
configured reviewer actually *applies* the configured policy, through
the same ``SlotChain.append(impl, config)`` path the stage mutation API
uses (``stage.add_to_chain('reviewers', impl, config)``).
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review.artifact.default.reviewers import (
    DestructiveResultReviewer,
    SensitivePatternReviewer,
    SizeReviewer,
)
from geny_executor.stages.s11_tool_review.artifact.default.stage import ToolReviewStage
from geny_executor.stages.s11_tool_review.interface import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
)


def _state() -> PipelineState:
    return PipelineState(session_id="s")


def _web_call(url: str, call_id: str = "t1") -> dict:
    return {"id": call_id, "name": "WebFetch", "input": {"url": url}}


class TestNetworkAllowlistViaChainConfig:
    @pytest.mark.asyncio
    async def test_disallowed_host_flagged_error(self):
        stage = ToolReviewStage(reviewers=[])
        stage.add_to_chain("reviewers", "network", {"allowed_hosts": ["good.example"]})
        reviewer = stage.get_strategy_chains()["reviewers"].items[0]

        flags = await reviewer.review([_web_call("https://evil.example/x")], [], _state())

        assert len(flags) == 1
        assert flags[0].severity == SEVERITY_ERROR
        assert "evil.example" in flags[0].reason

    @pytest.mark.asyncio
    async def test_allowed_host_stays_advisory(self):
        stage = ToolReviewStage(reviewers=[])
        stage.add_to_chain("reviewers", "network", {"allowed_hosts": ["good.example"]})
        reviewer = stage.get_strategy_chains()["reviewers"].items[0]

        flags = await reviewer.review([_web_call("https://good.example/x")], [], _state())

        assert len(flags) == 1
        assert flags[0].severity == SEVERITY_INFO


class TestSensitivePatternsReplaceDefaults:
    @pytest.mark.asyncio
    async def test_configured_pattern_matches(self):
        reviewer = SensitivePatternReviewer.from_config(
            {"patterns": [["internal_token", r"itok_[a-f0-9]{6}"]]}
        )
        call = {"id": "t1", "name": "Bash", "input": {"command": "echo itok_a1b2c3"}}

        flags = await reviewer.review([call], [], _state())

        assert len(flags) == 1
        assert flags[0].details["pattern"] == "internal_token"

    @pytest.mark.asyncio
    async def test_default_patterns_replaced_not_merged(self):
        """Replacement semantics: a tenant that supplies its own pattern set
        owns the whole policy — leftovers from the default set would make
        the effective policy unauditable from the manifest alone."""
        reviewer = SensitivePatternReviewer.from_config(
            {"patterns": [["internal_token", r"itok_[a-f0-9]{6}"]]}
        )
        # Would match the DEFAULT api_key_assignment pattern.
        call = {"id": "t1", "name": "Bash", "input": {"command": "api_key=hunter2"}}

        flags = await reviewer.review([call], [], _state())

        assert flags == []


class TestDestructiveSeverityEscalation:
    @pytest.mark.asyncio
    async def test_configured_severity_applied(self):
        reviewer = DestructiveResultReviewer.from_config(
            {"destructive_tools": ["Bash"], "severity": SEVERITY_WARN}
        )
        calls = [{"id": "t1", "name": "Bash", "input": {}}]
        results = [{"tool_use_id": "t1", "content": "done"}]

        flags = await reviewer.review(calls, results, _state())

        assert len(flags) == 1
        assert flags[0].severity == SEVERITY_WARN


class TestSizeThresholdsViaConfig:
    @pytest.mark.asyncio
    async def test_configured_thresholds_scale_severity(self):
        reviewer = SizeReviewer.from_config(
            {"warn_threshold_bytes": 5, "error_threshold_bytes": 10}
        )
        results = [
            {"tool_use_id": "small", "content": "1234"},  # under warn
            {"tool_use_id": "mid", "content": "1234567"},  # warn band
            {"tool_use_id": "big", "content": "12345678901"},  # error band
        ]

        flags = await reviewer.review([], results, _state())

        by_id = {f.tool_call_id: f.severity for f in flags}
        assert "small" not in by_id
        assert by_id["mid"] == SEVERITY_WARN
        assert by_id["big"] == SEVERITY_ERROR
