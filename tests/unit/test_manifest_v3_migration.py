"""Unit tests for the v2 → v3 manifest migration (S9a.4)."""

from __future__ import annotations

import json

from geny_executor.core.environment import (
    MANIFEST_VERSION,
    EnvironmentManifest,
)


def _v2_payload(stages: list[dict] | None = None) -> dict:
    base = stages if stages is not None else [
        {"order": 1, "name": "input", "active": True},
        {"order": 6, "name": "api", "active": True, "artifact": "default"},
        {"order": 9, "name": "parse", "active": True},
        # In the v2 numbering, yield was at order 16. The migration
        # leaves the entry's order untouched — Pipeline.from_manifest
        # uses entry.name to resolve the stage class, and the class
        # reports its actual order on registration.
        {"order": 16, "name": "yield", "active": True},
    ]
    return {
        "version": "2.0",
        "metadata": {"id": "env_legacy_v2", "name": "Legacy v2"},
        "model": {"model": "claude-sonnet-4-6"},
        "pipeline": {"name": "legacy"},
        "stages": base,
        "tools": {},
    }


# ── Version detection ───────────────────────────────────────────────────


class TestVersionDetection:
    def test_explicit_v2_payload_migrates(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        assert m.version == MANIFEST_VERSION
        assert MANIFEST_VERSION == "3.0"

    def test_v3_payload_passes_through_unchanged(self):
        # A payload already at the current version should not have
        # its stages re-touched.
        m = EnvironmentManifest.from_dict(_v2_payload())
        roundtripped = EnvironmentManifest.from_dict(m.to_dict())
        assert roundtripped.to_dict() == m.to_dict()

    def test_v1_payload_chains_through_v2_to_v3(self):
        v1 = {
            "version": "1.0",
            "metadata": {"id": "env_v1"},
            "stages": [{"order": 1, "name": "input", "active": True}],
        }
        m = EnvironmentManifest.from_dict(v1)
        assert m.version == MANIFEST_VERSION
        # 1 from v1 + 5 v3 pads.
        assert len(m.stages) == 6


# ── Padding behaviour ──────────────────────────────────────────────────


class TestPadding:
    def test_pads_all_five_new_orders_when_missing(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        orders = {int(s["order"]) for s in m.stages}
        for order in (11, 13, 15, 19, 20):
            assert order in orders

    def test_padded_entries_inactive_by_default(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        by_order = {int(s["order"]): s for s in m.stages}
        for order in (11, 13, 15, 19, 20):
            assert by_order[order]["active"] is False

    def test_padded_entries_use_default_artifact(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        by_order = {int(s["order"]): s for s in m.stages}
        for order in (11, 13, 15, 19, 20):
            assert by_order[order]["artifact"] == "default"

    def test_padded_entry_names(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        by_order = {int(s["order"]): s for s in m.stages}
        assert by_order[11]["name"] == "tool_review"
        assert by_order[13]["name"] == "task_registry"
        assert by_order[15]["name"] == "hitl"
        assert by_order[19]["name"] == "summarize"
        assert by_order[20]["name"] == "persist"

    def test_existing_entries_preserved_byte_for_byte(self):
        original = _v2_payload(
            stages=[
                {
                    "order": 6,
                    "name": "api",
                    "active": True,
                    "artifact": "default",
                    "strategies": {"router": "adaptive"},
                    "strategy_configs": {"router": {"min_budget": 1000}},
                    "config": {"timeout_ms": 5000},
                    "tool_binding": {"allowed": ["x"]},
                    "model_override": {"model": "claude-opus-4-7"},
                    "chain_order": {},
                }
            ]
        )
        m = EnvironmentManifest.from_dict(original)
        api = next(s for s in m.stages if int(s["order"]) == 6)
        assert api["strategies"] == {"router": "adaptive"}
        assert api["strategy_configs"] == {"router": {"min_budget": 1000}}
        assert api["config"] == {"timeout_ms": 5000}
        assert api["tool_binding"] == {"allowed": ["x"]}
        assert api["model_override"] == {"model": "claude-opus-4-7"}

    def test_already_present_new_order_not_double_padded(self):
        # If the v2 payload already contained an order=11 entry
        # (e.g. someone hand-edited a manifest), the migration must
        # not duplicate it.
        payload = _v2_payload(
            stages=[
                {"order": 1, "name": "input", "active": True},
                {
                    "order": 11,
                    "name": "tool_review",
                    "active": True,
                    "artifact": "custom",
                },
            ]
        )
        m = EnvironmentManifest.from_dict(payload)
        order_11 = [s for s in m.stages if int(s["order"]) == 11]
        assert len(order_11) == 1
        # The pre-existing entry's `active=True` and `artifact="custom"`
        # are not overwritten by the pad.
        assert order_11[0]["active"] is True
        assert order_11[0]["artifact"] == "custom"

    def test_stages_sorted_by_order(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        orders = [int(s["order"]) for s in m.stages]
        assert orders == sorted(orders)


# ── End-to-end JSON round-trip ─────────────────────────────────────────


class TestJsonRoundTrip:
    def test_v2_json_loads_as_v3(self):
        raw = json.dumps(_v2_payload())
        m = EnvironmentManifest.from_dict(json.loads(raw))
        assert m.version == MANIFEST_VERSION

    def test_v3_serialises_with_current_version_field(self):
        m = EnvironmentManifest.from_dict(_v2_payload())
        out = m.to_dict()
        assert out["version"] == MANIFEST_VERSION
