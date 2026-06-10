"""2.2.0 Wave 2 — order-keyed stage diffing + ``drift_against``.

Audit §3.1 / stage_model: ``EnvironmentDiff.compute`` collapsed
unequal-length stage lists into ONE opaque "changed" blob, so a
16-stage stored manifest diffed against the 21-stage canonical layout
reported nothing usable. Stage lists (any list of dicts with unique
int ``order`` keys) now diff per order with stable
``stages[order=N].…`` paths.
"""

from __future__ import annotations

from geny_executor.core.diff import EnvironmentDiff
from geny_executor.core.environment import EnvironmentManifest
from geny_executor.core.manifest_factory import build_manifest


def _entry(order, name, active=True, **extra):
    d = {"order": order, "name": name, "active": active}
    d.update(extra)
    return d


class TestOrderKeyedDiff:
    def test_unequal_length_reports_per_order_added(self):
        a = {"stages": [_entry(1, "input")]}
        b = {"stages": [_entry(1, "input"), _entry(6, "api"), _entry(9, "parse")]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary == {"added": 2, "removed": 0, "changed": 0}
        assert {e.path for e in diff.entries} == {
            "stages[order=6]",
            "stages[order=9]",
        }
        added_6 = next(e for e in diff.entries if e.path == "stages[order=6]")
        assert added_6.new_value == _entry(6, "api")

    def test_unequal_length_reports_per_order_removed(self):
        a = {"stages": [_entry(1, "input"), _entry(6, "api")]}
        b = {"stages": [_entry(1, "input")]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary == {"added": 0, "removed": 1, "changed": 0}
        assert diff.entries[0].path == "stages[order=6]"
        assert diff.entries[0].old_value == _entry(6, "api")

    def test_same_order_changes_use_order_path(self):
        a = {"stages": [_entry(16, "loop", config={"max_turns": 30})]}
        b = {"stages": [_entry(16, "loop", config={"max_turns": 10})]}
        diff = EnvironmentDiff.compute(a, b)
        assert len(diff.entries) == 1
        assert diff.entries[0].path == "stages[order=16].config.max_turns"
        assert diff.entries[0].old_value == 30
        assert diff.entries[0].new_value == 10

    def test_reordered_arrays_with_equal_entries_diff_empty(self):
        """Array position is presentation; order keys are identity. The
        positional differ would have reported every field changed."""
        a = {"stages": [_entry(1, "input"), _entry(6, "api")]}
        b = {"stages": [_entry(6, "api"), _entry(1, "input")]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.identical

    def test_duplicate_orders_fall_back_to_positional(self):
        """Duplicate orders make order-keying dishonest — same-length
        dict lists keep the positional element-wise diff."""
        a = {"stages": [_entry(6, "api", active=True), _entry(6, "api", active=True)]}
        b = {"stages": [_entry(6, "api", active=True), _entry(6, "api", active=False)]}
        diff = EnvironmentDiff.compute(a, b)
        assert [e.path for e in diff.entries] == ["stages[1].active"]

    def test_lists_without_order_keys_keep_legacy_behaviour(self):
        a = {"tags": ["a", "b"]}
        b = {"tags": ["a", "b", "c"]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary["changed"] == 1
        assert diff.entries[0].path == "tags"

    def test_mixed_dicts_without_order_fall_back(self):
        a = {"items": [{"name": "x"}]}
        b = {"items": [{"name": "x"}, {"name": "y"}]}
        diff = EnvironmentDiff.compute(a, b)
        # Not order-keyed (no 'order' keys), unequal length → changed blob.
        assert diff.summary["changed"] == 1


class TestSixteenVsTwentyOne:
    def test_stored_16_slot_manifest_diffs_meaningfully_against_canon(self):
        """The audit's motivating case: a pre-9a 16-slot manifest vs the
        21-slot canonical layout must name the five missing orders, not
        emit one giant blob."""
        canon = build_manifest("worker_adaptive", provider="anthropic").to_dict()
        stored = build_manifest("worker_adaptive", provider="anthropic").to_dict()
        stored["stages"] = [
            e for e in stored["stages"] if e["order"] not in (11, 13, 15, 19, 20)
        ]
        diff = EnvironmentDiff.compute(stored, canon)
        added_paths = {e.path for e in diff.filter_by_type("added").entries}
        assert added_paths == {
            "stages[order=11]",
            "stages[order=13]",
            "stages[order=15]",
            "stages[order=19]",
            "stages[order=20]",
        }


class TestDriftAgainst:
    def test_clean_blank_layout_reports_strategy_choices_only(self):
        """drift_against compares to the introspected default layout; a
        preset manifest's drift is exactly its deliberate strategy/config
        choices — and every entry is addressable per order."""
        manifest = build_manifest("worker_adaptive", provider="anthropic")
        diff = manifest.drift_against()
        assert not diff.identical
        assert all(e.path.startswith("stages[order=") for e in diff.entries)

    def test_missing_stage_shows_as_removed(self):
        manifest = build_manifest("worker_adaptive", provider="anthropic")
        manifest.stages = [e for e in manifest.stages if e["order"] != 19]
        diff = manifest.drift_against()
        removed = [e for e in diff.entries if e.change_type == "removed"]
        assert any(e.path == "stages[order=19]" for e in removed)

    def test_accepts_precomputed_catalog(self):
        from geny_executor.core.introspection import introspect_all

        catalog = introspect_all()
        manifest = build_manifest("vtuber", provider="anthropic")
        diff_cached = manifest.drift_against(catalog)
        diff_fresh = manifest.drift_against()
        assert {e.path for e in diff_cached.entries} == {e.path for e in diff_fresh.entries}

    def test_from_dict_roundtrip_keeps_drift_stable(self):
        manifest = build_manifest("vtuber", provider="anthropic")
        reloaded = EnvironmentManifest.from_dict(manifest.to_dict())
        assert {e.path for e in reloaded.drift_against().entries} == {
            e.path for e in manifest.drift_against().entries
        }
