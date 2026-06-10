"""2.2.0 Wave 2 — public ``validate_manifest`` write-time validation.

Audit §1-1: a wide class of manifest declarations was "accepted,
stored, schema-green, and inert". ``validate_manifest`` makes each of
those failure modes a first-class :class:`ManifestIssue` — error where
the declaration cannot take effect, warning where it is merely
suspicious. One test class per check.
"""

from __future__ import annotations

import pytest

from geny_executor import ManifestIssue, build_manifest, validate_manifest
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)

from tests._fixtures.manifest_entries import required_stage_entries


def _manifest(extra_entries=(), *, model=None, version=None) -> EnvironmentManifest:
    """Minimal valid manifest (required stages active) + extra entries."""
    m = EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_v", name="validate"),
        model=dict(model) if model else {},
        pipeline={},
        stages=required_stage_entries() + [e.to_dict() for e in extra_entries],
        tools=ToolsSnapshot(),
    )
    if version is not None:
        m.version = version
    return m


def _codes(issues, severity=None):
    return [i.code for i in issues if severity is None or i.severity == severity]


def _by_code(issues, code):
    found = [i for i in issues if i.code == code]
    assert found, f"expected an issue with code {code!r}; got {[i.code for i in issues]}"
    return found[0]


# ── Baseline ─────────────────────────────────────────────────


class TestCleanManifest:
    def test_minimal_required_manifest_is_clean(self):
        assert validate_manifest(_manifest()) == []

    def test_factory_presets_have_no_errors(self):
        for preset in ("worker_adaptive", "vtuber"):
            issues = validate_manifest(build_manifest(preset, provider="anthropic"))
            assert _codes(issues, "error") == []

    def test_validation_does_not_mutate_manifest(self):
        m = _manifest()
        before = m.to_dict()
        validate_manifest(m)
        assert m.to_dict() == before


# ── Stage identity checks ────────────────────────────────────


class TestStageIdentity:
    def test_unknown_stage_name_active_is_error(self):
        issues = validate_manifest(
            _manifest([StageManifestEntry(order=7, name="tokn", active=True)])
        )
        issue = _by_code(issues, "stage.unknown_name")
        assert issue.severity == "error"
        assert issue.stage_order == 7
        assert issue.stage_name == "tokn"

    def test_unknown_stage_name_inactive_is_warning(self):
        issues = validate_manifest(
            _manifest([StageManifestEntry(order=7, name="tokn", active=False)])
        )
        assert _by_code(issues, "stage.unknown_name").severity == "warning"

    def test_duplicate_orders_with_active_entry_is_error(self):
        issues = validate_manifest(
            _manifest([StageManifestEntry(order=9, name="parse", active=True)])
        )
        issue = _by_code(issues, "stage.duplicate_order")
        assert issue.severity == "error"
        assert issue.stage_order == 9

    def test_duplicate_orders_all_inactive_is_warning(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(order=17, name="emit", active=False),
                    StageManifestEntry(order=17, name="emit", active=False),
                ]
            )
        )
        assert _by_code(issues, "stage.duplicate_order").severity == "warning"

    def test_unknown_artifact_active_is_error(self):
        issues = validate_manifest(
            _manifest(
                [StageManifestEntry(order=17, name="emit", active=True, artifact="ghost")]
            )
        )
        issue = _by_code(issues, "stage.unknown_artifact")
        assert issue.severity == "error"
        assert issue.field == "artifact"

    def test_order_mismatch_is_warning(self):
        issues = validate_manifest(
            _manifest([StageManifestEntry(order=99, name="emit", active=False)])
        )
        issue = _by_code(issues, "stage.order_mismatch")
        assert issue.severity == "warning"
        assert "17" in issue.message


# ── Strategy slot / impl checks ──────────────────────────────


class TestStrategySelections:
    def test_unknown_slot_active_is_error(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategies={"ghost_slot": "standard"},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.unknown_slot")
        assert issue.severity == "error"
        assert issue.field == "strategies.ghost_slot"

    def test_unknown_impl_active_is_error(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategies={"controller": "multi_dim_budgt"},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.unknown_impl")
        assert issue.severity == "error"
        assert "multi_dim_budget" in issue.message  # suggests the registry

    def test_unknown_impl_inactive_is_warning(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=False,
                        strategies={"controller": "multi_dim_budgt"},
                    )
                ]
            )
        )
        assert _by_code(issues, "strategy.unknown_impl").severity == "warning"


class TestStrategyConfigs:
    def test_config_for_noop_configure_is_error(self):
        """THE audit §2.1 class: a strategy_configs block aimed at a
        strategy that never overrode Strategy.configure parses, stores,
        round-trips — and does nothing. ``system_cache`` is one of the
        many no-op-configure impls (verified by scanning slot
        registries for ``configure is Strategy.configure``)."""
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=5, name="cache", active=True,
                        strategies={"strategy": "system_cache"},
                        strategy_configs={"strategy": {"ttl": 60}},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.config_dropped")
        assert issue.severity == "error"
        assert "system_cache" in issue.message

    def test_config_for_real_configure_is_clean(self):
        """multi_dim_budget gained a real configure() in Wave 1 — the
        exact shape Geny prod declares must validate clean."""
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategies={"controller": "multi_dim_budget"},
                        strategy_configs={"controller": {"dimensions": ["iterations"]}},
                    )
                ]
            )
        )
        assert "strategy.config_dropped" not in _codes(issues)

    def test_unpaired_config_is_warning(self):
        """restore only applies a slot's config alongside its strategy
        selection — a config without the matching strategies key never
        lands."""
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategies={},
                        strategy_configs={"controller": {"dimensions": ["iterations"]}},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.config_unpaired")
        assert issue.severity == "warning"

    def test_config_for_unknown_slot_is_error(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategy_configs={"ghost": {"x": 1}},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.unknown_slot")
        assert issue.field == "strategy_configs.ghost"

    def test_config_aimed_at_chain_is_warning(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=4, name="guard", active=True,
                        strategy_configs={"guards": {"x": 1}},
                    )
                ]
            )
        )
        issue = _by_code(issues, "strategy.config_unpaired")
        assert "chain" in issue.message


# ── Chain ordering checks ────────────────────────────────────


class TestChainOrder:
    def test_unknown_chain_is_error_when_active(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=4, name="guard", active=True,
                        chain_order={"gards": ["iteration"]},
                    )
                ]
            )
        )
        assert _by_code(issues, "chain.unknown").severity == "error"

    def test_unknown_chain_impl_is_error_when_active(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=4, name="guard", active=True,
                        chain_order={"guards": ["iteration", "ghost_guard"]},
                    )
                ]
            )
        )
        assert _by_code(issues, "chain.unknown_impl").severity == "error"

    def test_unappliable_order_is_warning(self):
        """The default guard chain ships empty; restore can only reorder
        existing items. Hosts that populate at runtime (Geny's
        populate_guard_chain) suppress by doing exactly that."""
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=4, name="guard", active=True,
                        chain_order={"guards": ["token_budget", "iteration"]},
                    )
                ]
            )
        )
        assert _by_code(issues, "chain.order_unappliable").severity == "warning"

    def test_empty_order_is_ignored(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=17, name="emit", active=True,
                        chain_order={"emitters": []},
                    )
                ]
            )
        )
        assert _codes(issues) == []


# ── Stage config schema check ────────────────────────────────


class TestStageConfigKeys:
    def test_unknown_config_key_is_warning(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        config={"max_turns": 5, "max_trns": 10},
                    )
                ]
            )
        )
        issue = _by_code(issues, "config.unknown_key")
        assert issue.severity == "warning"
        assert "max_trns" in issue.message

    def test_provider_override_is_engine_key_not_unknown(self):
        """``provider_override`` is consumed by Stage.resolve_local_client
        on every stage — it must not be flagged."""
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        config={"provider_override": "openai"},
                    )
                ]
            )
        )
        assert "config.unknown_key" not in _codes(issues)


# ── Structural checks ────────────────────────────────────────


class TestRequiredStages:
    def test_missing_required_stage_is_error(self):
        entries = [e for e in required_stage_entries() if e["order"] != 9]
        m = EnvironmentManifest(stages=entries)
        issues = validate_manifest(m)
        issue = _by_code(issues, "stage.required_inactive")
        assert issue.severity == "error"
        assert issue.stage_name == "s09_parse"

    def test_inactive_required_stage_is_error(self):
        entries = required_stage_entries()
        for e in entries:
            if e["order"] == 6:
                e["active"] = False
        issues = validate_manifest(EnvironmentManifest(stages=entries))
        codes = _codes(issues, "error")
        assert "stage.required_inactive" in codes

    def test_empty_manifest_reports_all_four(self):
        issues = validate_manifest(EnvironmentManifest(stages=[]))
        names = {i.stage_name for i in issues if i.code == "stage.required_inactive"}
        assert names == {"s01_input", "s06_api", "s09_parse", "s21_yield"}


class TestProviderChecks:
    def test_provider_missing_on_active_s06_is_error(self):
        entries = required_stage_entries()
        for e in entries:
            if e["order"] == 6:
                e["config"] = {}
        issues = validate_manifest(EnvironmentManifest(stages=entries))
        assert _by_code(issues, "provider.missing").severity == "error"

    def test_legacy_strategies_provider_is_error_even_inactive(self):
        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=6, name="api", active=False,
                        strategies={"provider": "anthropic"},
                    )
                ]
            )
        )
        assert _by_code(issues, "provider.legacy_location").severity == "error"


class TestModelDualHome:
    def test_model_in_both_homes_is_warning(self):
        entries = required_stage_entries()
        for e in entries:
            if e["order"] == 6:
                e["config"]["model"] = "claude-haiku-4-5"
        m = EnvironmentManifest(stages=entries, model={"model": "claude-sonnet-4-5"})
        issue = _by_code(validate_manifest(m), "model.dual_home")
        assert issue.severity == "warning"
        assert "top-level" in issue.message  # names the single home

    def test_top_level_only_is_clean(self):
        m = _manifest(model={"model": "claude-sonnet-4-5"})
        assert "model.dual_home" not in _codes(validate_manifest(m))


class TestVersionCheck:
    def test_unknown_version_is_warning(self):
        issues = validate_manifest(_manifest(version="9.0"))
        issue = _by_code(issues, "version.unknown")
        assert issue.severity == "warning"
        assert issue.field == "version"

    def test_supported_versions_are_clean(self):
        for v in ("1.0", "2.0", "3.0"):
            assert "version.unknown" not in _codes(validate_manifest(_manifest(version=v)))


# ── Injection + issue shape ──────────────────────────────────


class _FakeSlot:
    def __init__(self, registry):
        self.registry = registry
        self.current_impl = next(iter(registry), "")


class _FakeStage:
    """Catalogue double proving registry_introspection is consulted."""

    def __init__(self):
        self.requested = True

    def get_strategy_slots(self):
        return {"only_slot": _FakeSlot({})}

    def get_strategy_chains(self):
        return {}

    def get_config_schema(self):
        return None


class TestInjection:
    def test_registry_introspection_overrides_catalogue(self):
        requests = []

        def fake_catalogue(module, artifact):
            requests.append((module, artifact))
            return _FakeStage()

        issues = validate_manifest(
            _manifest(
                [
                    StageManifestEntry(
                        order=16, name="loop", active=True,
                        strategies={"controller": "standard"},
                    )
                ]
            ),
            registry_introspection=fake_catalogue,
        )
        assert ("s16_loop", "default") in requests
        # The fake stage has no 'controller' slot → flagged via injected catalogue.
        assert "strategy.unknown_slot" in _codes(issues)


class TestManifestIssueShape:
    def test_to_dict_shape(self):
        issue = ManifestIssue(
            severity="error",
            code="stage.unknown_name",
            message="boom",
            stage_order=7,
            stage_name="tokn",
            field="name",
        )
        assert issue.to_dict() == {
            "severity": "error",
            "code": "stage.unknown_name",
            "message": "boom",
            "stage_order": 7,
            "stage_name": "tokn",
            "field": "name",
        }

    def test_frozen(self):
        issue = ManifestIssue(severity="warning", code="x", message="y")
        with pytest.raises(AttributeError):
            issue.severity = "error"
