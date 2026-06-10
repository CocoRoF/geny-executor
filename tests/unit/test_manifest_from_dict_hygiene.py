"""2.2.0 Wave 2 — ``EnvironmentManifest.from_dict`` load hygiene.

Two behaviours under test:

1. Unknown top-level / per-stage-entry keys still load (forward
   compat) but warn once per load, listing the offenders — the silent
   drop is what drove GAPT to dual-write its model settings
   (audit §1-3, "両쪽 다 써야 안전").
2. The legacy ``s06.strategies['provider'] == 'mock'`` value is
   migrated to ``'anthropic'`` on load — absorbed from Geny's
   ``_migrate_legacy_mock_provider`` (pre-0.13.5 blank manifests
   recorded MockProvider from session-less introspection; at runtime
   every "agent reply" became the literal string "Mock response").
"""

from __future__ import annotations

import copy
import logging

from geny_executor.core.environment import EnvironmentManifest

from tests._fixtures.manifest_entries import required_stage_entries

_ENV_LOGGER = "geny_executor.core.environment"


def _payload(**overrides):
    data = {
        "version": "3.0",
        "metadata": {"id": "env_h", "name": "hygiene"},
        "model": {},
        "pipeline": {},
        "stages": required_stage_entries(),
        "tools": {},
    }
    data.update(overrides)
    return data


class TestUnknownKeyWarnings:
    def test_clean_payload_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            EnvironmentManifest.from_dict(_payload())
        assert not [r for r in caplog.records if "unknown" in r.message]

    def test_unknown_top_level_key_warns_listing_it(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            EnvironmentManifest.from_dict(_payload(modle={"model": "typo"}))
        messages = [r.getMessage() for r in caplog.records]
        assert any("modle" in m and "top-level" in m for m in messages)

    def test_unknown_stage_entry_key_warns_listing_it(self, caplog):
        stages = required_stage_entries()
        stages[0]["strategys"] = {"validator": "default"}  # typo'd key
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            EnvironmentManifest.from_dict(_payload(stages=stages))
        messages = [r.getMessage() for r in caplog.records]
        assert any("strategys" in m and "stage-entry" in m for m in messages)

    def test_one_warning_per_load_even_with_many_offenders(self, caplog):
        stages = required_stage_entries()
        stages[0]["ghost_a"] = 1
        stages[1]["ghost_b"] = 2
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            EnvironmentManifest.from_dict(_payload(stages=stages, ghost_top=True))
        unknown_warnings = [r for r in caplog.records if "unknown" in r.getMessage()]
        assert len(unknown_warnings) == 1
        message = unknown_warnings[0].getMessage()
        assert "ghost_top" in message
        assert "ghost_a" in message and "ghost_b" in message

    def test_unknown_keys_are_still_accepted(self):
        """Back-compat: the payload loads; the foreign key is simply not
        consumed (it lives only in the original dict)."""
        manifest = EnvironmentManifest.from_dict(_payload(future_section={"x": 1}))
        assert manifest.metadata.name == "hygiene"


class TestLegacyMockProviderMigration:
    def _mock_payload(self):
        stages = required_stage_entries()
        for entry in stages:
            if entry["order"] == 6:
                entry["strategies"] = {"provider": "mock", "retry": "exponential_backoff"}
                entry["config"] = {}
        return _payload(stages=stages)

    def test_mock_provider_rewritten_to_anthropic(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            manifest = EnvironmentManifest.from_dict(self._mock_payload())
        s6 = next(e for e in manifest.stages if e["order"] == 6)
        assert s6["strategies"]["provider"] == "anthropic"
        # Other strategies untouched.
        assert s6["strategies"]["retry"] == "exponential_backoff"
        assert any("mock" in r.getMessage() for r in caplog.records)

    def test_input_payload_is_not_mutated(self):
        payload = self._mock_payload()
        snapshot = copy.deepcopy(payload)
        EnvironmentManifest.from_dict(payload)
        assert payload == snapshot

    def test_non_default_artifact_untouched(self):
        """Geny's shim scoped the rewrite to artifact='default' — a mock
        *artifact* legitimately wants a mock provider."""
        stages = required_stage_entries()
        for entry in stages:
            if entry["order"] == 6:
                entry["artifact"] = "mock"
                entry["strategies"] = {"provider": "mock"}
        manifest = EnvironmentManifest.from_dict(_payload(stages=stages))
        s6 = next(e for e in manifest.stages if e["order"] == 6)
        assert s6["strategies"]["provider"] == "mock"

    def test_non_mock_value_untouched(self):
        stages = required_stage_entries()
        for entry in stages:
            if entry["order"] == 6:
                entry["strategies"] = {"provider": "openai"}
        manifest = EnvironmentManifest.from_dict(_payload(stages=stages))
        s6 = next(e for e in manifest.stages if e["order"] == 6)
        assert s6["strategies"]["provider"] == "openai"

    def test_v1_payload_migrates_through_version_chain(self):
        """The mock rewrite runs after the v1→v3 chain, so legacy payloads
        get both migrations in one load."""
        payload = {
            "version": "1.0",
            "metadata": {"id": "env_l", "name": "legacy"},
            "stages": [
                {"order": 6, "name": "api", "active": True,
                 "strategies": {"provider": "mock"}},
            ],
        }
        manifest = EnvironmentManifest.from_dict(payload)
        assert manifest.version == "3.0"
        s6 = next(e for e in manifest.stages if e["order"] == 6)
        assert s6["strategies"]["provider"] == "anthropic"
        # v2→v3 padding happened too.
        assert {e["order"] for e in manifest.stages} >= {6, 11, 13, 15, 19, 20}
