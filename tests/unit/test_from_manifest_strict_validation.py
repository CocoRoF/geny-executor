"""2.2.0 Wave 2 — strict ``from_manifest`` runs ``validate_manifest``.

Staged strictness (audit §5.3 decision): error-severity issues raise
:class:`ConfigError` at strict build time, listing every issue at once
(no fix-one-rebuild-discover-the-next loops); warnings log. Lenient
mode logs everything — errors included, downgraded — and keeps
building, because lenient is the recovery path and recovery must be
loud, not blind.
"""

from __future__ import annotations

import logging

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    StageManifestEntry,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.llm_client.credentials import ConfigError, CredentialBundle

from tests._fixtures.manifest_entries import required_stage_entries

_PIPE_LOGGER = "geny_executor.core.pipeline"


def _manifest(extra_entries=(), *, drop_orders=()) -> EnvironmentManifest:
    stages = [e for e in required_stage_entries() if e["order"] not in drop_orders]
    stages += [e.to_dict() for e in extra_entries]
    return EnvironmentManifest(stages=stages)


class TestStrictErrors:
    def test_missing_required_stage_raises(self):
        with pytest.raises(ConfigError, match="stage.required_inactive"):
            Pipeline.from_manifest(
                _manifest(drop_orders=(9,)), credentials=CredentialBundle(), strict=True
            )

    def test_noop_configure_config_raises(self):
        entry = StageManifestEntry(
            order=5, name="cache", active=True,
            strategies={"strategy": "system_cache"},
            strategy_configs={"strategy": {"ttl": 60}},
        )
        with pytest.raises(ConfigError, match="strategy.config_dropped"):
            Pipeline.from_manifest(
                _manifest([entry]), credentials=CredentialBundle(), strict=True
            )

    def test_error_lists_every_issue_at_once(self):
        """Operators get the full bill in one raise, not one error per
        rebuild."""
        bad_loop = StageManifestEntry(
            order=16, name="loop", active=True,
            strategies={"controller": "no_such_controller"},
        )
        with pytest.raises(ConfigError) as exc_info:
            Pipeline.from_manifest(
                _manifest([bad_loop], drop_orders=(9,)),
                credentials=CredentialBundle(),
                strict=True,
            )
        message = str(exc_info.value)
        assert "strategy.unknown_impl" in message
        assert "stage.required_inactive" in message

    def test_strict_warnings_log_but_build(self, caplog):
        entry = StageManifestEntry(
            order=4, name="guard", active=True,
            chain_order={"guards": ["token_budget"]},  # unappliable → warning
        )
        with caplog.at_level(logging.WARNING, logger=_PIPE_LOGGER):
            pipeline = Pipeline.from_manifest(
                _manifest([entry]), credentials=CredentialBundle(), strict=True
            )
        assert pipeline.get_stage(4) is not None
        assert any("chain.order_unappliable" in r.getMessage() for r in caplog.records)


class TestLenientLogsOnly:
    def test_lenient_logs_errors_and_builds(self, caplog):
        """Lenient is the recovery path: it must keep loading degraded
        manifests, but no longer silently — every error-severity issue
        logs as a warning."""
        with caplog.at_level(logging.WARNING, logger=_PIPE_LOGGER):
            pipeline = Pipeline.from_manifest(
                _manifest(drop_orders=(9, 21)),
                credentials=CredentialBundle(),
                strict=False,
            )
        assert pipeline.get_stage(1) is not None
        messages = [r.getMessage() for r in caplog.records]
        assert any("stage.required_inactive" in m for m in messages)

    def test_lenient_clean_manifest_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_PIPE_LOGGER):
            Pipeline.from_manifest(
                _manifest(), credentials=CredentialBundle(), strict=False
            )
        assert not [r for r in caplog.records if "manifest issue" in r.getMessage()]
