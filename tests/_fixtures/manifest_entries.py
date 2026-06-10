"""Shared manifest stage-entry helpers for tests.

2.2.0 strict ``Pipeline.from_manifest`` enforces the required-stage
contract (``introspection._STAGE_REQUIRED``): ``s01_input`` /
``s06_api`` / ``s09_parse`` / ``s21_yield`` must be present and active,
because a pipeline without input/api/parse/yield is not an agent loop
(audit 2026-06-09 §1-2 — strict mode used to happily build "pipelines"
with no LLM call in them).

Many older test manifests were minimal on purpose (``stages=[]`` for
tool-registration tests, a lone Stage-6 entry for provider-location
tests). Those fixtures now compose this helper so the *subject under
test* stays the same while the manifest meets the structural contract
every real manifest has to meet anyway.
"""

from __future__ import annotations

from typing import Any, Dict, List

from geny_executor.core.environment import StageManifestEntry


def required_stage_entries(provider: str = "anthropic") -> List[Dict[str, Any]]:
    """Minimal active stage entries satisfying strict required-stage validation.

    Returns dicts (``StageManifestEntry.to_dict()`` form) ready to be
    assigned to ``EnvironmentManifest.stages`` or concatenated with a
    test's own entries. Stage 6 carries ``config['provider']`` because
    an active API stage without a provider is itself a strict error.
    """
    return [
        StageManifestEntry(order=1, name="input", active=True).to_dict(),
        StageManifestEntry(
            order=6, name="api", active=True, config={"provider": provider}
        ).to_dict(),
        StageManifestEntry(order=9, name="parse", active=True).to_dict(),
        StageManifestEntry(order=21, name="yield", active=True).to_dict(),
    ]
