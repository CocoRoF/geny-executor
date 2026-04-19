"""C4 — REST coverage.

Acceptance (`docs/MEMORY_SPEC.yaml::completeness_criteria[3]`):
The web backend exposes the §3.7 endpoint set (notes CRUD, wikilink,
graph, tags, importance filter, keyword search, vector search,
reflect, reindex, snapshot, restore, promote) and round-trips its
dataclass payloads through the active provider with no web-side
domain logic.

State: **red** until Phase 4 lands the executor-web mirror routers.
This file lives in the executor repo as the *contract* the web layer
must satisfy. The actual HTTP-level test runs in
`geny-executor-web/backend/tests/` and pulls this manifest as input.
"""

from __future__ import annotations

import pytest

PHASE_REASON = (
    "C4 awaits Phase 4: executor-web `routers/memory.py` exposing the "
    "§3.7 endpoint set as a thin proxy over MemoryProvider."
)

# The endpoint set the web mirror must implement. Any mismatch with
# `geny-executor-web/backend/app/routers/memory.py` is a Phase 4 bug.
EXPECTED_ROUTES = [
    ("GET",    "/api/sessions/{sid}/memory/descriptor"),
    ("GET",    "/api/sessions/{sid}/memory"),
    ("GET",    "/api/sessions/{sid}/memory/stats"),
    ("GET",    "/api/sessions/{sid}/memory/tags"),
    ("GET",    "/api/sessions/{sid}/memory/graph"),
    ("GET",    "/api/sessions/{sid}/memory/notes"),
    ("GET",    "/api/sessions/{sid}/memory/notes/{filename}"),
    ("POST",   "/api/sessions/{sid}/memory/notes"),
    ("PUT",    "/api/sessions/{sid}/memory/notes/{filename}"),
    ("DELETE", "/api/sessions/{sid}/memory/notes/{filename}"),
    ("POST",   "/api/sessions/{sid}/memory/notes/{filename}/link"),
    ("POST",   "/api/sessions/{sid}/memory/search"),
    ("POST",   "/api/sessions/{sid}/memory/reflect"),
    ("POST",   "/api/sessions/{sid}/memory/reindex"),
    ("GET",    "/api/sessions/{sid}/memory/snapshot"),
    ("POST",   "/api/sessions/{sid}/memory/restore"),
    ("POST",   "/api/sessions/{sid}/memory/promote"),
    ("GET",    "/api/memory/providers"),
    ("GET",    "/api/memory/providers/{name}/config-schema"),
]


def test_c4_route_manifest_is_at_least_eighteen_endpoints():
    """Sanity check the manifest itself: §3.7 demands ≥18."""
    assert len(EXPECTED_ROUTES) >= 18


@pytest.mark.skip(reason=PHASE_REASON)
def test_c4_web_mirror_serves_all_endpoints():
    """When activated this test will:

    1. Import the FastAPI app from `geny_executor_web.app`.
    2. Walk `app.routes` and assert every (method, path) in
       EXPECTED_ROUTES is registered.
    3. For each POST, assert the request schema is the executor-side
       dataclass (RetrievalQuery, NoteDraft, ReindexPlanRequest, ...)
       not a web-private model.
    """
    raise AssertionError("C4 acceptance not yet implemented")  # noqa: TRY003
