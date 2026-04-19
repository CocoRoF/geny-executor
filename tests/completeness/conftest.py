"""Shared fixtures for the completeness suite.

Loads `docs/MEMORY_SPEC.yaml` as the `spec` fixture and exposes the
`provider_kind` parametrize point that Phase 2+ implementations fill
in. Until Phase 1 lands `EphemeralMemoryProvider`, `provider_kind`
is the empty list and the parametrized C-tests are collected but
report "no providers registered" — clearly distinguishing "not yet
implemented" from "broken".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "docs" / "MEMORY_SPEC.yaml"


@pytest.fixture(scope="session")
def spec_path() -> Path:
    return SPEC_PATH


@pytest.fixture(scope="session")
def spec() -> Dict[str, Any]:
    """The frozen MEMORY_SPEC.yaml as a dict.

    PyYAML is loaded lazily so the fixture only fails (with a clear
    skip reason) inside tests that actually need the spec, instead of
    aborting collection of the entire suite.
    """
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def registered_providers() -> List[str]:
    """Names of MemoryProvider implementations the C-suite should run
    against. Empty until Phase 1 (EphemeralMemoryProvider) and Phase 2
    (FileMemoryProvider, SQLMemoryProvider, ...) register entries.

    The activation point is intentionally a fixture rather than a hard
    import so we can run `tests/completeness/` in any partial state.
    """
    return []


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Tag every C-criterion test with its acceptance phase so we can
    select e.g. `pytest -m c_phase_2` once Phase 2 lands.
    """
    for item in items:
        name = item.nodeid.rsplit("/", 1)[-1]
        if name.startswith("test_c1_"):
            item.add_marker(pytest.mark.c_phase_1)
        elif (
            name.startswith("test_c2_")
            or name.startswith("test_c3_")
            or name.startswith("test_c5_")
            or name.startswith("test_c6_")
        ):
            item.add_marker(pytest.mark.c_phase_2)
        elif name.startswith("test_c4_"):
            item.add_marker(pytest.mark.c_phase_4)
        elif name.startswith("test_c7_"):
            item.add_marker(pytest.mark.c_phase_3)
