"""Shared fixtures for the completeness suite.

Exposes:
- `spec` / `spec_path` — the frozen `docs/MEMORY_SPEC.yaml` as a dict.
- `registered_providers` — list of `(name, factory)` tuples that the
  C-criteria tests iterate over. Each factory is an async callable
  `(tmp_path: Path) -> MemoryProvider` returning a freshly-initialised
  provider, which keeps per-provider setup out of the test bodies.

Phase 2 populates `registered_providers` with the providers that
survive each sub-PR: `ephemeral`, `file`, `sql`, and `geny-adapter`
(quarantined C7 fixture). C1·C2·C3·C5·C6 activate by virtue of the
list being non-empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import pytest

from geny_executor.memory.provider import MemoryProvider
from geny_executor.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)
from tests.completeness.fixtures.adapter import GenyManagerAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "docs" / "MEMORY_SPEC.yaml"


ProviderFactory = Callable[[Path], Awaitable[MemoryProvider]]


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


# ── provider factories ─────────────────────────────────────────────


async def _ephemeral_factory(_: Path) -> MemoryProvider:
    provider = EphemeralMemoryProvider()
    await provider.initialize()
    return provider


async def _file_factory(tmp: Path) -> MemoryProvider:
    provider = FileMemoryProvider(root=tmp / "file_root")
    await provider.initialize()
    return provider


async def _sql_factory(tmp: Path) -> MemoryProvider:
    provider = SQLMemoryProvider(dsn=str(tmp / "sql.db"))
    await provider.initialize()
    return provider


async def _adapter_factory(_: Path) -> MemoryProvider:
    provider = GenyManagerAdapter()
    await provider.initialize()
    return provider


@pytest.fixture(scope="session")
def registered_providers() -> List[Tuple[str, ProviderFactory]]:
    """Providers the C-suite parametrises over.

    Empty-list means "Phase 0/1" — every C-test skips with a clear
    reason. With Phase 2e landed, all four providers ship: ephemeral,
    file, sql, and the quarantined geny-adapter fixture that C7
    (Phase 3) will parity-check the native providers against.
    """
    return [
        ("ephemeral", _ephemeral_factory),
        ("file", _file_factory),
        ("sql", _sql_factory),
        ("geny-adapter", _adapter_factory),
    ]


# ── collection hook ─────────────────────────────────────────────────


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
