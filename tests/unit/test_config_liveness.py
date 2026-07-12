"""Config-liveness contract — every declared schema field must be ALIVE.

Audit 2026-06-09 §3.5 ("wire it or delete it") and §2.1: fields that
pass schema validation, serialize, round-trip through ``get_config`` —
and are read by NOTHING — are decoys. The operator sees a green check
and no behaviour change; Geny prod shipped exactly this with the
evaluation-chain knobs. Wave 1 wired the knobs the audit named (s04
``fail_fast``/``max_chain_length``, s05 ``cache_prefix``, s06
``timeout_ms``, s16 ``max_turns``); this suite makes liveness a
standing CONTRACT instead of a one-time cleanup:

  * Every field in every one of the 21 stages' declared ConfigSchemas
    must be classified in ``LIVENESS`` below as one of:
      - ``Probe``    — a callable in THIS module demonstrating that
                       setting the field changes observable behaviour;
      - ``CoveredBy``— an existing test module pins the behaviour
                       (pointer is verified to exist and to mention
                       the field, so it can't rot silently);
      - ``Decoy``    — known-inert today; pinned by a strict-xfail
                       probe in this module so the moment someone
                       wires it the xfail flips and the entry must be
                       upgraded to ``Probe``;
      - ``Reserved`` — intentionally not behaviour-affecting; the
                       field's own description must say so
                       ('reserved' / 'ui-only').
  * A NEW schema field that is not classified fails
    ``test_every_schema_field_is_classified`` with "add a liveness
    probe or mark reserved" — future decoy fields cannot land quietly.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Tuple, Union

import pytest

from geny_executor.core.errors import GuardRejectError
from geny_executor.core.introspection import STAGE_MODULES, create_stage
from geny_executor.core.state import PipelineState

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Inventory: all declared stage-level schema fields
# ---------------------------------------------------------------------------


def _schema_inventory() -> Dict[Tuple[int, str], Any]:
    """(stage_order, field_name) → ConfigField for all 21 stages."""
    inventory: Dict[Tuple[int, str], Any] = {}
    for order in sorted(STAGE_MODULES):
        stage = create_stage(order)
        schema = stage.get_config_schema()
        if schema is None:
            continue
        for field in schema.fields:
            inventory[(order, field.name)] = field
    return inventory


# ---------------------------------------------------------------------------
# Classification entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """Behavioural probe living in this module."""

    fn: Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class CoveredBy:
    """Behaviour pinned by an existing test module (repo-relative path)."""

    test_path: str


@dataclass(frozen=True)
class Decoy:
    """Known-inert field — pinned by a strict-xfail probe below."""

    xfail_test: str  # name of the xfail test function in this module


@dataclass(frozen=True)
class Reserved:
    """Intentionally non-behavioural; description must declare it."""


Entry = Union[Probe, CoveredBy, Decoy, Reserved]


# ---------------------------------------------------------------------------
# Probes — each demonstrates one observable behaviour change
# ---------------------------------------------------------------------------


async def _probe_s02_stateless() -> None:
    """stateless=True flips should_bypass — the engine skips the stage."""
    from geny_executor.stages.s02_context import ContextStage

    stage = ContextStage()
    assert stage.should_bypass(PipelineState()) is False
    stage.update_config({"stateless": True})
    assert stage.should_bypass(PipelineState()) is True


async def _probe_s03_prompt() -> None:
    """prompt lands verbatim on state.system for the API call."""
    from geny_executor.stages.s03_system import SystemStage

    stage = SystemStage()
    stage.update_config({"prompt": "You are the liveness probe."})
    state = PipelineState()
    await stage.execute("in", state)
    assert state.system == "You are the liveness probe."


async def _probe_s03_template_vars() -> None:
    """template_vars substitute {name} placeholders into the built prompt."""
    from geny_executor.stages.s03_system import SystemStage

    stage = SystemStage()
    stage.update_config(
        {"prompt": "Hello {name}", "template_vars": {"name": "Liveness"}}
    )
    state = PipelineState()
    await stage.execute("in", state)
    assert "Liveness" in str(state.system)


async def _probe_s03_volatile_placement() -> None:
    """volatile_placement decides whether volatile blocks (clock/memory)
    leave the system prompt (turn_context) or stay in it (system)."""
    from geny_executor.stages.s03_system import SystemStage
    from geny_executor.stages.s03_system.artifact.default.builders import (
        ComposablePromptBuilder,
        DateTimeBlock,
        PersonaBlock,
    )

    def _stage() -> SystemStage:
        return SystemStage(
            builder=ComposablePromptBuilder(
                blocks=[PersonaBlock("Persona."), DateTimeBlock()]
            )
        )

    # Default: volatile tail leaves the system prompt.
    stage = _stage()
    state = PipelineState()
    await stage.execute("in", state)
    assert "Current date" not in str(state.system)
    assert "Current date" in state.shared.get("turn_context_text", "")

    # Legacy: volatile tail stays in the system prompt.
    stage = _stage()
    stage.update_config({"volatile_placement": "system"})
    state = PipelineState()
    await stage.execute("in", state)
    assert "Current date" in str(state.system)
    assert "turn_context_text" not in state.shared


class _CountingGuard:
    """Minimal Guard double recording whether it ran."""

    def __init__(self, label: str, *, passed: bool) -> None:
        self._label = label
        self._passed = passed
        self.calls = 0

    @property
    def name(self) -> str:
        return self._label

    def check(self, state: PipelineState):  # noqa: ANN001
        from geny_executor.stages.s04_guard.types import GuardResult

        self.calls += 1
        return GuardResult(
            passed=self._passed,
            guard_name=self._label,
            message="" if self._passed else f"{self._label} says no",
            action="reject",
        )


async def _probe_s04_max_chain_length() -> None:
    """Oversized chains are rejected with an error naming the knob."""
    from geny_executor.stages.s04_guard import GuardStage

    stage = GuardStage(
        guards=[_CountingGuard("a", passed=True), _CountingGuard("b", passed=True)]
    )
    await stage.execute("in", PipelineState())  # within the default limit

    stage.update_config({"max_chain_length": 1})
    with pytest.raises(GuardRejectError, match="max_chain_length=1"):
        await stage.execute("in", PipelineState())


async def _probe_s04_fail_fast() -> None:
    """fail_fast toggles first-failure short-circuit vs run-all."""
    from geny_executor.stages.s04_guard import GuardStage

    first = _CountingGuard("g1", passed=False)
    second = _CountingGuard("g2", passed=False)
    stage = GuardStage(guards=[first, second])  # default fail_fast=True
    with pytest.raises(GuardRejectError):
        await stage.execute("in", PipelineState())
    assert second.calls == 0  # short-circuited

    first2 = _CountingGuard("g1", passed=False)
    second2 = _CountingGuard("g2", passed=False)
    stage2 = GuardStage(guards=[first2, second2])
    stage2.update_config({"fail_fast": False})
    with pytest.raises(GuardRejectError):
        await stage2.execute("in", PipelineState())
    assert second2.calls == 1  # every guard ran, violations aggregated


async def _probe_s05_cache_prefix() -> None:
    """Different prefixes namespace the cache key for identical content."""
    from geny_executor.stages.s05_cache import CacheStage, SystemCacheStrategy

    def _state() -> PipelineState:
        state = PipelineState(session_id="s")
        state.system = "You are a helpful assistant."
        return state

    state_a, state_b = _state(), _state()
    stage_a = CacheStage(strategy=SystemCacheStrategy())
    stage_a.update_config({"cache_prefix": "tenant-a"})
    stage_b = CacheStage(strategy=SystemCacheStrategy())
    stage_b.update_config({"cache_prefix": "tenant-b"})
    await stage_a.execute("x", state_a)
    await stage_b.execute("x", state_b)
    assert state_a.shared["cache_key"].startswith("tenant-a:")
    assert state_a.shared["cache_key"] != state_b.shared["cache_key"]


async def _probe_s06_provider() -> None:
    """provider selects which client class the stage builds and calls."""
    from geny_executor.stages.s06_api import APIStage

    anthropic_client = APIStage(api_key="k", provider="anthropic")._resolve_client(
        PipelineState()
    )
    stage = APIStage(api_key="k", provider="anthropic")
    stage.update_config({"provider": "openai"})
    openai_client = stage._resolve_client(PipelineState())
    assert anthropic_client.provider == "anthropic"
    assert openai_client.provider == "openai"
    assert type(anthropic_client) is not type(openai_client)


async def _probe_s06_base_url() -> None:
    """base_url reaches the constructed client (vLLM / proxy routing)."""
    from geny_executor.stages.s06_api import APIStage

    stage = APIStage(api_key="k", provider="openai")
    stage.update_config({"base_url": "http://probe.local/v1"})
    client = stage._resolve_client(PipelineState())
    assert client._base_url == "http://probe.local/v1"


async def _probe_s06_stream() -> None:
    """stream=False at the stage level routes through the non-streaming
    client call (no text.delta events), winning over state.stream=True."""
    from geny_executor.stages.s06_api import APIStage, MockProvider

    stage = APIStage(provider=MockProvider(default_text="alpha beta"))
    stage.update_config({"stream": False})
    state = PipelineState(session_id="stream-knob")
    state.add_message("user", "hi")
    await stage.execute("in", state)
    deltas = [e for e in state.events if e["type"] == "text.delta"]
    assert deltas == [], (
        "stream=False at the stage level should route through the "
        "non-streaming client call (no text.delta events)"
    )


async def _probe_s06_timeout_ms() -> None:
    """timeout_ms reaches the call site: clients that can't take the
    kwarg get the api.timeout_unsupported event instead of a silent drop."""
    from geny_executor.stages.s06_api import APIStage, MockProvider

    def _events(state: PipelineState) -> list:
        return [e["type"] for e in state.events if e["type"] == "api.timeout_unsupported"]

    stage = APIStage(provider=MockProvider(default_text="x"))
    state = PipelineState(session_id="t0")
    state.add_message("user", "hi")
    await stage.execute("in", state)
    assert _events(state) == []  # knob unset → nothing to report

    stage2 = APIStage(provider=MockProvider(default_text="x"))
    stage2.update_config({"timeout_ms": 1234})
    state2 = PipelineState(session_id="t1")
    state2.add_message("user", "hi")
    await stage2.execute("in", state2)
    assert _events(state2) == ["api.timeout_unsupported"]


class _RecordingOrchestrator:
    """AgentOrchestrator double that always wants to delegate."""

    name = "recording"

    def __init__(self) -> None:
        self.calls = 0

    def configure(self, config):  # noqa: ANN001 — Strategy surface
        pass

    async def orchestrate(self, state):  # noqa: ANN001
        from geny_executor.stages.s12_agent.types import AgentResult

        self.calls += 1
        return AgentResult(
            delegated=True,
            sub_results=[{"agent_type": "probe", "success": True, "text": "hi"}],
        )


async def _probe_s12_max_delegations() -> None:
    """max_delegations truncates delegate_requests before dispatch; cap=0
    refuses delegation outright and announces agent.delegations_capped."""
    from geny_executor.stages.s12_agent import AgentStage

    orchestrator = _RecordingOrchestrator()
    stage = AgentStage(orchestrator=orchestrator)
    stage.update_config({"max_delegations": 0})
    state = PipelineState(session_id="delegation-cap")
    state.delegate_requests = [{"agent_type": "probe", "task": "x"}]

    await stage.execute("in", state)

    assert state.agent_results == [], (
        "max_delegations=0 must refuse delegation — sub-agent results "
        "were appended anyway"
    )
    assert orchestrator.calls == 0, "orchestrator dispatched despite cap=0"
    capped = [e for e in state.events if e["type"] == "agent.delegations_capped"]
    assert [e["data"] for e in capped] == [{"requested": 1, "cap": 0}]


async def _probe_s16_max_turns() -> None:
    """max_turns caps the loop ahead of state.max_iterations."""
    from geny_executor.stages.s16_loop import LoopStage

    def _state() -> PipelineState:
        state = PipelineState(session_id="loop")
        state.iteration = 1
        state.pending_tool_calls = [{"id": "t1"}]
        return state

    state = _state()
    await LoopStage().execute("in", state)
    assert state.loop_decision == "continue"  # default cap is max_iterations=50

    stage = LoopStage()
    stage.update_config({"max_turns": 1})
    state2 = _state()
    await stage.execute("in", state2)
    assert state2.loop_decision == "complete"


async def _probe_s16_early_stop_on() -> None:
    """Listed completion signals abort the loop immediately."""
    from geny_executor.stages.s16_loop import LoopStage

    def _state() -> PipelineState:
        state = PipelineState(session_id="loop")
        state.iteration = 1
        state.pending_tool_calls = [{"id": "t1"}]
        state.completion_signal = "budget_exhausted"
        return state

    state = _state()
    await LoopStage().execute("in", state)
    assert state.loop_decision == "continue"  # signal unrecognized → loop on

    stage = LoopStage()
    stage.update_config({"early_stop_on": ["budget_exhausted"]})
    state2 = _state()
    await stage.execute("in", state2)
    assert state2.loop_decision == "complete"


async def _probe_s18_stateless() -> None:
    from geny_executor.stages.s18_memory import MemoryStage

    stage = MemoryStage()  # default AppendOnlyStrategy → not bypassed
    assert stage.should_bypass(PipelineState()) is False
    stage.update_config({"stateless": True})
    assert stage.should_bypass(PipelineState()) is True


async def _probe_s18_persistence_path() -> None:
    """Setting the path swaps in FilePersistence and writes the session."""
    from geny_executor.stages.s18_memory import MemoryStage

    state = PipelineState(session_id="liveness-sess")
    state.add_message("user", "hello")
    stage = MemoryStage()
    await stage.execute("in", state)
    assert not any(e["type"] == "memory.persisted" for e in state.events)

    with tempfile.TemporaryDirectory() as tmp:
        stage2 = MemoryStage()
        stage2.update_config({"persistence_path": tmp})
        state2 = PipelineState(session_id="liveness-sess")
        state2.add_message("user", "hello")
        await stage2.execute("in", state2)
        assert any(e["type"] == "memory.persisted" for e in state2.events)
        written = [f for _r, _d, fs in os.walk(tmp) for f in fs]
        assert written, "FilePersistence wrote nothing under persistence_path"


# ---------------------------------------------------------------------------
# The curated map — keep in lockstep with the stage schemas
# ---------------------------------------------------------------------------


LIVENESS: Dict[Tuple[int, str], Entry] = {
    (2, "stateless"): Probe(_probe_s02_stateless),
    (3, "prompt"): Probe(_probe_s03_prompt),
    (3, "template_vars"): Probe(_probe_s03_template_vars),
    (3, "volatile_placement"): Probe(_probe_s03_volatile_placement),
    (4, "max_chain_length"): Probe(_probe_s04_max_chain_length),
    (4, "fail_fast"): Probe(_probe_s04_fail_fast),
    (5, "cache_prefix"): Probe(_probe_s05_cache_prefix),
    (6, "provider"): Probe(_probe_s06_provider),
    (6, "base_url"): Probe(_probe_s06_base_url),
    (6, "stream"): Probe(_probe_s06_stream),
    (6, "timeout_ms"): Probe(_probe_s06_timeout_ms),
    (10, "max_concurrency"): CoveredBy("tests/unit/test_tool_stage_max_concurrency.py"),
    (12, "max_delegations"): Probe(_probe_s12_max_delegations),
    (16, "max_turns"): Probe(_probe_s16_max_turns),
    (16, "early_stop_on"): Probe(_probe_s16_early_stop_on),
    (18, "stateless"): Probe(_probe_s18_stateless),
    (18, "persistence_path"): Probe(_probe_s18_persistence_path),
}


# ---------------------------------------------------------------------------
# The contract tests
# ---------------------------------------------------------------------------


def test_every_schema_field_is_classified() -> None:
    """Exhaustiveness both ways: a NEW schema field must come with a
    liveness probe (or an explicit reserved marking), and a removed
    field must take its map entry with it."""
    inventory = set(_schema_inventory())
    classified = set(LIVENESS)

    unclassified = sorted(inventory - classified)
    assert not unclassified, (
        f"schema fields without a liveness classification: {unclassified} — "
        "add a liveness probe or mark reserved (see LIVENESS in "
        "tests/unit/test_config_liveness.py)"
    )

    stale = sorted(classified - inventory)
    assert not stale, (
        f"LIVENESS entries for fields no longer in any stage schema: {stale} — "
        "remove the stale entries"
    )


@pytest.mark.parametrize(
    "key",
    [k for k, v in LIVENESS.items() if isinstance(v, Probe)],
    ids=lambda k: f"s{k[0]:02d}.{k[1]}",
)
@pytest.mark.asyncio
async def test_field_liveness_probe(key: Tuple[int, str]) -> None:
    entry = LIVENESS[key]
    assert isinstance(entry, Probe)
    await entry.fn()


@pytest.mark.parametrize(
    "key",
    [k for k, v in LIVENESS.items() if isinstance(v, CoveredBy)],
    ids=lambda k: f"s{k[0]:02d}.{k[1]}",
)
def test_covered_by_pointer_is_honest(key: Tuple[int, str]) -> None:
    """A CoveredBy pointer must reference an existing test module that
    actually mentions the field — otherwise the pointer rots into a
    liveness claim nothing backs."""
    entry = LIVENESS[key]
    assert isinstance(entry, CoveredBy)
    path = REPO_ROOT / entry.test_path
    assert path.exists(), f"{entry.test_path} no longer exists"
    assert key[1] in path.read_text(), (
        f"{entry.test_path} never mentions field {key[1]!r} — "
        "point at the real covering test or write a probe"
    )


def test_decoy_entries_have_their_xfail_probe() -> None:
    """Each Decoy entry must name a strict-xfail test in this module —
    the mechanism that forces the entry to be upgraded when wired."""
    module_names = set(globals())
    for key, entry in LIVENESS.items():
        if isinstance(entry, Decoy):
            assert entry.xfail_test in module_names, (
                f"Decoy entry {key} names missing test {entry.xfail_test!r}"
            )


def test_reserved_fields_say_so_in_their_description() -> None:
    """Reserved is an explicit, user-visible contract — the schema
    description (what UIs render) must carry the word, not just this map."""
    inventory = _schema_inventory()
    for key, entry in LIVENESS.items():
        if isinstance(entry, Reserved):
            description = inventory[key].description.lower()
            assert "reserved" in description or "ui-only" in description, (
                f"field {key} is marked Reserved in LIVENESS but its schema "
                f"description does not say so: {inventory[key].description!r}"
            )


# ---------------------------------------------------------------------------
# Decoy probes — strict xfail; flipping one means the field got wired
# and its LIVENESS entry must become Probe(...)
#
# (Empty since 2.2.0 wave 4: the s03 template_vars / s06 stream /
# s12 max_delegations decoys were wired and their entries upgraded to
# Probe(...) above. The Decoy mechanism stays for the next inert field.)
# ---------------------------------------------------------------------------
