"""2.2.0 EventTypes catalogue honesty tests (audit 2026-06-09 §3.2).

The event stream is the de-facto host contract; GAPT shipped a
100%-text-loss bug and a $0-cost bug by *guessing* event names. The
AST-driven test below fails the build whenever an emit site ships a
name the catalogue doesn't list (or vice versa for the engine-owned
families), so the catalogue can never drift from reality the way the
hosts' hand-maintained mapping switches did.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set, Tuple

import pytest

from geny_executor import EVENT_CATALOG_VERSION, EventTypes, known_event_types
from geny_executor.events import PAYLOADS
from geny_executor.memory.provider import MemoryEvent

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "geny_executor"

#: Dynamic emit sites the AST scan cannot resolve to a literal. Keep
#: TINY and justified — every entry is a place where the event name is
#: computed at runtime:
#:
#: - ``core/pipeline.py`` — the ``_pending_runtime_events`` flush loop
#:   (``state.add_event(event_type, data)``); the literals are captured
#:   separately at their ``.append`` sites by the scan below.
#: - ``stages/s16_loop/.../stage.py`` — ``f"loop.{decision}"``; the four
#:   canonical verdicts are catalogued (LOOP_CONTINUE/COMPLETE/ERROR/
#:   ESCALATE). A custom LoopController emitting a fifth verdict owns
#:   its custom name.
#: - ``stages/s10_tool/.../routers.py`` — ``_state_audit``'s own body
#:   forwards its ``event_type`` parameter; the literals live at the
#:   call sites, which the scan resolves.
#: - ``stages/s18_memory`` / ``stages/s02_context`` — ``MemoryEvent.X
#:   .value`` references; covered exhaustively by
#:   ``test_memory_event_values_all_catalogued``.
DYNAMIC_ALLOWLIST: Set[str] = {
    "core/pipeline.py",
    "stages/s16_loop/artifact/default/stage.py",
    "stages/s10_tool/artifact/default/routers.py",
    "stages/s18_memory/artifact/default/stage.py",
    "stages/s02_context/artifact/default/stage.py",
}


def _scan_emit_sites() -> Tuple[Set[str], List[Tuple[str, int]]]:
    """Collect every event-name string the engine can emit.

    Returns ``(literals, unresolved_dynamic_sites)`` where literals
    covers:

    - first positional arg of ``add_event(...)`` calls,
    - second positional arg of ``_state_audit(...)`` calls,
    - first positional arg of ``self._emit(...)`` calls (Pipeline's
      bus-native channel),
    - first element of tuples appended to ``_pending_runtime_events``
      (deferred run-start announcements flushed through add_event),
    - ``"type"`` values of dict literals passed to ``event_sink`` /
      ``self._event_sink`` (the llm_client boundary channel).
    """
    literals: Set[str] = set()
    dynamic: List[Tuple[str, int]] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = str(path.relative_to(SRC_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id

            if name in ("add_event", "_emit", "_state_audit"):
                idx = 1 if name == "_state_audit" else 0
                if len(node.args) > idx:
                    arg = node.args[idx]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        literals.add(arg.value)
                    else:
                        dynamic.append((rel, node.lineno))

            # self._pending_runtime_events.append(("event.name", {...}))
            if (
                name == "append"
                and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "_pending_runtime_events"
                and node.args
                and isinstance(node.args[0], ast.Tuple)
                and node.args[0].elts
            ):
                first = node.args[0].elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    literals.add(first.value)
                else:
                    dynamic.append((rel, node.lineno))

            # self._event_sink({"type": "llm_client...", ...})
            if name == "_event_sink" and node.args and isinstance(node.args[0], ast.Dict):
                for key, value in zip(node.args[0].keys, node.args[0].values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "type"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        literals.add(value.value)

    return literals, dynamic


class TestCatalogueCompleteness:
    def test_every_emitted_literal_is_catalogued(self):
        """The load-bearing check: no emit site ships uncatalogued."""
        literals, _ = _scan_emit_sites()
        catalogued = set(known_event_types())
        missing = sorted(literals - catalogued)
        assert not missing, (
            f"event name(s) emitted in src/ but missing from "
            f"events/catalog.py EventTypes: {missing} — add them (and a "
            "PAYLOADS entry) in the same change (audit 2026-06-09 §3.2)"
        )

    def test_dynamic_sites_are_allowlisted(self):
        """Every non-literal emit site must be a known, documented one."""
        _, dynamic = _scan_emit_sites()
        unexpected = sorted({rel for rel, _ in dynamic} - DYNAMIC_ALLOWLIST)
        assert not unexpected, (
            f"new dynamic event-name emit site(s) {unexpected} — either "
            "use a string literal (preferred; the catalogue test can then "
            "see it) or add the file to DYNAMIC_ALLOWLIST with a "
            "justification comment"
        )

    def test_allowlist_has_no_stale_entries(self):
        """An allowlist entry whose dynamic site disappeared must go."""
        _, dynamic = _scan_emit_sites()
        actual = {rel for rel, _ in dynamic}
        stale = sorted(DYNAMIC_ALLOWLIST - actual)
        assert not stale, f"DYNAMIC_ALLOWLIST entries no longer dynamic: {stale}"

    def test_memory_event_values_all_catalogued(self):
        """``MemoryEvent.X.value`` emit sites are dynamic to the AST scan;
        cover the whole spec enum instead so none can slip through."""
        catalogued = set(known_event_types())
        missing = sorted(e.value for e in MemoryEvent if e.value not in catalogued)
        assert not missing, f"MemoryEvent values missing from EventTypes: {missing}"


class TestCatalogueShape:
    def test_values_are_wire_strings(self):
        assert EventTypes.TEXT_DELTA == "text.delta"
        assert EventTypes.PIPELINE_COMPLETE == "pipeline.complete"
        assert EventTypes.API_TOOL_USE == "api.tool_use"
        assert EventTypes.THINKING_DELTA == "thinking.delta"
        assert EventTypes.LLM_CLIENT_UNKNOWN_WIRE_SHAPE == "llm_client.unknown_wire_shape"

    def test_str_enum_compares_equal_to_raw_strings(self):
        """Hosts match raw strings off the wire; the enum must be one."""
        assert isinstance(EventTypes.TEXT_DELTA, str)
        assert "text.delta" in {EventTypes.TEXT_DELTA}

    def test_values_unique(self):
        values = [e.value for e in EventTypes]
        assert len(values) == len(set(values))

    def test_every_member_has_payload_doc(self):
        missing = [e.value for e in EventTypes if e not in PAYLOADS]
        assert not missing, f"PAYLOADS missing entries: {missing}"

    def test_payloads_has_no_orphan_entries(self):
        orphans = [k for k in PAYLOADS if k not in set(EventTypes)]
        assert not orphans

    def test_known_event_types_sorted_and_complete(self):
        names = known_event_types()
        assert names == sorted(names)
        assert len(names) == len(list(EventTypes))

    def test_catalog_version_is_int(self):
        assert isinstance(EVENT_CATALOG_VERSION, int)
        assert EVENT_CATALOG_VERSION >= 1

    def test_exported_from_package_root(self):
        import geny_executor

        assert geny_executor.EventTypes is EventTypes
        assert "EventTypes" in geny_executor.__all__
        assert "EVENT_CATALOG_VERSION" in geny_executor.__all__
        assert "known_event_types" in geny_executor.__all__


class TestStabilityPins:
    """Names hosts already consume in prod — renaming any of these is a
    major-version event. Pin a representative sample explicitly so a
    refactor cannot 'helpfully' normalise them."""

    @pytest.mark.parametrize(
        "wire",
        [
            "pipeline.start",
            "pipeline.complete",
            "pipeline.error",
            "stage.enter",
            "stage.exit",
            "stage.bypass",
            "stage.error",
            "text.delta",
            "api.request",
            "api.response",
            "tool.execute_start",
            "tool.execute_complete",
            "hitl.request",
            "loop.force_complete",
            "config.override_applied",
            "runtime.llm_client_override",
        ],
    )
    def test_wire_string_present(self, wire: str):
        assert wire in set(known_event_types())
