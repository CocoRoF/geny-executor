"""2.2.0 hook taxonomy honesty tests (audit 2026-06-09 §3.5).

``FIRED_EVENTS`` must track what the engine actually emits — the enum
advertising 16 kinds while ~3 fired cost both hosts dead-handler
debugging time. The grep-driven test below fails the build whenever a
fire-site ships without updating the set (or the set claims an event
nobody fires).
"""

from __future__ import annotations

import re
from pathlib import Path

from geny_executor.hooks import FIRED_EVENTS, HookEvent

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "geny_executor"

# What the engine emits as of 2.2.0: the Stage 10 tool-invocation trio,
# the permission pair fired from the same dispatch path, and the
# pipeline-lifecycle five fired by Pipeline itself (wave 2 — pipeline
# start/end, stage enter/exit, loop iteration end).
EXPECTED_FIRED = {
    HookEvent.PIPELINE_START,
    HookEvent.PIPELINE_END,
    HookEvent.STAGE_ENTER,
    HookEvent.STAGE_EXIT,
    HookEvent.LOOP_ITERATION_END,
    HookEvent.PRE_TOOL_USE,
    HookEvent.POST_TOOL_USE,
    HookEvent.POST_TOOL_FAILURE,
    HookEvent.PERMISSION_REQUEST,
    HookEvent.PERMISSION_DENIED,
}


class TestFiredEvents:
    def test_fired_events_matches_expected_hardcode(self):
        assert FIRED_EVENTS == frozenset(EXPECTED_FIRED)

    def test_fired_events_is_subset_of_taxonomy(self):
        assert FIRED_EVENTS <= set(HookEvent)

    def test_exported_from_package(self):
        import geny_executor.hooks as hooks_pkg

        assert "FIRED_EVENTS" in hooks_pkg.__all__

    def test_fired_events_matches_engine_reality(self):
        """Grep-driven reality check.

        Every ``HookEvent.X`` reference in engine code outside the
        hooks package itself is a fire-site (payload construction or
        ``runner.fire``). If this set diverges from ``FIRED_EVENTS``
        in either direction, someone shipped a fire-site without
        updating the contract — or removed one and left the contract
        stale. Update ``FIRED_EVENTS`` (and its docstring) in
        ``src/geny_executor/hooks/events.py`` together with the code.
        """
        pattern = re.compile(r"HookEvent\.([A-Z_]+)")
        referenced: set = set()
        for path in SRC_ROOT.rglob("*.py"):
            # The hooks package declares the taxonomy; references there
            # are definitions/dispatch plumbing, not fire-sites.
            if "hooks" in path.relative_to(SRC_ROOT).parts[:1]:
                continue
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                referenced.add(match.group(1))

        fired_names = {e.name for e in FIRED_EVENTS}
        assert referenced == fired_names, (
            f"engine fire-sites {sorted(referenced)} != FIRED_EVENTS "
            f"{sorted(fired_names)} — keep hooks/events.py in lockstep "
            "with reality (audit 2026-06-09 §3.5)"
        )

    def test_reserved_events_documented_in_source(self):
        """Every never-fired member carries the 'reserved' marker so
        hosts reading the enum don't bind dead handlers."""
        events_src = (SRC_ROOT / "hooks" / "events.py").read_text(encoding="utf-8")
        for event in HookEvent:
            if event in FIRED_EVENTS:
                continue
            # Find the member's declaration line and check its comment.
            line = next(
                ln for ln in events_src.splitlines() if ln.strip().startswith(f"{event.name} =")
            )
            assert "reserved" in line, (
                f"HookEvent.{event.name} never fires but its declaration "
                "lacks the 'reserved' comment"
            )
