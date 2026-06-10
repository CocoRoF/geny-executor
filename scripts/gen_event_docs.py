#!/usr/bin/env python3
"""Generate ``docs/events.md`` from the EventTypes catalogue.

The catalogue (``geny_executor.events.catalog``) is the single source
of truth for event names and payload fields; this script renders it to
markdown so the doc can never drift from the code by more than one
regeneration. Run from the repo root::

    python scripts/gen_event_docs.py            # rewrite docs/events.md
    python scripts/gen_event_docs.py --check    # exit 1 if the doc is stale

``--check`` makes the script CI-friendly: it renders to a string and
diffs against the committed file without writing.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geny_executor.events.catalog import (  # noqa: E402  (path bootstrap above)
    EVENT_CATALOG_VERSION,
    PAYLOADS,
    EventTypes,
)

DOC_PATH = REPO_ROOT / "docs" / "events.md"

# Family prefix → human heading. Families not listed fall back to the
# prefix itself, so a new family never breaks generation.
FAMILY_HEADINGS = {
    "pipeline": "Pipeline lifecycle",
    "stage": "Stage lifecycle",
    "config": "Run-start configuration announcements",
    "runtime": "Run-start configuration announcements",
    "loop": "Loop control (Stage 16)",
    "input": "Stage 1 — Input",
    "context": "Stage 2 — Context",
    "system": "Stage 3 — System",
    "guard": "Stage 4 — Guard",
    "cache": "Stage 5 — Cache",
    "api": "Stage 6 — API (incl. streaming chunk forwarding)",
    "text": "Stage 6 — API (incl. streaming chunk forwarding)",
    "thinking": "Stage 6 — API (incl. streaming chunk forwarding)",
    "token": "Stage 7 — Token",
    "think": "Stage 8 — Think",
    "parse": "Stage 9 — Parse",
    "tool": "Stage 10 — Tool",
    "tool_review": "Stage 11 — Tool review",
    "agent": "Stage 12 — Agent",
    "task": "Stage 13 — Task registry",
    "task_registry": "Stage 13 — Task registry",
    "evaluate": "Stage 14 — Evaluate",
    "hitl": "Stage 15 — HITL",
    "emit": "Stage 17 — Emit",
    "memory": "Stage 18 — Memory (+ Stage 2 compaction)",
    "summary": "Stage 19 — Summarize",
    "checkpoint": "Stage 20 — Persist",
    "yield": "Stage 21 — Yield",
    "llm_client": "llm_client event_sink channel (boundary telemetry)",
}


def render() -> str:
    lines: list[str] = []
    out = lines.append

    out("# Event catalogue")
    out("")
    out("<!-- AUTO-GENERATED — do not edit by hand. -->")
    out("<!-- Regenerate: python scripts/gen_event_docs.py -->")
    out("")
    out(f"> Generated from `geny_executor.events.catalog` on {date.today().isoformat()}.")
    out(f"> Catalogue version: **{EVENT_CATALOG_VERSION}** · events: **{len(EventTypes)}**")
    out("")
    out("Every event name the engine emits, value == wire string. The enum")
    out("is a *names registry*, not a rename — consumers matching raw strings")
    out("keep working. Contract rules (see the module docstring for the full")
    out("statement):")
    out("")
    out("- **Append-only**: new events arrive in minor releases; renaming or")
    out("  removing a member is a major-version change.")
    out("- **Payloads may gain fields** in minor releases; existing fields keep")
    out("  their meaning. Field docs below are descriptive, not strict schemas.")
    out("- `…?` marks fields present only in some emissions of the event.")
    out("")
    out("Consume via `pipeline.on(event_type, handler)`, `pipeline.run_stream(...)`,")
    out("or the multi-subscriber tap `pipeline.events(replay_from=...)` (2.2.0).")
    out("`llm_client.*` events travel through the client's `event_sink` callback.")
    out("")

    current_heading = None
    for member in EventTypes:  # definition order == stage order
        family = member.value.split(".", 1)[0]
        heading = FAMILY_HEADINGS.get(family, f"`{family}.*`")
        if heading != current_heading:
            current_heading = heading
            out(f"## {heading}")
            out("")
        payload = PAYLOADS[member]
        out(f"### `{member.value}`")
        out("")
        out(f"Enum member: `EventTypes.{member.name}`")
        out("")
        if payload:
            out("| Field | Description |")
            out("|---|---|")
            for field_name, description in payload.items():
                out(f"| `{field_name}` | {_escape_cell(description)} |")
        else:
            out("_No payload fields — identity is carried on the event envelope_")
            out("_(type / stage / iteration / seq / run_id / session_id)._")
        out("")

    out("---")
    out("")
    out("Companion docs: [error_codes.md](error_codes.md) for the `code` values")
    out("carried by error events; [architecture.md](architecture.md) for where")
    out("each stage sits in the 21-stage layout.")
    out("")
    return "\n".join(lines)


def _escape_cell(text: str) -> str:
    """Keep markdown table cells one-line and pipe-safe."""
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    rendered = render()
    if "--check" in sys.argv[1:]:
        committed = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        # Ignore the generation-date line when diffing so --check does
        # not go stale by calendar alone.
        strip = lambda s: "\n".join(  # noqa: E731
            line for line in s.splitlines() if not line.startswith("> Generated from")
        )
        if strip(committed) != strip(rendered):
            print(
                "docs/events.md is stale — regenerate with: python scripts/gen_event_docs.py",
                file=sys.stderr,
            )
            return 1
        print("docs/events.md is up to date.")
        return 0
    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DOC_PATH.relative_to(REPO_ROOT)} ({len(EventTypes)} events).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
