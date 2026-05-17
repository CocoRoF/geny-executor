#!/usr/bin/env python3
"""Tiny fake ``gh`` binary used by Copilot CLI tests.

Only the ``copilot`` subcommand is handled. Scenarios driven by
``FAKE_GH_SCENARIO`` env var:

  - ``ok``               — emit ``FAKE_GH_TEXT`` (default greeting) to stdout
  - ``auth_fail``        — exit 4 with "not logged in" stderr
  - ``not_installed``    — exit 1 with "extension not found" stderr
  - ``permission_fail``  — exit 1 with "permission denied" stderr
  - ``crash``            — exit 2 with generic stderr
  - ``hang``             — sleep forever
  - ``echo_argv``        — print argv as JSON
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List


def _ok(argv: List[str]) -> int:
    text = os.environ.get("FAKE_GH_TEXT", "Hello from fake gh copilot.")
    sys.stdout.write(text)
    return 0


def _auth_fail(argv: List[str]) -> int:
    sys.stderr.write("gh: not logged in. Run `gh auth login`.\n")
    return 4


def _not_installed(argv: List[str]) -> int:
    sys.stderr.write("gh: extension not found: github/gh-copilot\n")
    return 1


def _permission_fail(argv: List[str]) -> int:
    sys.stderr.write("permission denied: tool shell(rm) blocked\n")
    return 1


def _crash(argv: List[str]) -> int:
    sys.stderr.write("gh: copilot service unreachable\n")
    return 2


def _hang(argv: List[str]) -> int:
    time.sleep(60)
    return 0


def _echo_argv(argv: List[str]) -> int:
    sys.stdout.write(json.dumps(argv))
    return 0


SCENARIOS = {
    "ok": _ok,
    "auth_fail": _auth_fail,
    "not_installed": _not_installed,
    "permission_fail": _permission_fail,
    "crash": _crash,
    "hang": _hang,
    "echo_argv": _echo_argv,
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "copilot":
        sys.stderr.write(f"fake_gh: only the 'copilot' subcommand is supported (got {argv!r})\n")
        return 99
    scenario = os.environ.get("FAKE_GH_SCENARIO", "ok")
    fn = SCENARIOS.get(scenario)
    if fn is None:
        sys.stderr.write(f"fake_gh: unknown scenario {scenario!r}\n")
        return 99
    return fn(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
