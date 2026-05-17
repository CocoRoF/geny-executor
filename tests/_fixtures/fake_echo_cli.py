#!/usr/bin/env python3
"""Tiny fake CLI used by ``test_cli_runtime``.

Subcommands (positional first arg):

  echo <text>           print <text> to stdout, exit 0
  echo-stdin            copy stdin → stdout, exit 0
  fail <code> <msg>     print <msg> to stderr, exit <code>
  hang <seconds>        sleep <seconds>, exit 0
  lines <n>             emit ``line-0`` .. ``line-{n-1}`` to stdout (newline-delim)
  json-stream <n>       emit ``{"i": 0}`` .. ``{"i": n-1}`` JSON lines

The script intentionally has no side effects and reads no env vars apart from
those caller passes through.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 2

    cmd = args[0]
    rest = args[1:]

    if cmd == "echo":
        sys.stdout.write(" ".join(rest))
        return 0

    if cmd == "echo-stdin":
        sys.stdout.write(sys.stdin.read())
        return 0

    if cmd == "fail":
        code = int(rest[0]) if rest else 1
        msg = " ".join(rest[1:]) if len(rest) > 1 else "failure"
        sys.stderr.write(msg + "\n")
        return code

    if cmd == "hang":
        secs = float(rest[0]) if rest else 5.0
        time.sleep(secs)
        return 0

    if cmd == "lines":
        n = int(rest[0]) if rest else 3
        for i in range(n):
            sys.stdout.write(f"line-{i}\n")
            sys.stdout.flush()
        return 0

    if cmd == "json-stream":
        n = int(rest[0]) if rest else 3
        for i in range(n):
            sys.stdout.write(json.dumps({"i": i}) + "\n")
            sys.stdout.flush()
        return 0

    sys.stderr.write(f"unknown command: {cmd}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
