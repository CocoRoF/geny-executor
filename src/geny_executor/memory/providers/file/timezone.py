"""Timezone resolution for FileMemoryProvider.

Geny uses the `GENY_TIMEZONE` environment variable to stamp all
disk writes (dated LTM filenames, note frontmatter `created` /
`modified`). Replicating that behaviour is a format-compatibility
requirement — a provider that writes UTC where Geny writes KST
produces files that the legacy reader interprets wrong.

Resolution precedence:
  1. Explicit `timezone_name` argument passed to the provider.
  2. `GENY_TIMEZONE` env var (e.g., `Asia/Seoul`, `UTC`, `+09:00`).
  3. Python's `datetime.now().astimezone().tzinfo` (the system local
     timezone).
  4. UTC, as a deterministic last-resort default.

The resolver accepts IANA zone names (`Asia/Seoul`), `UTC`, and
numeric offsets (`+09:00`, `-05:30`). Unknown strings fall through
to the next precedence level rather than raise, so a misconfigured
env var never blocks session creation.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone, tzinfo
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover — Python <3.9 not supported but keep a guard
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]


_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def resolve_timezone(name: Optional[str] = None) -> tzinfo:
    """Return the `tzinfo` for `name`, or fall through the precedence.

    Never raises: an unknown `name` is silently dropped and the env /
    local / UTC fallbacks apply in order.
    """
    for candidate in (name, os.environ.get("GENY_TIMEZONE")):
        parsed = _parse(candidate)
        if parsed is not None:
            return parsed
    local = datetime.now().astimezone().tzinfo
    return local or timezone.utc


def now_in(tz: tzinfo) -> datetime:
    """Current wall-clock time in `tz`."""
    return datetime.now(tz)


def _parse(raw: Optional[str]) -> Optional[tzinfo]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # Numeric offset: "+09:00", "-05:30", "+0900"
    match = _OFFSET_RE.match(raw)
    if match:
        sign, hh, mm = match.groups()
        delta = int(hh) * 60 + int(mm)
        if sign == "-":
            delta = -delta
        from datetime import timedelta

        return timezone(timedelta(minutes=delta), name=raw)
    # IANA zone name ("Asia/Seoul", "UTC", "America/New_York")
    if ZoneInfo is not None:
        try:
            return ZoneInfo(raw)
        except ZoneInfoNotFoundError:
            return None
        except Exception:
            return None
    return None
