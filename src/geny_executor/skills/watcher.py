"""Skill registry hot-reload — Phase 10.7.

The registry is built once at session boot. Operators editing
``SKILL.md`` files (especially their own at ``~/.geny/skills/<id>/``)
want changes to land in the *current* session, not the next one.

This module provides :class:`SkillRegistryWatcher` — a lightweight
poll-based watcher that re-scans the configured roots on a fixed
interval and rebuilds the registry when something on disk changed.
Stdlib only (``os.stat`` + ``threading``); no chokidar / watchdog
dependency, no platform-specific paths.

Why polling, not OS-level watch? Two reasons:

1. **Portability.** The same poll loop works on Linux, macOS,
   Windows, and inside Docker without bind-mount eventing quirks.
   ``watchdog`` / ``inotify`` work great on bare metal but fall
   over with NFS / SSHFS / certain Docker volume configurations.
2. **Cost.** Skill catalogs are small (single-digit to low double-
   digit files). Polling 50 ``stat()`` calls every 2s is on the
   order of microseconds; the saved complexity is worth more than
   the saved CPU.

Hosts that want OS-level watching can swap in a custom watcher —
the public surface is a single ``start()`` / ``stop()`` pair, easy
to shim.

Usage::

    registry = SkillRegistry()
    registry.register_many(load_skills_dir(Path("~/.geny/skills")).loaded)

    watcher = SkillRegistryWatcher(
        registry,
        roots=[Path("~/.geny/skills").expanduser()],
        on_change=lambda report: logger.info("reloaded %d skills", len(report.loaded)),
    )
    watcher.start()
    # ... session lifetime ...
    watcher.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from geny_executor.skills.loader import SkillLoadReport, load_skills_dir
from geny_executor.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


# Tuple ((path, mtime, size), ...) — the minimum we need to detect a
# change without parsing every file.
_FileSig = tuple


def _signature_for_root(root: Path) -> _FileSig:
    """Walk ``root`` and produce a signature representing every
    ``SKILL.md`` file's mtime + size. Sorted by path for determinism.

    Missing root returns the empty signature; the caller treats that
    the same as "nothing to watch" without distinguishing root-was-
    deleted from root-never-existed (idempotent semantics).
    """
    if not root.exists() or not root.is_dir():
        return tuple()
    entries: List[tuple] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        try:
            st = skill_md.stat()
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        entries.append((str(skill_md), st.st_mtime_ns, st.st_size))
    entries.sort()
    return tuple(entries)


def _signature(roots: Sequence[Path]) -> _FileSig:
    """Aggregate signature for multiple roots — used to detect any
    change across the whole watched set in one tick."""
    out: List[tuple] = []
    for root in roots:
        out.extend(_signature_for_root(root))
    out.sort()
    return tuple(out)


OnChangeCallback = Callable[[SkillLoadReport], None]


class SkillRegistryWatcher:
    """Poll-based skill-registry hot-reload.

    Owns a background thread that wakes every ``poll_interval_s``,
    re-scans the configured ``roots``, and — if any file's mtime or
    size has changed — rebuilds the registry from scratch. The
    registry is mutated in place (existing :class:`SkillTool` /
    :class:`SkillToolProvider` references stay valid).

    Args:
        registry: Registry to refresh on change. **Mutated in place.**
        roots: Directories to watch. Each follows the standard
            ``<root>/<id>/SKILL.md`` convention.
        poll_interval_s: Seconds between scans. Default 2s — fast
            enough to feel snappy when the operator hits Save in
            their editor, slow enough that idle pollers don't show
            up in profiling.
        debounce_s: Wait at least this long after the most recent
            on-disk change before triggering the reload. Lets the
            operator finish saving (some editors do write-rename
            atomic flips that emit two stat changes back-to-back).
        on_change: Optional callback fired after every successful
            reload with the resulting :class:`SkillLoadReport`.
        on_error: Optional callback fired when a reload pass fails.
            Default behaviour: log at WARNING and keep the old
            registry contents intact.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        roots: Iterable[Path],
        poll_interval_s: float = 2.0,
        debounce_s: float = 0.3,
        on_change: Optional[OnChangeCallback] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self._registry = registry
        self._roots: List[Path] = [Path(r) for r in roots]
        self._poll_interval_s = max(0.1, float(poll_interval_s))
        self._debounce_s = max(0.0, float(debounce_s))
        self._on_change = on_change
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Snapshot taken at start() so the first reload only fires on
        # an actual change after start, not an unconditional reload.
        self._last_sig: _FileSig = tuple()

    # ── Public API ─────────────────────────────────────────────────

    def start(self) -> None:
        """Begin polling. Idempotent — calling twice is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_sig = _signature(self._roots)
        self._thread = threading.Thread(
            target=self._run,
            name=f"skill-watcher:{','.join(str(r) for r in self._roots)}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "SkillRegistryWatcher: started poll=%ss debounce=%ss roots=%s",
            self._poll_interval_s,
            self._debounce_s,
            [str(r) for r in self._roots],
        )

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and join the thread. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reload_now(self) -> Optional[SkillLoadReport]:
        """Trigger an immediate reload pass synchronously. Useful
        from tests, or from a UI "refresh" button. Bypasses the
        debounce. Returns the new report (``None`` if reload failed
        and ``on_error`` was wired)."""
        return self._do_reload()

    # ── Internal loop ──────────────────────────────────────────────

    def _run(self) -> None:
        pending_since: Optional[float] = None
        while not self._stop.is_set():
            try:
                sig = _signature(self._roots)
            except Exception as exc:  # noqa: BLE001
                self._handle_error(exc)
                self._stop.wait(self._poll_interval_s)
                continue

            if sig != self._last_sig:
                # Something changed. Wait for the debounce window
                # before reloading — handles editor write-rename
                # double-stats.
                if pending_since is None:
                    pending_since = time.monotonic()
                if time.monotonic() - pending_since >= self._debounce_s:
                    self._do_reload(new_sig=sig)
                    pending_since = None
            else:
                pending_since = None

            self._stop.wait(self._poll_interval_s)

    def _do_reload(
        self,
        *,
        new_sig: Optional[_FileSig] = None,
    ) -> Optional[SkillLoadReport]:
        """Rebuild the registry from current disk state."""
        try:
            loaded: List = []
            errors: List = []
            for root in self._roots:
                report = load_skills_dir(root)
                loaded.extend(report.loaded)
                errors.extend(report.errors)
            # Replace registry contents atomically. We unregister all
            # then register many — between those calls the registry
            # is briefly empty, but list_tools() callers always
            # snapshot before reading so the race window doesn't
            # matter in practice.
            self._registry.clear()
            self._registry.register_many(loaded)
            self._last_sig = new_sig if new_sig is not None else _signature(self._roots)
            full_report = SkillLoadReport(loaded=loaded, errors=errors)
            if self._on_change is not None:
                try:
                    self._on_change(full_report)
                except Exception as exc:  # noqa: BLE001
                    self._handle_error(exc)
            return full_report
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
            return None

    def _handle_error(self, exc: Exception) -> None:
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:  # noqa: BLE001
                logger.exception("SkillRegistryWatcher: on_error itself raised")
        else:
            logger.warning("SkillRegistryWatcher: %s", exc, exc_info=True)


__all__ = [
    "SkillRegistryWatcher",
]
