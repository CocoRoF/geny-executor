"""SettingsLoader — hierarchical JSON cascade with deep merge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``. Dict-typed values
    merge recursively; everything else (lists, scalars) replaces.
    The original ``base`` is not mutated."""
    out = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


class SettingsLoader:
    """Multi-file JSON cascade.

    Pass paths in priority order — earlier paths are LOWER priority,
    later paths overlay (typical user → project → local).

    ``load`` is lazy + cached; ``reload`` invalidates. Sections are
    accessed via ``get_section(name)``; if a schema was registered for
    that name, the loader validates + returns the parsed model. Without
    a schema the raw dict is returned.
    """

    def __init__(self, paths: List[Path]) -> None:
        self._paths = list(paths)
        self._raw: Optional[Dict[str, Any]] = None

    @property
    def paths(self) -> List[Path]:
        return list(self._paths)

    def add_path(self, path: Path, *, position: Optional[int] = None) -> None:
        """Register an additional cascade path. Default appends (highest
        priority). Use ``position`` to insert at a specific index."""
        if position is None:
            self._paths.append(path)
        else:
            self._paths.insert(position, path)
        self._raw = None  # invalidate cache

    def load(self) -> Dict[str, Any]:
        """Walk the cascade, deep-merge each file, return the merged dict.

        Files that don't exist are skipped silently (operators may not
        have a project-level settings.json). Files that fail to parse
        log a warning and are skipped — we'd rather start with partial
        config than refuse to boot.
        """
        merged: Dict[str, Any] = {}
        for path in self._paths:
            try:
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("settings_path_unreadable path=%s err=%s", path, exc)
                continue
            try:
                data = json.loads(text) if text.strip() else {}
            except json.JSONDecodeError as exc:
                logger.warning(
                    "settings_invalid_json path=%s err=%s",
                    path,
                    exc,
                )
                continue
            if not isinstance(data, dict):
                logger.warning(
                    "settings_root_not_object path=%s type=%s",
                    path,
                    type(data).__name__,
                )
                continue
            merged = _deep_merge(merged, data)
        self._raw = merged
        return merged

    def reload(self) -> Dict[str, Any]:
        self._raw = None
        return self.load()

    @property
    def raw(self) -> Dict[str, Any]:
        if self._raw is None:
            return self.load()
        return self._raw

    def get_section(self, name: str, default: Any = None) -> Any:
        """Return section ``name``. Validated through registered schema
        when present; otherwise raw dict (or ``default`` when absent)."""
        from geny_executor.settings.section_registry import get_section_schema

        section = self.raw.get(name)
        if section is None:
            return default
        schema = get_section_schema(name)
        if schema is None:
            return section
        try:
            return schema(**section)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "settings_section_invalid name=%s err=%s",
                name,
                exc,
            )
            return default


# ── Process-wide singleton ───────────────────────────────────────────


_DEFAULT: Optional[SettingsLoader] = None


def get_default_loader() -> SettingsLoader:
    """Process singleton. Empty paths until something configures it."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SettingsLoader(paths=[])
    return _DEFAULT


def reset_default_loader() -> SettingsLoader:
    """Test helper — replace singleton with a fresh empty loader."""
    global _DEFAULT
    _DEFAULT = SettingsLoader(paths=[])
    return _DEFAULT


__all__ = ["SettingsLoader", "get_default_loader", "reset_default_loader"]
