"""Settings section schema registry.

Hosts register a section name + a schema (callable that accepts
``**section_dict`` kwargs and returns a parsed model). Without a
registered schema, ``SettingsLoader.get_section`` returns the raw
dict.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# section_name → callable(**dict) → parsed model
_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_section(name: str, schema: Callable[..., Any]) -> None:
    """Register a schema. A second registration for the same name
    overwrites + warns — useful for tests, suspicious in prod."""
    if name in _REGISTRY:
        logger.warning("settings_section_overwritten name=%s", name)
    _REGISTRY[name] = schema


def get_section_schema(name: str) -> Optional[Callable[..., Any]]:
    return _REGISTRY.get(name)


def list_section_names() -> List[str]:
    return sorted(_REGISTRY.keys())


def reset_section_registry() -> None:
    """Test helper — drop all registered schemas."""
    _REGISTRY.clear()


__all__ = [
    "get_section_schema",
    "list_section_names",
    "register_section",
    "reset_section_registry",
]
