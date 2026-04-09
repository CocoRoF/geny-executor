"""Artifact system — pluggable stage implementations.

Each stage directory contains:
  interface.py   — ABC / Protocol definitions (strategy contracts)
  types.py       — Shared data types
  artifact/
    default/     — Built-in implementation
    {custom}/    — User-provided alternative implementations

Convention: every artifact's __init__.py MUST export ``Stage`` — the concrete
stage class that implements ``geny_executor.core.stage.Stage``.

Usage:
    from geny_executor.core.artifact import create_stage, list_artifacts

    # Create a stage from the default artifact
    stage = create_stage("s01_input")

    # Create a stage from a custom artifact
    stage = create_stage("s01_input", artifact="custom_v2", validator=MyValidator())

    # List available artifacts
    names = list_artifacts("s01_input")  # ["default", "custom_v2"]
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Stage

# ── Constants ──

STAGES_PACKAGE = "geny_executor.stages"
ARTIFACT_DIR = "artifact"
DEFAULT_ARTIFACT = "default"

# Canonical stage identifiers (order -> module name)
STAGE_MODULES: Dict[int, str] = {
    1: "s01_input",
    2: "s02_context",
    3: "s03_system",
    4: "s04_guard",
    5: "s05_cache",
    6: "s06_api",
    7: "s07_token",
    8: "s08_think",
    9: "s09_parse",
    10: "s10_tool",
    11: "s11_agent",
    12: "s12_evaluate",
    13: "s13_loop",
    14: "s14_emit",
    15: "s15_memory",
    16: "s16_yield",
}

# Reverse lookup: module name -> order
_MODULE_TO_ORDER: Dict[str, int] = {v: k for k, v in STAGE_MODULES.items()}

# Alias lookup: short name -> module name
STAGE_ALIASES: Dict[str, str] = {
    "input": "s01_input",
    "context": "s02_context",
    "system": "s03_system",
    "guard": "s04_guard",
    "cache": "s05_cache",
    "api": "s06_api",
    "token": "s07_token",
    "think": "s08_think",
    "parse": "s09_parse",
    "tool": "s10_tool",
    "agent": "s11_agent",
    "evaluate": "s12_evaluate",
    "loop": "s13_loop",
    "emit": "s14_emit",
    "memory": "s15_memory",
    "yield": "s16_yield",
}


def _resolve_stage_module(stage: str) -> str:
    """Resolve a stage identifier to its canonical module name.

    Accepts: "s01_input", "input", "1", 1
    """
    if isinstance(stage, int) or stage.isdigit():
        order = int(stage)
        if order not in STAGE_MODULES:
            raise ValueError(f"Unknown stage order: {order}")
        return STAGE_MODULES[order]
    if stage in STAGE_ALIASES:
        return STAGE_ALIASES[stage]
    if stage in _MODULE_TO_ORDER:
        return stage
    raise ValueError(
        f"Unknown stage identifier: {stage!r}. "
        f"Use module name (s01_input), short name (input), or order (1)."
    )


def load_artifact_module(stage: str, artifact: str = DEFAULT_ARTIFACT) -> Any:
    """Import and return an artifact module.

    The returned module must have a ``Stage`` attribute (the concrete class).

    Args:
        stage: Stage identifier (e.g., "s01_input", "input", or "1").
        artifact: Artifact name (directory under ``artifact/``). Default "default".

    Returns:
        The imported module.

    Raises:
        ImportError: If the artifact module cannot be found.
    """
    module_name = _resolve_stage_module(stage)
    module_path = f"{STAGES_PACKAGE}.{module_name}.{ARTIFACT_DIR}.{artifact}"
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot load artifact '{artifact}' for stage '{module_name}': {e}"
        ) from e


def create_stage(stage: str, artifact: str = DEFAULT_ARTIFACT, **kwargs: Any) -> Stage:
    """Create a stage instance from an artifact.

    Args:
        stage: Stage identifier.
        artifact: Artifact name.
        **kwargs: Passed to the Stage constructor.

    Returns:
        An instantiated Stage.
    """
    mod = load_artifact_module(stage, artifact)
    if not hasattr(mod, "Stage"):
        raise AttributeError(
            f"Artifact '{artifact}' for stage '{stage}' does not export 'Stage'. "
            f"Every artifact __init__.py must have: Stage = <ConcreteStageClass>"
        )
    return mod.Stage(**kwargs)


def list_artifacts(stage: str) -> List[str]:
    """List available artifact names for a stage.

    Scans the ``artifact/`` subdirectory for packages.

    Args:
        stage: Stage identifier.

    Returns:
        Sorted list of artifact names.
    """
    module_name = _resolve_stage_module(stage)
    artifact_package = f"{STAGES_PACKAGE}.{module_name}.{ARTIFACT_DIR}"

    try:
        pkg = importlib.import_module(artifact_package)
    except ImportError:
        return []

    if not hasattr(pkg, "__path__"):
        return []

    names: List[str] = []
    for importer, name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if is_pkg:
            names.append(name)

    return sorted(names)


def get_artifact_map(
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a complete stage→artifact mapping.

    Starts with "default" for every stage, then applies overrides.

    Args:
        overrides: Optional dict of stage_identifier→artifact_name.

    Returns:
        Dict mapping canonical module names to artifact names.
    """
    mapping = {mod: DEFAULT_ARTIFACT for mod in STAGE_MODULES.values()}
    if overrides:
        for key, art in overrides.items():
            module_name = _resolve_stage_module(key)
            mapping[module_name] = art
    return mapping
