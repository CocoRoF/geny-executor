"""Path security guard for built-in tools.

Validates that file operations stay within allowed directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def resolve_and_validate(
    file_path: str,
    working_dir: str,
    allowed_paths: Optional[List[str]] = None,
) -> Path:
    """Resolve a file path and validate it against allowed directories.

    Args:
        file_path: Absolute or relative path from the tool input.
        working_dir: Working directory for resolving relative paths.
        allowed_paths: If set, the resolved path must be under one of
            these directories. None means no restriction.

    Returns:
        Resolved absolute Path.

    Raises:
        PermissionError: If the path is outside allowed directories.
        ValueError: If the path is empty.
    """
    if not file_path:
        raise ValueError("file_path must not be empty")

    p = Path(file_path)
    if not p.is_absolute():
        p = Path(working_dir) / p
    resolved = p.resolve()

    if allowed_paths is not None and len(allowed_paths) > 0:
        ok = False
        for ap in allowed_paths:
            allowed = Path(ap).resolve()
            try:
                resolved.relative_to(allowed)
                ok = True
                break
            except ValueError:
                continue
        if not ok:
            raise PermissionError(
                f"Access denied: {resolved} is outside allowed directories "
                f"({', '.join(allowed_paths)})"
            )

    return resolved
