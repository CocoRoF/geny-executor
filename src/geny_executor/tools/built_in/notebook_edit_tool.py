"""NotebookEdit — replace, insert, or delete cells in a Jupyter notebook.

Cycle 20260424 executor uplift — Phase 3 Week 6.

A ``.ipynb`` file is just JSON — the notebook format spec defines a
schema with ``cells`` at the top. We edit it directly with stdlib
``json`` rather than pulling in ``nbformat`` as a runtime dependency.
That means a few spec niceties (auto-bumping ``nbformat_minor``,
canonical cell metadata) are out of scope; hosts that need strict
validation can round-trip through ``nbformat`` themselves.

Supported operations (passed in order):

* ``replace`` — overwrite a cell's ``source`` by index.
* ``insert`` — insert a new cell of ``cell_type`` (``code`` |
  ``markdown`` | ``raw``) at ``cell_index``.
* ``delete`` — remove the cell at ``cell_index``.

Each edit applies sequentially against the *running* cell list, so
authors describe their intent in document order (a delete at index 2
followed by a replace at index 2 now points at what used to be cell 3).

Safety:

* Path validated through the same path guard as ``Read``/``Write``.
* Refuses anything that isn't a ``.ipynb`` file.
* Parse errors surface as ``ToolResult(is_error=True)`` — the tool never
  overwrites a notebook it couldn't parse.
* Writes are atomic: serialise the modified notebook to a temp file in
  the same directory, then ``os.replace``. A crash mid-write leaves the
  original intact.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 6 Workflow).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate

_VALID_CELL_TYPES = ("code", "markdown", "raw")
_VALID_OPS = ("replace", "insert", "delete")


def _load_notebook(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ValueError("notebook JSON missing top-level 'cells' list")
    return nb


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via temp-file + rename.

    Keeps the original untouched if JSON serialisation or fsync fails.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".nbedit-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ensure_source_shape(src: Any) -> List[str]:
    """Notebook cells store ``source`` as either a string or a list of strings.

    ``json.load`` gives us whichever was on disk. To keep downstream
    edits predictable we normalise to a list of lines (trailing newlines
    preserved) — matches the style Jupyter writes out by default.
    """
    if isinstance(src, str):
        if not src:
            return []
        # Split preserving trailing newlines so re-serialisation doesn't
        # drop them — notebook authors expect the disk form to round trip.
        parts = src.splitlines(keepends=True)
        return parts or [src]
    if isinstance(src, list):
        return [s if isinstance(s, str) else str(s) for s in src]
    return [str(src)]


def _apply_replace(cells: List[Dict[str, Any]], op: Dict[str, Any]) -> None:
    idx = op["cell_index"]
    source = op.get("new_source")
    if source is None:
        raise ValueError(f"replace op at index {idx} missing 'new_source'")
    if not (0 <= idx < len(cells)):
        raise ValueError(f"replace: cell_index {idx} out of range (cells={len(cells)})")
    cell = cells[idx]
    cell["source"] = _ensure_source_shape(source)
    # Replacing a code cell's source invalidates its outputs — clear
    # them so stale results don't linger in the file.
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _apply_insert(cells: List[Dict[str, Any]], op: Dict[str, Any]) -> None:
    idx = op["cell_index"]
    cell_type = op.get("cell_type", "code")
    source = op.get("new_source", "")
    if cell_type not in _VALID_CELL_TYPES:
        raise ValueError(f"insert: cell_type {cell_type!r} not in {_VALID_CELL_TYPES}")
    if not (0 <= idx <= len(cells)):
        raise ValueError(f"insert: cell_index {idx} out of range (cells={len(cells)})")

    new_cell: Dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _ensure_source_shape(source),
    }
    if cell_type == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    cells.insert(idx, new_cell)


def _apply_delete(cells: List[Dict[str, Any]], op: Dict[str, Any]) -> None:
    idx = op["cell_index"]
    if not (0 <= idx < len(cells)):
        raise ValueError(f"delete: cell_index {idx} out of range (cells={len(cells)})")
    del cells[idx]


_OP_HANDLERS = {
    "replace": _apply_replace,
    "insert": _apply_insert,
    "delete": _apply_delete,
}


class NotebookEditTool(Tool):
    """Edit a Jupyter notebook by applying a list of cell operations.

    Operations apply in order against the evolving cell list, so callers
    describe their intent as a sequence of document-level diffs. The
    tool returns the new cell count + a short per-op trail so the LLM
    can confirm the effect before the next turn.
    """

    @property
    def name(self) -> str:
        return "NotebookEdit"

    @property
    def description(self) -> str:
        return (
            "Edit cells in a Jupyter (.ipynb) notebook. Supports "
            "'replace' (overwrite source), 'insert' (add a new code / "
            "markdown / raw cell), and 'delete'. Operations apply in "
            "order — an earlier delete shifts indices that come after."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb file.",
                },
                "operations": {
                    "type": "array",
                    "description": (
                        "List of edits to apply in order. Each entry has "
                        "'op' ('replace' | 'insert' | 'delete') and "
                        "'cell_index'. 'replace' needs 'new_source'. "
                        "'insert' accepts 'new_source' (default empty) "
                        "and 'cell_type' ('code' | 'markdown' | 'raw', "
                        "default 'code')."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": list(_VALID_OPS),
                            },
                            "cell_index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "new_source": {
                                "type": "string",
                                "description": "Source for replace / insert.",
                            },
                            "cell_type": {
                                "type": "string",
                                "enum": list(_VALID_CELL_TYPES),
                                "description": "Cell type for insert.",
                            },
                        },
                        "required": ["op", "cell_index"],
                    },
                    "minItems": 1,
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "If true (default), write the modified notebook "
                        "back to disk. If false, return the rendered "
                        "notebook in metadata without writing — useful "
                        "for dry-runs."
                    ),
                },
            },
            "required": ["file_path", "operations"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        save = bool(input.get("save", True))
        return ToolCapabilities(
            concurrency_safe=False,  # two edits on the same file would race
            read_only=not save,
            destructive=save,  # overwrites the notebook
            idempotent=False,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raw_path = input.get("file_path", "")
        save = bool(input.get("save", True))
        raw_ops = input.get("operations")

        if not isinstance(raw_ops, list) or not raw_ops:
            return ToolResult(
                content="'operations' must be a non-empty list",
                is_error=True,
            )

        try:
            resolved = resolve_and_validate(raw_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError) as exc:
            return ToolResult(content=str(exc), is_error=True)

        if not resolved.exists():
            return ToolResult(content=f"notebook not found: {resolved}", is_error=True)
        if resolved.is_dir():
            return ToolResult(content=f"not a file: {resolved}", is_error=True)
        if resolved.suffix.lower() != ".ipynb":
            return ToolResult(
                content=f"NotebookEdit only edits .ipynb files (got {resolved.suffix!r})",
                is_error=True,
            )

        try:
            notebook = _load_notebook(str(resolved))
        except json.JSONDecodeError as exc:
            return ToolResult(content=f"notebook JSON parse error: {exc}", is_error=True)
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True)
        except OSError as exc:
            return ToolResult(content=f"read error: {exc}", is_error=True)

        cells: List[Dict[str, Any]] = notebook["cells"]
        before_count = len(cells)
        trail: List[str] = []

        for op_idx, raw_op in enumerate(raw_ops):
            if not isinstance(raw_op, dict):
                return ToolResult(
                    content=f"op #{op_idx} must be an object",
                    is_error=True,
                )
            op_name = raw_op.get("op")
            if op_name not in _OP_HANDLERS:
                return ToolResult(
                    content=f"op #{op_idx}: unknown op {op_name!r}; expected one of {_VALID_OPS}",
                    is_error=True,
                )
            idx = raw_op.get("cell_index")
            if not isinstance(idx, int) or idx < 0:
                return ToolResult(
                    content=f"op #{op_idx}: 'cell_index' must be a non-negative integer",
                    is_error=True,
                )

            try:
                _OP_HANDLERS[op_name](cells, raw_op)
            except ValueError as exc:
                return ToolResult(content=f"op #{op_idx}: {exc}", is_error=True)

            trail.append(f"#{op_idx} {op_name}@{idx} (cells now {len(cells)})")

        after_count = len(cells)

        if save:
            try:
                _atomic_write(str(resolved), notebook)
            except OSError as exc:
                return ToolResult(content=f"write error: {exc}", is_error=True)

        summary = (
            f"NotebookEdit: {resolved}\n"
            f"  cells {before_count} → {after_count} "
            f"({len(raw_ops)} operation{'s' if len(raw_ops) != 1 else ''})\n"
            f"  {('saved' if save else 'not saved (save=false)')}\n"
            f"  trail:\n    " + "\n    ".join(trail)
        )
        return ToolResult(
            content=summary,
            metadata={
                "path": str(resolved),
                "before_cells": before_count,
                "after_cells": after_count,
                "saved": save,
                "operations_applied": len(raw_ops),
                # Don't persist notebook body into LLM context — if a host
                # needs it, reading the file back is cheap.
            },
        )
