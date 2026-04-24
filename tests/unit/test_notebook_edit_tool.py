"""Phase 3 Week 6 — NotebookEdit tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.notebook_edit_tool import NotebookEditTool


def _minimal_notebook(cells: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {
        "cells": cells or [],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code_cell(source: Any = "") -> Dict[str, Any]:
    if isinstance(source, str):
        source_list = source.splitlines(keepends=True) or ([source] if source else [])
    else:
        source_list = list(source)
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_list,
        "outputs": [],
        "execution_count": None,
    }


def _md_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _write(tmp_path, nb: Dict[str, Any], name: str = "book.ipynb") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return str(p)


def _read(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="")


class TestCapabilitiesAndSchema:
    def test_destructive_when_saving(self):
        caps = NotebookEditTool().capabilities({"save": True})
        assert caps.concurrency_safe is False
        assert caps.destructive is True
        assert caps.read_only is False

    def test_read_only_when_not_saving(self):
        caps = NotebookEditTool().capabilities({"save": False})
        assert caps.destructive is False
        assert caps.read_only is True


class TestReplaceOperation:
    @pytest.mark.asyncio
    async def test_replace_code_cell_source(self, tmp_path):
        nb = _minimal_notebook([_code_cell("x = 1\n"), _code_cell("y = 2\n")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [
                    {"op": "replace", "cell_index": 0, "new_source": "x = 100\n"}
                ],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        assert "".join(disk["cells"][0]["source"]) == "x = 100\n"
        # Outputs cleared on source change
        assert disk["cells"][0]["outputs"] == []
        assert disk["cells"][0]["execution_count"] is None

    @pytest.mark.asyncio
    async def test_replace_out_of_range(self, tmp_path):
        nb = _minimal_notebook([_code_cell("x")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [
                    {"op": "replace", "cell_index": 5, "new_source": "nope"}
                ],
            },
            _ctx(),
        )
        assert result.is_error
        assert "out of range" in result.content

    @pytest.mark.asyncio
    async def test_replace_missing_new_source(self, tmp_path):
        nb = _minimal_notebook([_code_cell("x")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {"file_path": path, "operations": [{"op": "replace", "cell_index": 0}]},
            _ctx(),
        )
        assert result.is_error
        assert "new_source" in result.content


class TestInsertOperation:
    @pytest.mark.asyncio
    async def test_insert_code_at_head(self, tmp_path):
        nb = _minimal_notebook([_code_cell("x")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [
                    {
                        "op": "insert",
                        "cell_index": 0,
                        "cell_type": "markdown",
                        "new_source": "# Title",
                    }
                ],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        assert len(disk["cells"]) == 2
        assert disk["cells"][0]["cell_type"] == "markdown"
        assert "".join(disk["cells"][0]["source"]) == "# Title"

    @pytest.mark.asyncio
    async def test_insert_default_code_cell(self, tmp_path):
        nb = _minimal_notebook([])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "pass"}],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        assert len(disk["cells"]) == 1
        cell = disk["cells"][0]
        assert cell["cell_type"] == "code"
        assert cell["execution_count"] is None
        assert cell["outputs"] == []

    @pytest.mark.asyncio
    async def test_insert_at_tail(self, tmp_path):
        nb = _minimal_notebook([_code_cell("a"), _code_cell("b")])
        path = _write(tmp_path, nb)
        # Inserting at len(cells) == end is allowed
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "insert", "cell_index": 2, "new_source": "c"}],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        assert len(disk["cells"]) == 3

    @pytest.mark.asyncio
    async def test_insert_invalid_cell_type(self, tmp_path):
        nb = _minimal_notebook([])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [
                    {
                        "op": "insert",
                        "cell_index": 0,
                        "cell_type": "weird",
                        "new_source": "x",
                    }
                ],
            },
            _ctx(),
        )
        assert result.is_error
        assert "cell_type" in result.content


class TestDeleteOperation:
    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        nb = _minimal_notebook([_code_cell("a"), _code_cell("b"), _code_cell("c")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "delete", "cell_index": 1}],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        sources = ["".join(c["source"]) for c in disk["cells"]]
        assert sources == ["a", "c"]

    @pytest.mark.asyncio
    async def test_delete_out_of_range(self, tmp_path):
        nb = _minimal_notebook([_code_cell("a")])
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "delete", "cell_index": 10}],
            },
            _ctx(),
        )
        assert result.is_error


class TestSequentialOps:
    @pytest.mark.asyncio
    async def test_delete_then_replace_applies_to_shifted_index(self, tmp_path):
        nb = _minimal_notebook(
            [_code_cell("a"), _code_cell("b"), _code_cell("c"), _code_cell("d")]
        )
        path = _write(tmp_path, nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [
                    # Delete cell 1 (b) → remaining: [a, c, d]
                    {"op": "delete", "cell_index": 1},
                    # Replace cell 1 (which is now 'c') → [a, C!, d]
                    {"op": "replace", "cell_index": 1, "new_source": "C!"},
                ],
            },
            _ctx(),
        )
        assert not result.is_error
        disk = _read(path)
        sources = ["".join(c["source"]) for c in disk["cells"]]
        assert sources == ["a", "C!", "d"]


class TestDryRun:
    @pytest.mark.asyncio
    async def test_save_false_does_not_write(self, tmp_path):
        original_nb = _minimal_notebook([_code_cell("x")])
        path = _write(tmp_path, original_nb)
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "save": False,
                "operations": [
                    {"op": "replace", "cell_index": 0, "new_source": "overridden"}
                ],
            },
            _ctx(),
        )
        assert not result.is_error
        # Disk unchanged
        disk = _read(path)
        assert "".join(disk["cells"][0]["source"]) == "x"
        assert result.metadata["saved"] is False
        assert "not saved" in result.content


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_missing_operations(self, tmp_path):
        path = _write(tmp_path, _minimal_notebook())
        result = await NotebookEditTool().execute(
            {"file_path": path, "operations": []}, _ctx()
        )
        assert result.is_error
        assert "operations" in result.content

    @pytest.mark.asyncio
    async def test_non_ipynb_rejected(self, tmp_path):
        p = tmp_path / "not_notebook.txt"
        p.write_text("plain")
        result = await NotebookEditTool().execute(
            {
                "file_path": str(p),
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "x"}],
            },
            _ctx(),
        )
        assert result.is_error
        assert ".ipynb" in result.content

    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path):
        result = await NotebookEditTool().execute(
            {
                "file_path": str(tmp_path / "ghost.ipynb"),
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "x"}],
            },
            _ctx(),
        )
        assert result.is_error
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_directory_rejected(self, tmp_path):
        result = await NotebookEditTool().execute(
            {
                "file_path": str(tmp_path),
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "x"}],
            },
            _ctx(),
        )
        assert result.is_error

    @pytest.mark.asyncio
    async def test_malformed_json_not_overwritten(self, tmp_path):
        p = tmp_path / "broken.ipynb"
        p.write_text("{not json")
        result = await NotebookEditTool().execute(
            {
                "file_path": str(p),
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "x"}],
            },
            _ctx(),
        )
        assert result.is_error
        assert "JSON parse error" in result.content
        # File content untouched
        assert p.read_text() == "{not json"

    @pytest.mark.asyncio
    async def test_missing_cells_rejected(self, tmp_path):
        p = tmp_path / "noneb.ipynb"
        p.write_text(json.dumps({"metadata": {}}))
        result = await NotebookEditTool().execute(
            {
                "file_path": str(p),
                "operations": [{"op": "insert", "cell_index": 0, "new_source": "x"}],
            },
            _ctx(),
        )
        assert result.is_error
        assert "cells" in result.content

    @pytest.mark.asyncio
    async def test_unknown_op_rejected(self, tmp_path):
        path = _write(tmp_path, _minimal_notebook([_code_cell("x")]))
        result = await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "mutate", "cell_index": 0}],
            },
            _ctx(),
        )
        assert result.is_error
        assert "unknown op" in result.content


class TestAtomicity:
    @pytest.mark.asyncio
    async def test_temp_file_cleaned_up(self, tmp_path):
        """After a successful write, no stray .nbedit-* temp files should remain."""
        nb = _minimal_notebook([_code_cell("x")])
        path = _write(tmp_path, nb)
        await NotebookEditTool().execute(
            {
                "file_path": path,
                "operations": [{"op": "replace", "cell_index": 0, "new_source": "y"}],
            },
            _ctx(),
        )
        leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".nbedit-")]
        assert leftovers == []


class TestRegistry:
    def test_registered(self):
        from geny_executor.tools.built_in import (
            BUILT_IN_TOOL_CLASSES,
            BUILT_IN_TOOL_FEATURES,
            NotebookEditTool,
        )

        assert BUILT_IN_TOOL_CLASSES["NotebookEdit"] is NotebookEditTool
        assert "NotebookEdit" in BUILT_IN_TOOL_FEATURES["filesystem"]
