"""Dev tools tests — LSP / REPL / Brief (PR-A.3.5)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    BriefTool,
    LSPTool,
    REPLTool,
)


def test_all_three_registered():
    for name in ("LSP", "REPL", "Brief"):
        assert name in BUILT_IN_TOOL_CLASSES


# ── LSPTool ──────────────────────────────────────────────────────────


class TestLSP:
    @pytest.mark.asyncio
    async def test_calls_adapter(self):
        called: Dict[str, Any] = {}

        async def py_adapter(*, action, file, line, col, cwd):
            called.update({"action": action, "file": file, "line": line, "col": col})
            return {"diagnostics": []}

        ctx = ToolContext(extras={"lsp_adapters": {"python": py_adapter}})
        result = await LSPTool().execute(
            {"language": "python", "action": "diagnostics", "file": "/x.py"}, ctx,
        )
        assert result.is_error is False
        assert called["action"] == "diagnostics"
        assert result.content["result"] == {"diagnostics": []}

    @pytest.mark.asyncio
    async def test_no_adapter(self):
        ctx = ToolContext(extras={"lsp_adapters": {}})
        result = await LSPTool().execute(
            {"language": "rust", "action": "diagnostics", "file": "x.rs"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_ADAPTER"

    @pytest.mark.asyncio
    async def test_adapter_failure(self):
        async def adapter(**kwargs):
            raise RuntimeError("server down")
        ctx = ToolContext(extras={"lsp_adapters": {"python": adapter}})
        result = await LSPTool().execute(
            {"language": "python", "action": "hover", "file": "x.py"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "LSP_FAILED"

    @pytest.mark.asyncio
    async def test_no_adapters_dict(self):
        ctx = ToolContext(extras={})
        result = await LSPTool().execute(
            {"language": "python", "action": "hover", "file": "x.py"}, ctx,
        )
        assert result.is_error is True


# ── REPLTool ─────────────────────────────────────────────────────────


class TestREPL:
    @pytest.mark.asyncio
    async def test_runs_expression(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await REPLTool().execute({"expression": "print(2 + 3)"}, ctx)
        assert result.is_error is False
        assert "5" in result.content["stdout"]
        assert result.content["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_captures_stderr(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await REPLTool().execute(
            {"expression": "import sys; print('e', file=sys.stderr)"}, ctx,
        )
        assert "e" in result.content["stderr"]

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await REPLTool().execute(
            {"expression": "import sys; sys.exit(7)"}, ctx,
        )
        assert result.content["exit_code"] == 7

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await REPLTool().execute(
            {"expression": "import time; time.sleep(10)", "timeout_seconds": 1}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "REPL_TIMEOUT"

    @pytest.mark.asyncio
    async def test_empty_expression(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await REPLTool().execute({"expression": ""}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "BAD_INPUT"


# ── BriefTool ────────────────────────────────────────────────────────


class _FakeSummarizer:
    def __init__(self, *, raise_exc=None):
        self.raise_exc = raise_exc
        self.last_scope = None

    async def summarize_now(self, *, scope):
        self.last_scope = scope
        if self.raise_exc is not None:
            raise self.raise_exc
        return type("R", (), {"summary": "compact result", "tokens_compressed": 1234})()


class TestBrief:
    @pytest.mark.asyncio
    async def test_invokes_summarizer(self):
        summarizer = _FakeSummarizer()
        ctx = ToolContext(extras={"summarize_strategy": summarizer})
        result = await BriefTool().execute({"scope": "all"}, ctx)
        assert result.is_error is False
        assert result.content["scope"] == "all"
        assert result.content["tokens_compressed"] == 1234
        assert summarizer.last_scope == "all"

    @pytest.mark.asyncio
    async def test_no_summarizer(self):
        ctx = ToolContext(extras={})
        result = await BriefTool().execute({}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_SUMMARIZER"

    @pytest.mark.asyncio
    async def test_failure(self):
        summarizer = _FakeSummarizer(raise_exc=RuntimeError("oom"))
        ctx = ToolContext(extras={"summarize_strategy": summarizer})
        result = await BriefTool().execute({}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "SUMMARIZE_FAILED"

    @pytest.mark.asyncio
    async def test_default_scope(self):
        summarizer = _FakeSummarizer()
        ctx = ToolContext(extras={"summarize_strategy": summarizer})
        await BriefTool().execute({}, ctx)
        assert summarizer.last_scope == "since_last_brief"
