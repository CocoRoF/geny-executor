"""Dev-environment tools — LSP / REPL / Brief (PR-A.3.5)."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


# ── LSPTool ──────────────────────────────────────────────────────────


class LSPTool(Tool):
    """Query a language server. Adapters are host-supplied via
    ``ctx.extras["lsp_adapters"]`` — a dict of language → adapter.

    Each adapter is an async callable::

        async def adapter(*, action, file, line, col, cwd) -> dict
    """

    @property
    def name(self) -> str:
        return "LSP"

    @property
    def description(self) -> str:
        return (
            "Query a language server for diagnostics / hover / definition / "
            "references. Hosts wire language adapters via ctx.extras['lsp_adapters']."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "action": {"enum": ["diagnostics", "hover", "definition", "references"]},
                "file": {"type": "string"},
                "line": {"type": "integer", "minimum": 0, "default": 0},
                "col": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["language", "action", "file"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input, context):
        adapters = context.extras.get("lsp_adapters") or {}
        adapter = adapters.get(input["language"])
        if adapter is None:
            return _err("NO_ADAPTER", f"no LSP adapter for language: {input['language']}")
        try:
            result = adapter(
                action=input["action"],
                file=input["file"],
                line=input.get("line", 0),
                col=input.get("col", 0),
                cwd=context.working_dir or ".",
            )
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:  # noqa: BLE001
            return _err("LSP_FAILED", str(exc))
        return ToolResult(content={
            "language": input["language"],
            "action": input["action"],
            "result": result,
        })


# ── REPLTool ─────────────────────────────────────────────────────────


class REPLTool(Tool):
    """Run a Python expression in a sandboxed subprocess."""

    @property
    def name(self) -> str:
        return "REPL"

    @property
    def description(self) -> str:
        return (
            "Execute a Python expression in a fresh subprocess. "
            "Returns stdout / stderr / exit_code. Bounded by timeout."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "maxLength": 8000},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60, "default": 5},
            },
            "required": ["expression"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, destructive=False)

    async def execute(self, input, context):
        expr = input.get("expression", "")
        if not expr:
            return _err("BAD_INPUT", "expression is required")
        timeout = int(input.get("timeout_seconds", 5))
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", expr,
                cwd=context.working_dir or ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return _err("SPAWN_FAILED", str(exc))
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _err("REPL_TIMEOUT", f"expression exceeded {timeout}s")
        return ToolResult(content={
            "stdout": stdout.decode(errors="replace")[:64_000],
            "stderr": stderr.decode(errors="replace")[:8_000],
            "exit_code": proc.returncode or 0,
        })


# ── BriefTool ────────────────────────────────────────────────────────


class BriefTool(Tool):
    """Manually trigger context summarization (Stage 19)."""

    @property
    def name(self) -> str:
        return "Brief"

    @property
    def description(self) -> str:
        return "Manually trigger context summarization. Returns a summary blob."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {"enum": ["all", "since_last_brief"], "default": "since_last_brief"},
            },
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False)

    async def execute(self, input, context):
        summarizer = context.extras.get("summarize_strategy")
        if summarizer is None:
            return _err("NO_SUMMARIZER", "summarize_strategy not wired into ctx.extras")
        scope = input.get("scope", "since_last_brief")
        # Try canonical method names.
        for method in ("summarize_now", "compact", "run"):
            fn = getattr(summarizer, method, None)
            if callable(fn):
                try:
                    result = fn(scope=scope) if _accepts_kw(fn, "scope") else fn()
                    if hasattr(result, "__await__"):
                        result = await result
                except Exception as exc:  # noqa: BLE001
                    return _err("SUMMARIZE_FAILED", str(exc))
                return ToolResult(content={
                    "scope": scope,
                    "summary": getattr(result, "summary", None) or str(result or ""),
                    "tokens_compressed": getattr(result, "tokens_compressed", None),
                })
        return _err("SUMMARIZE_API", "summarize_strategy has no summarize_now/compact/run")


def _accepts_kw(fn, name: str) -> bool:
    try:
        import inspect
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


__all__ = ["LSPTool", "REPLTool", "BriefTool"]
