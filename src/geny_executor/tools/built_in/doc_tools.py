"""Doc tools — office-document engine backed by edit2docs.

2.43.0 — replaces host-side python-docx/openpyxl/python-pptx tool
stacks (e.g. Geny's old ``docx_edit`` / ``xlsx_edit`` / ``pptx_edit``)
with `edit2docs &lt;https://pypi.org/project/edit2docs/&gt;`_: an AI-agent-
native document engine whose preview, outline and edit operations
share one address system (``para`` / ``table,row,col`` for DOCX,
``sheet`` + A1 ``cell`` for XLSX, ``slide,shape_id,para`` for PPTX).

Install: ``pip install 'geny-executor[docs]'``. The engine imports
lazily; when missing, every tool returns a ToolResult error carrying
the install hint.

Five tools mirror edit2docs' five verbs:

* ``DocAnalyze``    → ``analyze_doc``  — outline + addresses (no LLM)
* ``DocApplyEdits`` → ``set_doc_text`` — deterministic edits (no LLM)
* ``DocPreview``    → ``preview_doc``  — markdown / SVG preview (no LLM)
* ``DocGenerate``   → ``generate_doc`` — create a document from intent (LLM)
* ``DocEdit``       → ``edit_doc``     — natural-language editing (LLM)

The deterministic loop the model should prefer: DocAnalyze → pick
addresses → DocApplyEdits (statuses ``applied | stale | not_found |
invalid`` per edit — self-correct and retry the failed ones). The LLM
verbs read their Anthropic API key from ``ctx.extras['docs']
['api_key']``, the ``ANTHROPIC_API_KEY`` env var (ToolContext env or
process), in that order; ``ctx.extras['docs']['model']`` overrides the
engine's default model.

File access follows the executor's standard path guard: relative paths
resolve against ``ToolContext.working_dir`` and everything must stay
inside ``allowed_paths`` when the host sets them.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate

_INSTALL_HINT = (
    "The edit2docs engine is not installed. Install it with: "
    "pip install 'geny-executor[docs]'. Host operators: add "
    "'edit2docs>=0.4.0' to the deployment image."
)

_SUPPORTED_EXTS = (".docx", ".xlsx", ".pptx")

_EDIT_ADDRESSING_DOC = (
    "Edit objects are format-dispatched by the file extension. DOCX: "
    '{"action":"replace","para":i,"new_text":...} | {"action":"replace",'
    '"table":t,"row":r,"col":c,"new_text":...} | {"action":"insert_after",'
    '"para":i,"markdown":...} (para=-1 prepends) | {"action":"delete","para":i}. '
    'XLSX: {"action":"set_cell","sheet":name,"cell":"B3","value":...} | '
    '{"action":"append_rows","sheet":name,"rows":[[...]]} | {"action":'
    '"add_sheet","sheet":name,"headers":[...],"rows":[[...]]}. PPTX: '
    '{"slide":i,"shape_id":id,"para":p,"new_text":...} (table cells add '
    '"row"/"col"). Optional "old_text"/"old_value" guards reject stale edits. '
    "Get addresses from DocAnalyze first."
)


def _load_edit2docs():
    """Import edit2docs lazily. Raises RuntimeError with an install hint."""
    try:
        import edit2docs  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(_INSTALL_HINT) from exc
    return edit2docs


def _resolve_doc_path(path: str, context: ToolContext, *, must_exist: bool = True) -> Path:
    resolved = resolve_and_validate(
        path, context.working_dir or os.getcwd(), context.allowed_paths
    )
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"No such file: {resolved}")
    if must_exist and resolved.suffix.lower() not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported document format {resolved.suffix!r} — "
            f"supported: {', '.join(_SUPPORTED_EXTS)}"
        )
    return resolved


def _docs_settings(context: ToolContext) -> Dict[str, Any]:
    extras = getattr(context, "extras", None) or {}
    settings = extras.get("docs")
    return settings if isinstance(settings, dict) else {}

def _api_key(context: ToolContext) -> Optional[str]:
    settings = _docs_settings(context)
    key = settings.get("api_key")
    if key:
        return str(key)
    env = getattr(context, "env_vars", None) or {}
    return env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def _llm_kwargs(context: ToolContext) -> Dict[str, Any]:
    """api_key/model kwargs for the LLM verbs; raises without a key."""
    key = _api_key(context)
    if not key:
        raise RuntimeError(
            "No Anthropic API key for the document engine — set "
            "ctx.extras['docs']['api_key'] (host tool settings) or the "
            "ANTHROPIC_API_KEY environment variable. For deterministic "
            "edits without an LLM, use DocAnalyze + DocApplyEdits."
        )
    kwargs: Dict[str, Any] = {"api_key": key}
    model = _docs_settings(context).get("model")
    if model:
        kwargs["model"] = str(model)
    return kwargs


class _DocToolBase(Tool):
    """Shared plumbing: path guard + engine/install error handling."""

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            return await self._run(input, context)
        except (RuntimeError, FileNotFoundError, PermissionError, ValueError) as exc:
            return ToolResult(content=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 — engine faults become tool errors
            return ToolResult(
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
                metadata={"error_type": type(exc).__name__},
            )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raise NotImplementedError


class DocAnalyzeTool(_DocToolBase):
    """Outline a document — the address source for DocApplyEdits."""

    @property
    def name(self) -> str:
        return "DocAnalyze"

    @property
    def description(self) -> str:
        return (
            "Analyze a .docx/.xlsx/.pptx file and return its addressable "
            "outline: DOCX paragraphs ({para, style, text}) and table cells "
            "({table, row, col, text}); XLSX sheets with sizes and sample "
            "rows; PPTX slides with text shapes ({shape_id, para, text}). "
            "Use these addresses in DocApplyEdits. Deterministic — no LLM."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
            max_result_chars=60_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        info = await asyncio.to_thread(engine.analyze_doc, str(path))
        return ToolResult(
            content=json.dumps(info, ensure_ascii=False, indent=1, default=str),
            metadata={"path": str(path), "format": info.get("format")},
        )


class DocApplyEditsTool(_DocToolBase):
    """Apply deterministic, address-based text edits."""

    @property
    def name(self) -> str:
        return "DocApplyEdits"

    @property
    def description(self) -> str:
        return (
            "Apply precise text edits to a .docx/.xlsx/.pptx file at "
            "addresses from DocAnalyze. Deterministic — no LLM. Each edit "
            "reports status applied | stale | not_found | invalid with a "
            "reason; fix and resend only the failed ones. "
            + _EDIT_ADDRESSING_DOC
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "edits": {
                    "type": "array",
                    "items": {"type": "object"},
                    "minItems": 1,
                    "description": "Edit objects (see tool description for shapes).",
                },
                "output": {
                    "type": "string",
                    "description": (
                        "Output path. Default: edit in place (same path). "
                    ),
                },
            },
            "required": ["path", "edits"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=30_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        edits = input.get("edits") or []
        if not isinstance(edits, list) or not all(isinstance(e, dict) for e in edits):
            return ToolResult(content="edits must be a list of objects", is_error=True)
        # Default to in-place: the executor's file tools edit in place, and
        # hosts with draft conventions pass their own output path.
        out = input.get("output")
        output = (
            _resolve_doc_path(str(out), context, must_exist=False) if out else path
        )
        result = await asyncio.to_thread(
            engine.set_doc_text, str(path), edits, output=str(output)
        )
        results = list(getattr(result, "results", []) or [])
        failed = [r for r in results if r.get("status") != "applied"]
        summary = {
            "path": str(getattr(result, "path", output)),
            "applied": getattr(result, "applied", 0),
            "failed": len(failed),
            "results": results,
        }
        return ToolResult(
            content=json.dumps(summary, ensure_ascii=False, indent=1, default=str),
            # Partial application is normal engine feedback (stale guards
            # etc.) — surface it in content, not as a hard tool error.
            metadata={"applied": summary["applied"], "failed": len(failed)},
        )


class DocPreviewTool(_DocToolBase):
    """Deterministic human/LLM-readable preview."""

    @property
    def name(self) -> str:
        return "DocPreview"

    @property
    def description(self) -> str:
        return (
            "Render a readable preview of a .docx/.xlsx/.pptx file. DOCX/XLSX "
            "return markdown text; PPTX renders one SVG file per slide (to "
            "out_dir, default '<doc dir>/preview/') and returns the paths. "
            "Deterministic — no LLM."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "out_dir": {
                    "type": "string",
                    "description": (
                        "Directory for preview artifacts (PPTX SVGs / preview.md). "
                        "Omit for inline markdown (DOCX/XLSX) or the default "
                        "preview dir (PPTX)."
                    ),
                },
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=60_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        out_dir_in = input.get("out_dir")
        kwargs: Dict[str, Any] = {}
        if out_dir_in:
            kwargs["out_dir"] = str(
                resolve_and_validate(
                    str(out_dir_in), context.working_dir or os.getcwd(), context.allowed_paths
                )
            )
        elif path.suffix.lower() == ".pptx":
            # SVG strings inline would blow the context — always go to disk.
            kwargs["out_dir"] = str(path.parent / "preview")
        result = await asyncio.to_thread(engine.preview_doc, str(path), **kwargs)
        if isinstance(result, (list, tuple)):
            paths = [str(p) for p in result]
            return ToolResult(
                content="Preview written:\n" + "\n".join(paths),
                metadata={"paths": paths},
            )
        if isinstance(result, Path) or (
            isinstance(result, str) and kwargs.get("out_dir") and os.path.isfile(str(result))
        ):
            return ToolResult(
                content=f"Preview written: {result}", metadata={"paths": [str(result)]}
            )
        return ToolResult(content=str(result), metadata={"inline": True})


class DocGenerateTool(_DocToolBase):
    """Create a new document from an intent (LLM-backed)."""

    @property
    def name(self) -> str:
        return "DocGenerate"

    @property
    def description(self) -> str:
        return (
            "Generate a NEW .docx/.xlsx/.pptx document from a natural-"
            "language intent (the output extension picks the engine). "
            "Optional sources (files or URLs — pdf/docx/xlsx/pptx/html/epub/"
            "ipynb...) ground the content. PPTX generation can take minutes. "
            "Uses an LLM — requires an Anthropic API key (host docs settings "
            "or ANTHROPIC_API_KEY)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What to create, in natural language.",
                },
                "output": {
                    "type": "string",
                    "description": "Output path — .docx, .xlsx or .pptx.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Grounding sources: file paths or URLs.",
                },
                "lang": {
                    "type": "string",
                    "description": "Content language (default ko-KR).",
                },
            },
            "required": ["intent", "output"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            network_egress=True,
            max_result_chars=20_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        kwargs = _llm_kwargs(context)
        output = _resolve_doc_path(input.get("output") or "", context, must_exist=False)
        if output.suffix.lower() not in _SUPPORTED_EXTS:
            return ToolResult(
                content=f"output must end in one of {_SUPPORTED_EXTS}", is_error=True
            )
        sources = self._resolve_sources(input.get("sources"), context)
        if sources:
            kwargs["sources"] = sources
        if input.get("lang"):
            kwargs["lang"] = str(input["lang"])
        result = await engine.async_generate_doc(
            str(input.get("intent") or ""), output=str(output), **kwargs
        )
        payload = {
            "path": str(getattr(result, "path", output)),
            "page_count": getattr(result, "page_count", None),
            "warnings": list(getattr(result, "warnings", []) or []),
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            metadata=payload,
        )

    @staticmethod
    def _resolve_sources(raw: Any, context: ToolContext) -> List[str]:
        sources: List[str] = []
        for item in raw or []:
            s = str(item)
            if "://" in s:
                sources.append(s)  # URL — the engine ingests it directly
            else:
                sources.append(
                    str(
                        resolve_and_validate(
                            s, context.working_dir or os.getcwd(), context.allowed_paths
                        )
                    )
                )
        return sources


class DocEditTool(_DocToolBase):
    """Natural-language document editing (LLM-backed)."""

    @property
    def name(self) -> str:
        return "DocEdit"

    @property
    def description(self) -> str:
        return (
            "Edit a .docx/.xlsx/.pptx file with a natural-language "
            "instruction — the engine plans and applies the operations. "
            "Prefer DocAnalyze + DocApplyEdits for precise, deterministic "
            "changes; use this for broad or fuzzy edits. A pure question "
            "returns an answer without changing the file. Uses an LLM — "
            "requires an Anthropic API key (host docs settings or "
            "ANTHROPIC_API_KEY)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "instruction": {
                    "type": "string",
                    "description": "Natural-language edit instruction.",
                },
                "output": {
                    "type": "string",
                    "description": "Output path. Default: edit in place.",
                },
            },
            "required": ["path", "instruction"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            network_egress=True,
            max_result_chars=30_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        kwargs = _llm_kwargs(context)
        path = _resolve_doc_path(input.get("path") or "", context)
        out = input.get("output")
        output = _resolve_doc_path(str(out), context, must_exist=False) if out else path
        result = await engine.async_edit_doc(
            str(path),
            str(input.get("instruction") or ""),
            output=str(output),
            **kwargs,
        )
        payload = {
            "path": str(getattr(result, "path", output)),
            "changed": bool(getattr(result, "changed", False)),
            "reply": getattr(result, "reply", ""),
            "operations": list(getattr(result, "operations", []) or []),
            "warnings": list(getattr(result, "warnings", []) or []),
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            metadata={"path": payload["path"], "changed": payload["changed"]},
        )


DOC_TOOL_CLASSES: Dict[str, type] = {
    "DocAnalyze": DocAnalyzeTool,
    "DocApplyEdits": DocApplyEditsTool,
    "DocPreview": DocPreviewTool,
    "DocGenerate": DocGenerateTool,
    "DocEdit": DocEditTool,
}
