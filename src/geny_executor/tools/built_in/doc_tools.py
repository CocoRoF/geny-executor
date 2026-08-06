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

The family is a hierarchical skill with progressive disclosure: every
description is compact (frontmatter tier); ``DocGuide`` returns the
GENERATE|EDIT|INSPECT family map (body) and deep per-task guides by topic
(resources — build, generate, edit, edit.text, edit.chart, edit.xml, render,
recipes.slides, recipes.colors), rendered with the executor tool names via
``_GUIDE_NAME_MAP``. Seven always-on tools + two key-gated LLM conveniences:

* ``DocGuide``      → ``doc_guide``    — the skill map + topic guides (no LLM)
* ``DocAnalyze``    → ``analyze_doc``  — outline + addresses (no LLM)
* ``DocApplyEdits`` → ``set_doc_text`` + ``edit_chart`` — structured text
  AND chart edits, one surface (no LLM)
* ``DocBuild``      → ``build_doc``    — build from a spec (no LLM)
* ``DocXmlRead``    → ``list_doc_parts`` / ``get_doc_xml`` — the package's
  raw XML (no LLM)
* ``DocXmlEdit``    → ``set_doc_xml``  — patch/create/delete a part (no LLM)
* ``DocRender``     → ``render_doc``   — page PNG/PDF/SVG or readable
  ``md`` content (no LLM)
* ``DocGenerate``   → ``generate_doc`` — create from intent (LLM; gated on
  ``feature:docs_llm``)
* ``DocEdit``       → ``edit_doc``     — natural-language editing (LLM;
  gated on ``feature:docs_llm``)

The deterministic loop the model should prefer: DocAnalyze → pick
addresses → DocApplyEdits (text + charts); for anything the structured
verbs don't cover (colors, fills, fonts, geometry, add/remove slides —
documents ARE zipped XML) → DocXmlRead → DocXmlEdit. Office files should
never need python-pptx/python-docx in a REPL. The LLM verbs advertise
``required_config_keys() -> ["feature:docs_llm"]`` so hosts without an
Anthropic key never register them (no dead tools). They read their key
from ``ctx.extras['docs']['api_key']``, the ``ANTHROPIC_API_KEY`` env var
(ToolContext env or process), in that order; ``ctx.extras['docs']
['model']`` overrides the engine's default model.

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

# The LLM verbs (DocGenerate / DocEdit) are hidden unless the host marks
# this feature satisfied (i.e. an Anthropic key is actually available) —
# a tool that can only error must never reach the model.
_DOCS_LLM_FEATURE_KEY = "feature:docs_llm"



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
            "ANTHROPIC_API_KEY environment variable. Every edit is possible "
            "WITHOUT an LLM key: DocAnalyze then DocApplyEdits (text + "
            "chart edits), DocXmlRead + DocXmlEdit (any other edit — "
            "colors, fonts, formatting, add/remove slides), or DocBuild "
            "(new document from your spec)."
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


# Canonical edit2docs verb names → the executor tool names the model sees.
# doc_guide renders every guide with these, so recipes reference REAL tools.
_GUIDE_NAME_MAP = {
    "doc_guide": "DocGuide",
    "analyze_doc": "DocAnalyze",
    "render_doc": "DocRender",
    "set_doc_text": "DocApplyEdits",
    "arrange_doc": "DocArrange",
    "read_doc_xml": "DocXmlRead",
    "set_doc_xml": "DocXmlEdit",
    "build_doc": "DocBuild",
    "generate_doc": "DocGenerate",
    "edit_doc": "DocEdit",
}


class DocGuideTool(_DocToolBase):
    """The skill entry point — hierarchical guide, progressive disclosure."""

    @property
    def name(self) -> str:
        return "DocGuide"

    @property
    def description(self) -> str:
        return (
            "START HERE for .docx/.xlsx/.pptx work — the document skill. "
            "No topic: the GENERATE|EDIT|INSPECT map. topic: deep guide "
            "(build, generate, edit, edit.text, edit.chart, edit.xml, "
            "render, recipes.slides, recipes.colors). Free, instant."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic or prefix (e.g. 'recipes').",
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
            max_result_chars=30_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        guide_fn = getattr(engine, "doc_guide", None)
        if guide_fn is None:  # pragma: no cover — edit2docs < 0.13
            return ToolResult(
                content=(
                    "This edit2docs version has no doc_guide — upgrade to "
                    "edit2docs>=0.13.0 (pip install 'geny-executor[docs]')."
                ),
                is_error=True,
            )
        res = guide_fn(input.get("topic"), names=_GUIDE_NAME_MAP)
        return ToolResult(
            content=res["guide"],
            metadata={"topic": res.get("topic", ""), "topics": res.get("topics", [])},
        )


class DocAnalyzeTool(_DocToolBase):
    """Outline a document — the address source for DocApplyEdits."""

    @property
    def name(self) -> str:
        return "DocAnalyze"

    @property
    def description(self) -> str:
        return (
            "Outline + edit addresses + charts list for a .docx/.xlsx/.pptx. "
            "Deterministic, no key. Run FIRST before any edit. "
            "Guide: DocGuide('edit')."
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
            "Deterministic structured edits at DocAnalyze addresses — "
            "text/table/cell values AND chart title/data ({chart: i, ...}). "
            "No key; byte-preserves the rest; per-edit statuses. Shapes: "
            "DocGuide('edit.text'), DocGuide('edit.chart')."
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
        # One structured-edit surface: dicts with a `chart` key go to the
        # chart engine, the rest to the text engine, chained on one output.
        text_edits = [e for e in edits if "chart" not in e]
        chart_edits = [e for e in edits if "chart" in e]
        applied, results = 0, []
        src = str(path)
        if text_edits:
            result = await asyncio.to_thread(
                engine.set_doc_text, src, text_edits, output=str(output)
            )
            applied += getattr(result, "applied", 0)
            results.extend(list(getattr(result, "results", []) or []))
            src = str(getattr(result, "path", output))
        if chart_edits:
            result = await asyncio.to_thread(
                engine.edit_chart, src, chart_edits, output=str(output)
            )
            applied += getattr(result, "applied", 0)
            results.extend(list(getattr(result, "results", []) or []))
        failed = [r for r in results if r.get("status") != "applied"]
        summary = {
            "path": str(output),
            "applied": applied,
            "failed": len(failed),
            "results": results,
        }
        return ToolResult(
            content=json.dumps(summary, ensure_ascii=False, indent=1, default=str),
            # Partial application is normal engine feedback (stale guards
            # etc.) — surface it in content, not as a hard tool error.
            metadata={"applied": summary["applied"], "failed": len(failed)},
        )


class DocArrangeTool(_DocToolBase):
    """Deterministic STRUCTURAL edits — whole slides / sheets as objects."""

    @property
    def name(self) -> str:
        return "DocArrange"

    @property
    def description(self) -> str:
        return (
            "Deterministic STRUCTURAL edits: duplicate / move / delete whole "
            "slides (.pptx) or sheets (.xlsx); rename sheets. No key; byte-"
            "preserving. ops apply in order — target = slide index or sheet "
            "name/index, to = position. Guide: DocGuide('arrange')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "ops": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Structural ops, e.g. {op:'duplicate',target:0,to:3} or "
                        "{op:'rename',target:'Sheet1',name:'Summary'}. Shapes: "
                        "DocGuide('arrange')."
                    ),
                },
                "output": {
                    "type": "string",
                    "description": "Output path (default: in place).",
                },
            },
            "required": ["path", "ops"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=30_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        ops = input.get("ops") or []
        if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
            return ToolResult(content="ops must be a list of objects", is_error=True)
        out = input.get("output")
        output = (
            _resolve_doc_path(str(out), context, must_exist=False) if out else path
        )
        result = await asyncio.to_thread(
            engine.arrange_doc, str(path), ops, output=str(output)
        )
        results = list(getattr(result, "results", []) or [])
        failed = [r for r in results if r.get("status") != "applied"]
        summary = {
            "path": str(getattr(result, "path", output)),
            "applied": getattr(result, "applied", 0),
            "failed": len(failed),
            "results": results,
            "warnings": list(getattr(result, "warnings", []) or []),
        }
        return ToolResult(
            content=json.dumps(summary, ensure_ascii=False, indent=1, default=str),
            metadata={"applied": summary["applied"], "failed": len(failed)},
        )


class DocXmlReadTool(_DocToolBase):
    """Read the package's raw XML — parts map or one part's text."""

    @property
    def name(self) -> str:
        return "DocXmlRead"

    @property
    def description(self) -> str:
        return (
            "Documents are zips of XML. No part: the part map. With part: "
            "that part's exact XML text. Pair with DocXmlEdit for ANY edit "
            "(colors, fonts, slides). Guide: DocGuide('edit.xml')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "part": {
                    "type": "string",
                    "description": "Part name to read. Omit to list all parts.",
                },
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Slide/chart XML parts run large; give reads generous headroom.
        return ToolCapabilities(concurrency_safe=True, max_result_chars=200_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        part = input.get("part")
        if not part:
            parts = await asyncio.to_thread(engine.list_doc_parts, str(path))
            return ToolResult(
                content=json.dumps({"parts": parts}, ensure_ascii=False, indent=1),
                metadata={"path": str(path), "count": len(parts)},
            )
        xml = await asyncio.to_thread(engine.get_doc_xml, str(path), str(part))
        return ToolResult(
            content=xml,
            metadata={"path": str(path), "part": str(part), "chars": len(xml)},
        )


class DocXmlEditTool(_DocToolBase):
    """Patch one XML part — the universal deterministic escape hatch."""

    @property
    def name(self) -> str:
        return "DocXmlEdit"

    @property
    def description(self) -> str:
        return (
            "Patch (find/replace), CREATE (xml + content_type) or DELETE one "
            "XML part. Well-formed-or-nothing; byte-preserving. The universal "
            "edit — recolor, fonts, add/remove slides. Recipes: "
            "DocGuide('recipes.slides'), DocGuide('recipes.colors')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "part": {
                    "type": "string",
                    "description": "Part to patch, e.g. ppt/charts/chart1.xml.",
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["find", "replace"],
                    },
                    "description": "Exact-substring edits (use OR `xml`).",
                },
                "xml": {
                    "type": "string",
                    "description": (
                        "Full part XML — replaces, or CREATES a missing part."
                    ),
                },
                "content_type": {
                    "type": "string",
                    "description": (
                        "[Content_Types].xml Override for a newly created "
                        "part, e.g. application/vnd.openxmlformats-"
                        "officedocument.presentationml.slide+xml"
                    ),
                },
                "delete": {
                    "type": "boolean",
                    "description": "Remove the part (also patch referencing rels).",
                },
                "output": {
                    "type": "string",
                    "description": "Output path. Default: edit in place (same path).",
                },
            },
            "required": ["path", "part"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=30_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        path = _resolve_doc_path(input.get("path") or "", context)
        part = str(input.get("part") or "")
        edits = input.get("edits")
        xml = input.get("xml")
        delete = bool(input.get("delete"))
        modes = sum(1 for m in (edits is not None, xml is not None, delete) if m)
        if modes != 1:
            return ToolResult(
                content="pass exactly one of `edits`, `xml` or `delete: true`",
                is_error=True,
            )
        if edits is not None and (
            not isinstance(edits, list) or not all(isinstance(e, dict) for e in edits)
        ):
            return ToolResult(content="edits must be a list of objects", is_error=True)
        out = input.get("output")
        output = (
            _resolve_doc_path(str(out), context, must_exist=False) if out else path
        )
        result = await asyncio.to_thread(
            engine.set_doc_xml,
            str(path),
            part,
            edits,
            xml=xml,
            content_type=(str(input["content_type"]) if input.get("content_type") else None),
            delete=delete,
            output=str(output),
        )
        results = list(getattr(result, "results", []) or [])
        failed = [r for r in results if r.get("status") != "applied"]
        summary = {
            "path": str(getattr(result, "path", output)),
            "part": part,
            "applied": getattr(result, "applied", 0),
            "failed": len(failed),
            "results": results,
        }
        return ToolResult(
            content=json.dumps(summary, ensure_ascii=False, indent=1, default=str),
            metadata={"applied": summary["applied"], "failed": len(failed)},
        )


class DocBuildTool(_DocToolBase):
    """Build a new document from a structured spec (deterministic, no LLM)."""

    @property
    def name(self) -> str:
        return "DocBuild"

    @property
    def description(self) -> str:
        return (
            "GENERATE (deterministic): build a NEW document from YOUR spec — "
            ".docx←markdown, .xlsx←{sheets}, .pptx←{slides}. Instant, no key. "
            "Spec shapes: DocGuide('build')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spec": {
                    "type": ["string", "object"],
                    "description": (
                        "docx: markdown string. xlsx: {sheets:[...]}. "
                        "pptx: {slides:[...]}. Must match the output format."
                    ),
                },
                "output": {
                    "type": "string",
                    "description": "Output path — extension (.docx/.xlsx/.pptx) selects the format.",
                },
                "lang": {"type": "string", "description": "BCP-47 language tag (optional)."},
            },
            "required": ["spec", "output"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=20_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        output = _resolve_doc_path(input.get("output") or "", context, must_exist=False)
        if output.suffix.lower() not in _SUPPORTED_EXTS:
            return ToolResult(
                content=f"output must end in one of {_SUPPORTED_EXTS}", is_error=True
            )
        if "spec" not in input:
            return ToolResult(content="spec is required", is_error=True)
        kwargs: Dict[str, Any] = {}
        lang = input.get("lang")
        if lang:
            kwargs["lang"] = str(lang)
        result = await asyncio.to_thread(
            engine.build_doc, input["spec"], str(output), **kwargs
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


class DocGenerateTool(_DocToolBase):
    """Create a new document from an intent (LLM-backed)."""

    @property
    def name(self) -> str:
        return "DocGenerate"

    def required_config_keys(self):
        # Progressive disclosure: keyless hosts never register the LLM verbs.
        return [_DOCS_LLM_FEATURE_KEY]

    @property
    def description(self) -> str:
        return (
            "GENERATE (LLM): a complete designed document from a one-line "
            "intent (.pptx is slow — minutes). Options: DocGuide('generate'). "
            "Keyless alternative: DocBuild."
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

    def required_config_keys(self):
        # Progressive disclosure: keyless hosts never register the LLM verbs.
        return [_DOCS_LLM_FEATURE_KEY]

    @property
    def description(self) -> str:
        return (
            "EDIT (LLM): one natural-language edit turn. Prefer the "
            "deterministic path: DocAnalyze → DocApplyEdits / DocXmlEdit. "
            "Guide: DocGuide('edit')."
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


class DocRenderTool(_DocToolBase):
    """Page images / PDF via the edit2docs native pipeline (no LibreOffice)."""

    @property
    def name(self) -> str:
        return "DocRender"

    @property
    def description(self) -> str:
        return (
            "Render a document: to=md (read the content) | svg | png | pdf. "
            "Deterministic, no key. Guide: DocGuide('render')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document file path."},
                "to": {
                    "type": "string",
                    "enum": ["png", "pdf", "svg", "md"],
                    "description": "Output kind (default png; md = readable content).",
                },
                "out_dir": {
                    "type": "string",
                    "description": "Output directory (default '<doc dir>/render').",
                },
                "dpi": {
                    "type": "number",
                    "description": "Raster resolution (default 144).",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=20_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        engine = _load_edit2docs()
        if not hasattr(engine, "render_doc"):
            return ToolResult(
                content=(
                    "This edit2docs version has no render_doc — upgrade to "
                    "edit2docs>=0.6.0 (pip install 'geny-executor[docs]')."
                ),
                is_error=True,
            )
        path = _resolve_doc_path(input.get("path") or "", context)
        out = input.get("out_dir")
        kwargs: Dict[str, Any] = {
            "to": str(input.get("to") or "png"),
            "dpi": float(input.get("dpi") or 144.0),
        }
        if out:
            kwargs["out_dir"] = str(
                resolve_and_validate(
                    str(out), context.working_dir or os.getcwd(), context.allowed_paths
                )
            )
        result = await asyncio.to_thread(engine.render_doc, str(path), **kwargs)
        payload = {
            "paths": [str(p) for p in getattr(result, "paths", [])],
            "page_count": getattr(result, "page_count", 0),
            "format": getattr(result, "format", ""),
            "to": getattr(result, "to", kwargs["to"]),
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=1),
            metadata=payload,
        )


DOC_TOOL_CLASSES: Dict[str, type] = {
    "DocGuide": DocGuideTool,
    "DocAnalyze": DocAnalyzeTool,
    "DocApplyEdits": DocApplyEditsTool,
    "DocArrange": DocArrangeTool,
    "DocBuild": DocBuildTool,
    "DocXmlRead": DocXmlReadTool,
    "DocXmlEdit": DocXmlEditTool,
    "DocRender": DocRenderTool,
    # LLM conveniences — gated on feature:docs_llm (required_config_keys),
    # so keyless hosts never register them and the model never sees a tool
    # that can only error.
    "DocGenerate": DocGenerateTool,
    "DocEdit": DocEditTool,
}
