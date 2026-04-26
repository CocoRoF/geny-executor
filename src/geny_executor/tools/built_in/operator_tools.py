"""Operator-facing tools — Config / Monitor / SendUserFile (PR-A.3.6)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


# ── ConfigTool ───────────────────────────────────────────────────────


class ConfigTool(Tool):
    """Inspect / mutate runtime configuration via a host-supplied
    PipelineMutator (or fall back to direct manifest mutation).

    Action ``list_active`` is read-only and works without a mutator.
    Action ``set`` requires ``ctx.extras["pipeline_mutator"]``.
    """

    @property
    def name(self) -> str:
        return "Config"

    @property
    def description(self) -> str:
        return (
            "Inspect or mutate active strategies. "
            "Use action='list_active' for a read-only snapshot."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"enum": ["list_active", "get", "set"]},
                "section": {"type": "string"},
                "key": {"type": "string"},
                "value": {},
            },
            "required": ["action"],
        }

    def capabilities(self, input):
        return ToolCapabilities(
            concurrency_safe=False,
            destructive=(input.get("action") == "set"),
        )

    async def execute(self, input, context):
        action = input["action"]
        if action == "list_active":
            pipeline = context.extras.get("pipeline")
            if pipeline is None:
                return _err("NO_PIPELINE", "pipeline not wired into ctx.extras")
            stages = getattr(pipeline, "stages", None) or []
            out: List[Dict[str, Any]] = []
            for stage in stages:
                stage_name = getattr(stage, "name", None) or type(stage).__name__
                slots_fn = getattr(stage, "get_strategy_slots", None)
                if not callable(slots_fn):
                    continue
                try:
                    slots = slots_fn() or {}
                except Exception:
                    continue
                for slot_name, slot in slots.items():
                    strategy = getattr(slot, "strategy", slot)
                    out.append({
                        "stage": stage_name,
                        "slot": slot_name,
                        "impl": type(strategy).__name__,
                    })
            return ToolResult(content={"active": out})
        mutator = context.extras.get("pipeline_mutator")
        if mutator is None:
            return _err("NO_MUTATOR", "pipeline_mutator not wired for set/get actions")
        if action == "get":
            try:
                value = mutator.get(input.get("section"), input.get("key"))
            except Exception as exc:  # noqa: BLE001
                return _err("CONFIG_GET_FAILED", str(exc))
            return ToolResult(content={"section": input.get("section"), "key": input.get("key"), "value": value})
        # action == "set"
        try:
            mutator.set(input.get("section"), input.get("key"), input.get("value"))
        except Exception as exc:  # noqa: BLE001
            return _err("CONFIG_SET_FAILED", str(exc))
        return ToolResult(content={"section": input.get("section"), "key": input.get("key"), "set": True})


# ── MonitorTool ──────────────────────────────────────────────────────


class MonitorTool(Tool):
    """Subscribe to EventBus events for ``duration_seconds``; return
    collected events. The bus is host-supplied via
    ``ctx.extras["event_bus"]`` and only needs an ``async subscribe``
    context manager + iterable of events with ``.type`` / ``.ts`` /
    ``.data`` attributes.
    """

    @property
    def name(self) -> str:
        return "Monitor"

    @property
    def description(self) -> str:
        return (
            "Subscribe to runtime events for a bounded duration; "
            "returns the collected events as a list."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "events": {"type": "array", "items": {"type": "string"}},
                "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 5},
                "max_events": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
            },
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, read_only=True)

    async def execute(self, input, context):
        bus = context.extras.get("event_bus")
        if bus is None:
            return _err("NO_BUS", "event_bus not wired into ctx.extras")
        events_filter = input.get("events") or None
        duration = int(input.get("duration_seconds", 5))
        max_events = int(input.get("max_events", 100))
        collected: List[Dict[str, Any]] = []
        sub_fn = getattr(bus, "subscribe", None)
        if not callable(sub_fn):
            return _err("BUS_API", "event_bus has no subscribe method")
        try:
            ctx_mgr = sub_fn(events_filter) if events_filter is not None else sub_fn()
            async with ctx_mgr as stream:
                async def _drain():
                    async for evt in stream:
                        collected.append({
                            "type": getattr(evt, "type", None),
                            "ts": _isofy(getattr(evt, "ts", None)),
                            "data": getattr(evt, "data", None),
                        })
                        if len(collected) >= max_events:
                            return
                try:
                    await asyncio.wait_for(_drain(), timeout=duration)
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:  # noqa: BLE001
            return _err("MONITOR_FAILED", str(exc))
        return ToolResult(content={"count": len(collected), "events": collected})


def _isofy(value):
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


# ── SendUserFileTool ────────────────────────────────────────────────


class SendUserFileTool(Tool):
    """Deliver a file to the user via a host-supplied UserFileChannel."""

    @property
    def name(self) -> str:
        return "SendUserFile"

    @property
    def description(self) -> str:
        return "Deliver a file to the user. Returns delivery metadata (e.g. download URL)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "filename": {"type": "string"},
                "content_type": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["file_path"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, network_egress=True)

    async def execute(self, input, context):
        channel = context.extras.get("user_file_channel")
        if channel is None:
            return _err("NO_CHANNEL", "user_file_channel not wired into ctx.extras")
        cwd = context.working_dir or "."
        path = Path(input["file_path"])
        if not path.is_absolute():
            path = Path(cwd) / path
        if not path.exists():
            return _err("FILE_NOT_FOUND", str(path))
        if not path.is_file():
            return _err("NOT_A_FILE", str(path))
        try:
            result = await channel.send(
                path,
                filename=input.get("filename") or path.name,
                content_type=input.get("content_type"),
                description=input.get("description"),
            )
        except Exception as exc:  # noqa: BLE001
            return _err("SEND_FAILED", str(exc))
        return ToolResult(content={"delivered": True, "result": result})


__all__ = ["ConfigTool", "MonitorTool", "SendUserFileTool"]
