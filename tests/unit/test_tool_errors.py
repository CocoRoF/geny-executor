"""Unit tests for structured tool errors (Phase A, v0.22.0)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.errors import (
    ToolError,
    ToolErrorCode,
    ToolFailure,
    make_error_result,
    validate_input,
)
from geny_executor.tools.registry import ToolRegistry


# ── Helpers ──────────────────────────────────────────────


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", working_dir="/tmp")


class _EchoTool(Tool):
    """Tool that echoes `x`; requires `x` as a string."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echoes x"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=f"echo:{input['x']}")


class _FailureTool(Tool):
    """Raises ToolFailure with a TRANSPORT code."""

    @property
    def name(self) -> str:
        return "failure"

    @property
    def description(self) -> str:
        return "raises ToolFailure"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raise ToolFailure(
            "rate limited",
            code=ToolErrorCode.TRANSPORT,
            details={"retry_after": 30},
        )


class _CrashTool(Tool):
    """Raises an unexpected exception."""

    @property
    def name(self) -> str:
        return "crash"

    @property
    def description(self) -> str:
        return "crashes"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")


# ══════════════════════════════════════════════════════════
# ToolError dataclass / factories
# ══════════════════════════════════════════════════════════


class TestToolErrorPayload:
    def test_payload_shape(self):
        err = ToolError(
            code=ToolErrorCode.INVALID_INPUT,
            message="bad thing",
            details={"field": "x"},
        )
        payload = err.to_payload()
        assert payload == {
            "error": {
                "code": "invalid_input",
                "message": "bad thing",
                "details": {"field": "x"},
            }
        }

    def test_payload_copies_details(self):
        details = {"key": "value"}
        err = ToolError(code=ToolErrorCode.TOOL_CRASHED, message="m", details=details)
        payload = err.to_payload()
        payload["error"]["details"]["mutated"] = True
        assert "mutated" not in details

    def test_unknown_tool_factory(self):
        err = ToolError.unknown_tool("news_search", known=["echo", "crash"])
        assert err.code is ToolErrorCode.UNKNOWN_TOOL
        assert "news_search" in err.message
        assert err.details["tool_name"] == "news_search"
        assert err.details["known_tools"] == ["crash", "echo"]

    def test_unknown_tool_without_known(self):
        err = ToolError.unknown_tool("x")
        assert err.code is ToolErrorCode.UNKNOWN_TOOL
        assert "known_tools" not in err.details

    def test_invalid_input_factory(self):
        err = ToolError.invalid_input("echo", "missing x", path="x")
        assert err.code is ToolErrorCode.INVALID_INPUT
        assert err.details["tool_name"] == "echo"
        assert err.details["reason"] == "missing x"
        assert err.details["path"] == "x"

    def test_tool_crashed_factory(self):
        exc = RuntimeError("boom")
        err = ToolError.tool_crashed("echo", exc)
        assert err.code is ToolErrorCode.TOOL_CRASHED
        assert err.details["exception_type"] == "RuntimeError"
        assert err.details["exception_message"] == "boom"

    def test_access_denied_factory(self):
        err = ToolError.access_denied("echo")
        assert err.code is ToolErrorCode.ACCESS_DENIED
        assert err.details["tool_name"] == "echo"

    def test_transport_factory(self):
        err = ToolError.transport("web_search", "connection lost")
        assert err.code is ToolErrorCode.TRANSPORT
        assert err.details["server"] == "web_search"
        assert err.details["reason"] == "connection lost"


# ══════════════════════════════════════════════════════════
# make_error_result + ToolResult.to_api_format
# ══════════════════════════════════════════════════════════


class TestMakeErrorResult:
    def test_returns_tool_result_marked_error(self):
        err = ToolError.unknown_tool("x")
        result = make_error_result(err)
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert result.metadata["error_code"] == "unknown_tool"
        assert isinstance(result.content, dict)
        assert result.content["error"]["code"] == "unknown_tool"


class TestToApiFormatErrorHeader:
    def test_error_dict_gets_header_line(self):
        err = ToolError(
            code=ToolErrorCode.INVALID_INPUT,
            message="bad x",
            details={"path": "x"},
        )
        result = make_error_result(err)
        api = result.to_api_format("toolu_123")
        assert api["type"] == "tool_result"
        assert api["tool_use_id"] == "toolu_123"
        assert api["is_error"] is True
        content: str = api["content"]
        first_line, _, body = content.partition("\n")
        assert first_line == "ERROR invalid_input: bad x"
        parsed = json.loads(body)
        assert parsed["error"]["code"] == "invalid_input"
        assert parsed["error"]["details"]["path"] == "x"

    def test_non_error_dict_json_stringified(self):
        result = ToolResult(content={"foo": "bar"})
        api = result.to_api_format("toolu_1")
        assert api["content"] == json.dumps({"foo": "bar"}, ensure_ascii=False)
        assert "is_error" not in api

    def test_dict_missing_required_fields_no_header(self):
        # Looks like error but missing the `message` string → treat as plain dict.
        result = ToolResult(
            content={"error": {"code": "x"}},
            is_error=True,
        )
        api = result.to_api_format("toolu_1")
        assert not api["content"].startswith("ERROR ")
        assert api["is_error"] is True

    def test_string_content_passthrough(self):
        result = ToolResult(content="hello")
        api = result.to_api_format("toolu_1")
        assert api["content"] == "hello"

    def test_list_content_passthrough(self):
        blocks = [{"type": "text", "text": "ok"}]
        result = ToolResult(content=blocks)
        api = result.to_api_format("toolu_1")
        assert api["content"] is blocks


# ══════════════════════════════════════════════════════════
# validate_input
# ══════════════════════════════════════════════════════════


class TestValidateInput:
    def test_pass(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        validate_input(schema, {"x": "hello"})  # no raise

    def test_fail(self):
        import jsonschema

        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        with pytest.raises(jsonschema.ValidationError):
            validate_input(schema, {})


# ══════════════════════════════════════════════════════════
# ToolFailure
# ══════════════════════════════════════════════════════════


class TestToolFailure:
    def test_default_code(self):
        f = ToolFailure("boom")
        assert f.error.code is ToolErrorCode.TOOL_CRASHED
        assert f.error.message == "boom"
        assert f.error.details == {}

    def test_explicit_code_and_details(self):
        f = ToolFailure(
            "rate limited",
            code=ToolErrorCode.TRANSPORT,
            details={"retry_after": 30},
        )
        assert f.error.code is ToolErrorCode.TRANSPORT
        assert f.error.details == {"retry_after": 30}


# ══════════════════════════════════════════════════════════
# RegistryRouter integration
# ══════════════════════════════════════════════════════════


class TestRegistryRouter:
    def _router(self, *tools: Tool) -> RegistryRouter:
        reg = ToolRegistry()
        for t in tools:
            reg.register(t)
        return RegistryRouter(reg)

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        router = self._router(_EchoTool())
        result = await router.route("news_search", {}, _ctx())
        assert result.is_error
        assert result.content["error"]["code"] == "unknown_tool"
        assert result.content["error"]["details"]["tool_name"] == "news_search"
        assert "echo" in result.content["error"]["details"]["known_tools"]

    @pytest.mark.asyncio
    async def test_invalid_input(self):
        router = self._router(_EchoTool())
        result = await router.route("echo", {}, _ctx())  # missing `x`
        assert result.is_error
        assert result.content["error"]["code"] == "invalid_input"
        assert result.content["error"]["details"]["tool_name"] == "echo"

    @pytest.mark.asyncio
    async def test_invalid_input_wrong_type(self):
        router = self._router(_EchoTool())
        result = await router.route("echo", {"x": 123}, _ctx())
        assert result.is_error
        assert result.content["error"]["code"] == "invalid_input"
        assert result.content["error"]["details"]["path"] == "x"

    @pytest.mark.asyncio
    async def test_happy_path(self):
        router = self._router(_EchoTool())
        result = await router.route("echo", {"x": "hi"}, _ctx())
        assert not result.is_error
        assert result.content == "echo:hi"

    @pytest.mark.asyncio
    async def test_tool_failure_passthrough(self):
        router = self._router(_FailureTool())
        result = await router.route("failure", {}, _ctx())
        assert result.is_error
        assert result.content["error"]["code"] == "transport_error"
        assert result.content["error"]["message"] == "rate limited"
        assert result.content["error"]["details"]["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_unexpected_crash(self, caplog):
        router = self._router(_CrashTool())
        with caplog.at_level("ERROR"):
            result = await router.route("crash", {}, _ctx())
        assert result.is_error
        assert result.content["error"]["code"] == "tool_crashed"
        assert result.content["error"]["details"]["exception_type"] == "RuntimeError"
        assert any("crash" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_bind_registry_swaps_backend(self):
        router = RegistryRouter()
        result = await router.route("echo", {"x": "hi"}, _ctx())
        assert result.is_error  # empty registry

        reg = ToolRegistry().register(_EchoTool())
        router.bind_registry(reg)
        result = await router.route("echo", {"x": "hi"}, _ctx())
        assert not result.is_error
        assert result.content == "echo:hi"

    def test_name_and_description(self):
        router = RegistryRouter()
        assert router.name == "registry"
        assert "Registry" in router.description or "registry" in router.description
