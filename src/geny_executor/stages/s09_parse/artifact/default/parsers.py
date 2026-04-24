"""Response parsers — concrete Level 2 strategies for parsing API responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import jsonschema

from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s09_parse.interface import ResponseParser
from geny_executor.stages.s09_parse.types import ParsedResponse, ToolCall

logger = logging.getLogger(__name__)


class DefaultParser(ResponseParser):
    """Standard parser — extracts text, tool calls, thinking."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Standard text/tool/thinking extraction"

    def parse(self, response: APIResponse) -> ParsedResponse:
        text = response.text
        tool_calls = [
            ToolCall(
                tool_use_id=block.tool_use_id or "",
                tool_name=block.tool_name or "",
                tool_input=block.tool_input or {},
            )
            for block in response.tool_calls
        ]
        thinking_texts = [
            block.thinking_text or "" for block in response.thinking_blocks if block.thinking_text
        ]

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            thinking_texts=thinking_texts,
            api_response=response,
        )


class StructuredOutputParser(ResponseParser):
    """Parses response text as structured JSON output and (optionally)
    validates it against a JSON Schema.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.3 (Structured
    output schema contract). Pre-S7.3 the parser would extract a JSON
    body and silently leave the result as ``None`` on failure. With
    S7.3 hosts can supply a schema; the parser distinguishes:

    * **Parse failure** — text wasn't JSON. ``structured_output``
      stays ``None``; ``structured_output_error`` carries
      ``"JSON parse failed: ..."``.
    * **Validation failure** — JSON parsed but didn't match the
      schema. ``structured_output`` stays ``None``;
      ``structured_output_error`` carries
      ``"schema mismatch: ..."``.
    * **Success** — ``structured_output`` is the parsed value;
      ``structured_output_error`` is ``None``.

    Hosts that don't pass a schema get the legacy behaviour: parse
    on best-effort, ``structured_output`` is the parsed value or
    ``None``, ``structured_output_error`` populated only on parse
    failure.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        # Validate the schema itself once at construction time so
        # malformed contracts surface immediately rather than at
        # first-call. ``Draft7Validator.check_schema`` is the
        # widely-supported draft.
        if schema is not None:
            try:
                jsonschema.Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise ValueError(
                    f"StructuredOutputParser: invalid JSON Schema: {exc.message}"
                ) from exc
        self._schema = schema

    @property
    def name(self) -> str:
        return "structured_output"

    @property
    def description(self) -> str:
        if self._schema is None:
            return "JSON structured output parser (no schema)"
        return "JSON structured output parser (schema-validated)"

    @property
    def schema(self) -> Optional[Dict[str, Any]]:
        """The bound JSON Schema, or ``None`` for unvalidated parse."""
        return self._schema

    def parse(self, response: APIResponse) -> ParsedResponse:
        text = response.text
        tool_calls = [
            ToolCall(
                tool_use_id=block.tool_use_id or "",
                tool_name=block.tool_name or "",
                tool_input=block.tool_input or {},
            )
            for block in response.tool_calls
        ]

        structured, error = self._extract(text)

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            structured_output=structured,
            structured_output_error=error,
            api_response=response,
        )

    def _extract(self, text: str) -> Tuple[Optional[Any], Optional[str]]:
        """Parse + (optional) validate; return ``(value, error)``.

        Either both are ``None`` (no structured output expected /
        empty text) OR one is non-``None``. Never returns both
        populated — a validation failure clears ``value``.
        """
        if not text or not text.strip():
            return None, None

        parsed = _try_parse_json(text)
        if parsed is None:
            return None, "JSON parse failed: text contained no valid JSON"

        if self._schema is None:
            return parsed, None

        try:
            jsonschema.validate(parsed, self._schema)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
            return None, f"schema mismatch at {path}: {exc.message}"

        return parsed, None


def _try_parse_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from raw text.

    Tries direct parse first, then peels a ````json ... ```` code
    fence. Returns ``None`` for any failure mode — callers wanting
    structured error messages should layer on top.
    """
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    return None
