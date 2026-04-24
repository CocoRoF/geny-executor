"""Phase 7 Sprint S7.3 — StructuredOutputParser schema-validation tests."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from geny_executor.llm_client.types import ContentBlock
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s09_parse.artifact.default.parsers import (
    StructuredOutputParser,
    _try_parse_json,
)


def _resp(text: str) -> APIResponse:
    return APIResponse(content=[ContentBlock(type="text", text=text)])


# ─────────────────────────────────────────────────────────────────
# _try_parse_json helper
# ─────────────────────────────────────────────────────────────────


class TestTryParseJson:
    def test_direct_parse(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_whitespace_trimmed(self):
        assert _try_parse_json('   \n {"a": 1}\n   ') == {"a": 1}

    def test_code_fence_extracted(self):
        text = "Here is the result:\n```json\n{\"x\": 42}\n```\nDone."
        assert _try_parse_json(text) == {"x": 42}

    def test_unfenced_code_block_extracted(self):
        text = "```\n[1, 2, 3]\n```"
        assert _try_parse_json(text) == [1, 2, 3]

    def test_garbage_returns_none(self):
        assert _try_parse_json("not json at all") is None

    def test_empty_returns_none(self):
        assert _try_parse_json("") is None


# ─────────────────────────────────────────────────────────────────
# Parser without schema (legacy behaviour)
# ─────────────────────────────────────────────────────────────────


class TestNoSchemaParser:
    def test_clean_json_passes(self):
        parser = StructuredOutputParser()
        result = parser.parse(_resp('{"x": 1}'))
        assert result.structured_output == {"x": 1}
        assert result.structured_output_error is None

    def test_garbage_text_records_parse_error(self):
        parser = StructuredOutputParser()
        result = parser.parse(_resp("hello, no json here"))
        assert result.structured_output is None
        assert result.structured_output_error is not None
        assert "parse failed" in result.structured_output_error.lower()

    def test_empty_text_no_error(self):
        """Empty text isn't a contract failure — it's just absent
        structured output."""
        parser = StructuredOutputParser()
        result = parser.parse(_resp(""))
        assert result.structured_output is None
        assert result.structured_output_error is None

    def test_code_fence_extracted(self):
        parser = StructuredOutputParser()
        result = parser.parse(_resp("```json\n{\"y\": 2}\n```"))
        assert result.structured_output == {"y": 2}


# ─────────────────────────────────────────────────────────────────
# Parser with schema
# ─────────────────────────────────────────────────────────────────


_VALID_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


class TestSchemaParser:
    def test_passing_payload(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(_resp('{"name": "Alice", "age": 30}'))
        assert result.structured_output == {"name": "Alice", "age": 30}
        assert result.structured_output_error is None

    def test_missing_required_field_records_error(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(_resp('{"name": "Bob"}'))
        # Validation failure clears the value to None
        assert result.structured_output is None
        assert result.structured_output_error is not None
        assert "schema mismatch" in result.structured_output_error.lower()
        assert "age" in result.structured_output_error

    def test_wrong_type_records_error(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(_resp('{"name": "Bob", "age": "thirty"}'))
        assert result.structured_output is None
        assert "schema mismatch" in result.structured_output_error.lower()

    def test_additional_property_records_error(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(
            _resp('{"name": "Bob", "age": 30, "unexpected": true}')
        )
        assert result.structured_output is None
        assert "schema mismatch" in result.structured_output_error.lower()

    def test_minimum_constraint_enforced(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(_resp('{"name": "Bob", "age": -5}'))
        assert result.structured_output is None
        assert "schema mismatch" in result.structured_output_error.lower()

    def test_parse_failure_takes_precedence_over_schema(self):
        """A non-JSON text should report the parse failure, not a
        schema-shaped error."""
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        result = parser.parse(_resp("not json at all"))
        assert result.structured_output is None
        assert "parse failed" in result.structured_output_error.lower()

    def test_invalid_schema_at_construction_raises(self):
        # ``minimum`` should be a number, not a string
        bad_schema = {"type": "integer", "minimum": "zero"}
        with pytest.raises(ValueError, match="invalid JSON Schema"):
            StructuredOutputParser(schema=bad_schema)


# ─────────────────────────────────────────────────────────────────
# Strategy metadata
# ─────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_no_schema_description(self):
        parser = StructuredOutputParser()
        assert "no schema" in parser.description.lower()

    def test_with_schema_description(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        assert "schema-validated" in parser.description.lower()

    def test_schema_property_returns_bound_schema(self):
        parser = StructuredOutputParser(schema=_VALID_SCHEMA)
        assert parser.schema == _VALID_SCHEMA

    def test_schema_property_none_when_unset(self):
        parser = StructuredOutputParser()
        assert parser.schema is None


# ─────────────────────────────────────────────────────────────────
# ParsedResponse.structured_output_error field surface
# ─────────────────────────────────────────────────────────────────


class TestParsedResponseField:
    def test_default_is_none(self):
        from geny_executor.stages.s09_parse.types import ParsedResponse

        r = ParsedResponse()
        assert r.structured_output_error is None

    def test_can_round_trip(self):
        from geny_executor.stages.s09_parse.types import ParsedResponse

        r = ParsedResponse(
            text="x",
            structured_output=None,
            structured_output_error="schema mismatch at root: missing 'x'",
        )
        assert r.structured_output_error == "schema mismatch at root: missing 'x'"
