"""Response parsers — Level 2 strategies for parsing API responses."""

from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, Optional

from geny_executor.core.stage import Strategy
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s09_parse.types import ParsedResponse, ToolCall


class ResponseParser(Strategy):
    """Base interface for response parsing."""

    @abstractmethod
    def parse(self, response: APIResponse) -> ParsedResponse:
        """Parse API response into structured form."""
        ...


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
    """Parses response text as structured JSON output."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self._schema = schema

    @property
    def name(self) -> str:
        return "structured_output"

    @property
    def description(self) -> str:
        return "JSON structured output parser"

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

        # Try to extract structured output from text
        structured = self._try_parse_json(text)

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            structured_output=structured,
            api_response=response,
        )

    def _try_parse_json(self, text: str) -> Optional[Any]:
        """Try to parse JSON from text, including code blocks."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from ```json ... ``` blocks
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        return None
