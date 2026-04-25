"""``AskUserQuestion`` — let the LLM ask the user a free-text question (PR-A.3.1).

Why this is its own tool (not Stage 15 HITL): HITL is approve / reject /
cancel — a binary verdict on a tool call the LLM already proposed.
``AskUserQuestion`` is the inverse: the LLM doesn't yet know what to
do, so it asks for a free-text answer (or one of an enumerated list)
and waits.

Wiring contract: the host injects ``question_handler`` into
``ToolContext.extras``. Signature::

    async def question_handler(
        *,
        question: str,
        options: Optional[List[str]],
        default: Optional[str],
        timeout_seconds: int,
        prompt_id: str,
    ) -> str

The handler is the host's UI side — it shows the question, waits for
the user, returns the answer (or raises ``asyncio.TimeoutError`` /
``QuestionCancelled``).
"""

from __future__ import annotations

import asyncio
import secrets
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


class QuestionCancelled(Exception):
    """Raised by the host's question_handler when the user dismisses
    the prompt without answering."""


class AskUserQuestionTool(Tool):
    @property
    def name(self) -> str:
        return "AskUserQuestion"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question and wait for their response. Use sparingly — "
            "only when the answer is required to continue. Optional 'options' offers "
            "a multiple-choice list."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1, "maxLength": 1000},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
                "default": {"type": "string"},
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 86400,
                    "default": 600,
                },
            },
            "required": ["question"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Blocks waiting on a human; never run in parallel with other
        # tools because the user typically can answer one at a time
        # and parallel asks are confusing.
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=True,
            destructive=False,
            interrupt="cancel",
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        handler = context.extras.get("question_handler")
        if handler is None:
            return ToolResult(
                content={
                    "error": {
                        "code": "NO_HANDLER",
                        "message": (
                            "question_handler was not wired into ToolContext.extras. "
                            "Host must register an async handler at startup."
                        ),
                    },
                },
                is_error=True,
            )
        question = input.get("question", "")
        if not question:
            return ToolResult(
                content={"error": {"code": "BAD_INPUT", "message": "question is required"}},
                is_error=True,
            )
        timeout = int(input.get("timeout_seconds", 600))
        prompt_id = secrets.token_urlsafe(12)
        try:
            answer = await asyncio.wait_for(
                handler(
                    question=question,
                    options=input.get("options"),
                    default=input.get("default"),
                    timeout_seconds=timeout,
                    prompt_id=prompt_id,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content={
                    "error": {
                        "code": "TIMEOUT",
                        "message": f"user did not answer within {timeout} seconds",
                    },
                },
                is_error=True,
            )
        except QuestionCancelled as exc:
            return ToolResult(
                content={
                    "error": {
                        "code": "CANCELLED",
                        "message": str(exc) or "user cancelled the prompt",
                    },
                },
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface to LLM
            return ToolResult(
                content={
                    "error": {
                        "code": "HANDLER_FAILED",
                        "message": str(exc),
                    },
                },
                is_error=True,
            )
        if not isinstance(answer, str):
            answer = str(answer)
        return ToolResult(content={"answer": answer, "prompt_id": prompt_id})


__all__ = ["AskUserQuestionTool", "QuestionCancelled"]
