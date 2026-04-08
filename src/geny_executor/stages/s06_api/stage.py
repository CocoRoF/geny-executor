"""Stage 6: API — calls Anthropic Messages API."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.providers import APIProvider, AnthropicProvider
from geny_executor.stages.s06_api.retry import ExponentialBackoffRetry, RetryStrategy
from geny_executor.stages.s06_api.types import APIRequest, APIResponse


class APIStage(Stage[Any, APIResponse]):
    """Stage 6: API.

    Dual abstraction:
      - Level 2 provider: actual API call implementation
      - Level 2 retry: error recovery strategy
    """

    def __init__(
        self,
        provider: Optional[APIProvider] = None,
        retry: Optional[RetryStrategy] = None,
        *,
        api_key: str = "",
        base_url: Optional[str] = None,
    ):
        if provider:
            self._provider = provider
        elif api_key:
            self._provider = AnthropicProvider(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("Either 'provider' or 'api_key' must be provided")

        self._retry = retry or ExponentialBackoffRetry()

    @property
    def name(self) -> str:
        return "api"

    @property
    def order(self) -> int:
        return 6

    @property
    def category(self) -> str:
        return "execution"

    async def execute(self, input: Any, state: PipelineState) -> APIResponse:
        request = self._build_request(state)
        state.add_event("api.request", {
            "model": request.model,
            "message_count": len(request.messages),
            "has_tools": bool(request.tools),
            "has_thinking": bool(request.thinking),
        })

        response = await self._call_with_retry(request, state)

        # Store raw response for downstream stages
        state.last_api_response = response

        # Add assistant message to conversation
        assistant_content = self._build_assistant_content(response)
        state.add_message("assistant", assistant_content)

        state.add_event("api.response", {
            "stop_reason": response.stop_reason,
            "text_length": len(response.text),
            "tool_calls": len(response.tool_calls),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        return response

    def _build_request(self, state: PipelineState) -> APIRequest:
        """Build APIRequest from pipeline state."""
        request = APIRequest(
            model=state.model,
            messages=list(state.messages),
            max_tokens=state.max_tokens,
            system=state.system,
            temperature=state.temperature,
            stop_sequences=state.stop_sequences,
        )

        if state.tools:
            request.tools = state.tools
        if state.tool_choice:
            request.tool_choice = state.tool_choice
        if state.thinking_enabled:
            request.thinking = {
                "type": "enabled",
                "budget_tokens": state.thinking_budget_tokens,
            }

        return request

    async def _call_with_retry(self, request: APIRequest, state: PipelineState) -> APIResponse:
        """Execute API call with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._provider.create_message(request)
                return response
            except APIError as e:
                last_error = e
                if not self._retry.should_retry(e.category, attempt):
                    raise
                delay = self._retry.get_delay(attempt)
                state.add_event("api.retry", {
                    "attempt": attempt + 1,
                    "category": e.category.value,
                    "delay": delay,
                })
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                category = ErrorCategory.UNKNOWN
                if not self._retry.should_retry(category, attempt):
                    raise APIError(str(e), category=category, cause=e) from e
                delay = self._retry.get_delay(attempt)
                state.add_event("api.retry", {
                    "attempt": attempt + 1,
                    "category": category.value,
                    "delay": delay,
                })
                await asyncio.sleep(delay)

        raise last_error or APIError("Max retries exceeded", category=ErrorCategory.UNKNOWN)

    def _build_assistant_content(self, response: APIResponse) -> Any:
        """Build assistant content for message history."""
        blocks = []
        for block in response.content:
            if block.raw:
                blocks.append(block.raw)
            elif block.type == "text":
                blocks.append({"type": "text", "text": block.text or ""})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use",
                    "id": block.tool_use_id,
                    "name": block.tool_name,
                    "input": block.tool_input,
                })
        return blocks if len(blocks) != 1 or blocks[0].get("type") != "text" else blocks[0].get("text", "")

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="provider",
                current_impl=type(self._provider).__name__,
                available_impls=[
                    "AnthropicProvider",
                    "MockProvider",
                    "RecordingProvider",
                ],
            ),
            StrategyInfo(
                slot_name="retry",
                current_impl=type(self._retry).__name__,
                available_impls=[
                    "ExponentialBackoffRetry",
                    "NoRetry",
                    "RateLimitAwareRetry",
                ],
            ),
        ]
