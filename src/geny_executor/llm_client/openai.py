"""OpenAI Chat Completions API client.

Ported from the former :class:`OpenAIProvider` in
``stages/s06_api/artifact/openai/providers.py``. Translators are
imported from :mod:`geny_executor.llm_client.translators`, which
re-exports from the s06_api module during the PR-3→PR-4 bridge.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators import (
    canonical_messages_to_openai,
    canonical_thinking_to_openai,
    canonical_tool_choice_to_openai,
    canonical_tools_to_openai,
    normalize_stop_reason,
)
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


logger = logging.getLogger(__name__)


# ── ``max_tokens`` → ``max_completion_tokens`` migration ────────────
#
# OpenAI's reasoning families reject the classic ``max_tokens`` kwarg:
#
#   ``Unsupported parameter: 'max_tokens' is not supported with this
#     model. Use 'max_completion_tokens' instead.``
#
# The audit (§1-4) flagged this drift as "already real in prod" with
# zero defense on the OpenAI boundary. Two layers, mirroring the
# Anthropic pattern proven in 2.1.2/2.1.3:
#
#   1. Proactive — the static prefix table below sends
#      ``max_completion_tokens`` up front for families known to demand
#      it. Prefix match so dated/sized variants (``o3-mini``,
#      ``gpt-5.2-codex``) ride along without a code change.
#   2. Reactive — ``_heal_request_kwargs`` rebuilds + retries once when
#      the 400 names the rename, covering whatever family ships after
#      this table goes stale. Every reactive heal warns loudly so the
#      table gets refreshed instead of paying the retry forever.
_MAX_COMPLETION_TOKENS_PREFIXES: tuple[str, ...] = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def _model_requires_max_completion_tokens(model: str) -> bool:
    """True iff ``model`` belongs to a family known to reject
    ``max_tokens`` in favour of ``max_completion_tokens``."""
    return any(model.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES)


class OpenAIClient(BaseClient):
    """OpenAI Chat Completions API client.

    Requires: ``pip install geny-executor[openai]``
    """

    provider = "openai"
    _sdk_module = "openai"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=True,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=False,
        requires_workspace=False,
        streaming_granularity="token",
        drops=("thinking_enabled", "top_k"),
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._client: Optional[Any] = None

    def configure(self, **kwargs: Any) -> None:
        super().configure(**kwargs)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI client requires the 'openai' package. "
                    "Install with: pip install geny-executor[openai]"
                ) from e
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Self-heal the ``max_tokens`` → ``max_completion_tokens`` rename.

        Triggers only when the 400 message names the problem — it must
        mention ``max_tokens`` *and* either say the param is not
        supported or name the replacement kwarg. The live phrasing
        (verified against openai 2.x):

          ``Unsupported parameter: 'max_tokens' is not supported with
            this model. Use 'max_completion_tokens' instead.``

        Covers reasoning families the static
        ``_MAX_COMPLETION_TOKENS_PREFIXES`` table doesn't know yet.
        """
        if "max_tokens" not in kwargs:
            return None
        msg = str(getattr(exc, "message", "") or exc).lower()
        if "max_tokens" not in msg:
            return None
        if "max_completion_tokens" not in msg and "not supported" not in msg:
            return None
        retry = dict(kwargs)
        retry["max_completion_tokens"] = retry.pop("max_tokens")
        return retry

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        raw = await self._invoke_with_heal(
            client.chat.completions.create,
            kwargs,
            purpose=purpose or "chat.completions.create",
        )
        return self._parse_response(raw)

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        # Without this flag the Chat Completions stream sends NO usage
        # chunk at all — the harvesting branch below ran for months while
        # every streamed call aggregated $0 (audit §2.5: CostBudgetGuard
        # and both hosts' cost displays silently neutralized). The usage
        # arrives as a final chunk with empty ``choices``.
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        model = request.model
        finish_reason = ""
        usage_data: Optional[Any] = None

        # The SDK validates the request and raises the 400 at ``create()``
        # time even with ``stream=True`` — so the retry-on-heal wrapper
        # can guard stream setup exactly like the non-streaming path.
        stream = await self._invoke_with_heal(
            client.chat.completions.create,
            kwargs,
            purpose=purpose or "chat.completions.create(stream)",
        )

        try:
            async for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, "model") and chunk.model:
                    model = chunk.model

                if delta and delta.content:
                    accumulated_content += delta.content
                    yield {"type": "text_delta", "text": delta.content}

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = accumulated_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

        except Exception as e:
            raise self._classify_error(e) from e

        blocks: List[ContentBlock] = []
        if accumulated_content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=accumulated_content,
                    raw={"type": "text", "text": accumulated_content},
                )
            )
        for tc in accumulated_tool_calls.values():
            tool_input = self._parse_tool_arguments(tc["arguments"])
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["id"],
                    tool_name=tc["name"],
                    tool_input=tool_input,
                    raw={
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tool_input,
                    },
                )
            )

        response = APIResponse(
            content=blocks,
            stop_reason=normalize_stop_reason(finish_reason, "openai"),
            usage=self._parse_usage(usage_data),
            model=model,
            raw=self._provenance(),
        )
        yield {"type": "message_complete", "response": response}

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        """Canonical APIRequest → OpenAI Chat Completions kwargs."""
        messages = canonical_messages_to_openai(request.messages, request.system)

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if request.max_tokens:
            # Reasoning families reject the classic kwarg — see the
            # ``_MAX_COMPLETION_TOKENS_PREFIXES`` block at module top.
            # Sending the right name up front avoids burning a 400 +
            # retry on every single call to o-series / gpt-5 models.
            if _model_requires_max_completion_tokens(request.model):
                logger.debug(
                    "openai: model %r takes max_completion_tokens — sent %d via the renamed kwarg",
                    request.model,
                    request.max_tokens,
                )
                kwargs["max_completion_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = canonical_tools_to_openai(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = canonical_tool_choice_to_openai(request.tool_choice)

        if request.thinking:
            effort = canonical_thinking_to_openai(request.thinking)
            if effort:
                kwargs["reasoning_effort"] = effort

        return kwargs

    def _parse_response(self, raw: Any) -> APIResponse:
        choice = raw.choices[0]
        blocks: List[ContentBlock] = []

        if choice.message.content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=choice.message.content,
                    raw={"type": "text", "text": choice.message.content},
                )
            )

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_input = self._parse_tool_arguments(tc.function.arguments)
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tc.id,
                        tool_name=tc.function.name,
                        tool_input=tool_input,
                        raw={
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tool_input,
                        },
                    )
                )

        stop_reason = normalize_stop_reason(choice.finish_reason or "", "openai")

        # ``raw`` is the provenance channel — see ``BaseClient._provenance``.
        provenance = self._provenance()
        provenance["response"] = raw

        return APIResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=self._parse_usage(getattr(raw, "usage", None)),
            model=raw.model,
            message_id=raw.id,
            raw=provenance,
        )

    def _parse_tool_arguments(self, raw: Any) -> Any:
        """Parse a tool-call ``arguments`` JSON string into Python.

        Default: strict ``json.loads`` with a ``{}`` fallback — the OpenAI
        SDK emits well-formed JSON, so nothing fancier is warranted. The
        OpenAI-compatible *local* clients (Ollama / LM Studio / custom)
        override this to repair the malformed JSON that local servers
        commonly emit (trailing commas, ``None``/``True`` literals,
        markdown fences) before giving up. Shared by the streaming and
        non-streaming parse paths so the two can't drift.
        """
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _parse_usage(self, usage_data: Any) -> TokenUsage:
        """OpenAI usage object → canonical :class:`TokenUsage`.

        Shared by the streaming and non-streaming paths so the cache
        accounting can't drift between them again. Maps
        ``prompt_tokens_details.cached_tokens`` (OpenAI's automatic
        prompt-cache hit counter) onto ``cache_read_input_tokens`` —
        same semantics as Anthropic's field. NOTE: unlike Anthropic,
        OpenAI's ``prompt_tokens`` already *includes* the cached
        portion; pricing code discounts cache reads, it must not add
        them on top. ``prompt_tokens_details`` may be an object (openai
        SDK) or a plain dict (vLLM and other compatible servers).
        """
        if usage_data is None:
            return TokenUsage()
        details = getattr(usage_data, "prompt_tokens_details", None)
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0) or 0
        else:
            cached = getattr(details, "cached_tokens", 0) or 0
        return TokenUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            cache_read_input_tokens=cached,
        )

    def _classify_error(self, e: Exception) -> APIError:
        try:
            import openai
        except ImportError:
            return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)

        if isinstance(e, openai.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, openai.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, openai.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, openai.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, openai.BadRequestError):
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, status_code=400, cause=e)
        if isinstance(e, openai.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
