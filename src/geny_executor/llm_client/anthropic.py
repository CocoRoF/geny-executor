"""Anthropic Messages API client.

Near-verbatim port of the former :class:`AnthropicProvider` in
``stages/s06_api/artifact/default/providers.py``, restructured to
inherit from :class:`BaseClient` and expose a :class:`ClientCapabilities`
profile.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators import canonical_messages_to_anthropic
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


logger = logging.getLogger(__name__)


# ── Alias resolution ────────────────────────────────────────────────
#
# The Anthropic Messages API only accepts canonical model IDs
# (``claude-opus-4-7``, ``claude-sonnet-4-6``, ``claude-haiku-4-5-…``);
# short aliases like ``opus`` / ``sonnet`` / ``haiku`` are only valid
# on the ``claude`` CLI binary surface, not on the HTTP API. Apps
# that share a model config between the CLI and HTTP paths (geny,
# anyone wrapping us) routinely tripped on this: the env stores
# ``opus`` from the CLI flow, the next session pins ``anthropic`` as
# its Stage 6 provider, and the API returns
# ``404 model: opus``.
#
# Resolve the well-known aliases to today's tier-leader canonical IDs
# right before the SDK call. Pinned to specific versions on purpose —
# silently floating an env's model id across releases would be a
# nasty surprise. Bump the right-hand side here when shipping a new
# default tier leader.
_ANTHROPIC_MODEL_ALIASES: Dict[str, str] = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def _resolve_anthropic_model(model: str) -> str:
    """Return the canonical ID for a known short alias, otherwise the
    input unchanged. Pure function — easy to unit-test in isolation."""
    canonical = _ANTHROPIC_MODEL_ALIASES.get(model)
    if canonical is None:
        return model
    if canonical != model:
        logger.info(
            "anthropic: model alias %r resolved to canonical %r",
            model, canonical,
        )
    return canonical


# ── Extended-thinking sampling-param compatibility ──────────────────
#
# The Anthropic Messages API rejects ``temperature``, ``top_p`` and
# ``top_k`` when extended thinking is enabled — the sampler is fixed
# by the thinking machinery. The error reads
# ``temperature is deprecated for this model`` (despite being model-
# agnostic when ``thinking`` is set).
#
# Drop the offending fields at the boundary. Logged at INFO so an
# operator who explicitly chose a temperature can see why it was
# silently ignored.
_THINKING_INCOMPATIBLE_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
)


# ── Models that reject sampling params unconditionally ──────────────
#
# Some models (currently the Opus 4.7 family — the only one verified
# against the live API in 2.1.2) reject ``temperature`` regardless of
# whether ``thinking`` is set. The error reads
# ``temperature is deprecated for this model.`` from
# ``api.anthropic.com``. The model is designed around fixed-sampler
# inference; the sampling kwargs become noise the API explicitly
# refuses.
#
# The set is keyed by the **resolved** canonical ID (so aliases get
# expanded first, see ``_resolve_anthropic_model``). Match is
# prefix-based — ``"claude-opus-4-7"`` covers any future
# ``claude-opus-4-7-20yyyymmdd`` pinned variant without needing an
# update here.
#
# AdaptiveModelRouter auto-promotes to Opus when ``thinking_enabled``
# is True (see ``stages/s06_api/artifact/default/router.py``), so an
# env that never sees Opus in its config can still hit this code
# path indirectly. The drop has to live at the boundary, not the
# router.
_TEMPERATURE_DEPRECATED_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
)


def _model_rejects_sampling_params(model: str) -> bool:
    """True iff ``model`` (canonical ID) belongs to a family that
    unconditionally rejects ``temperature``/``top_p``/``top_k``."""
    return any(
        model.startswith(prefix) for prefix in _TEMPERATURE_DEPRECATED_PREFIXES
    )


# ── Last-line retry on a deprecation 400 ────────────────────────────
#
# Future Anthropic releases will deprecate more sampling params for
# more models; the static prefix list above will go stale. When the
# API surfaces the deprecation error we strip the offending field
# and retry once. Captures the same exact 400 strings Anthropic emits
# (sometimes wrapped in backticks, sometimes not).
_DEPRECATION_MSG_TO_KWARG_KEY: Dict[str, str] = {
    "temperature is deprecated": "temperature",
    "`temperature` is deprecated": "temperature",
    "top_p is deprecated": "top_p",
    "`top_p` is deprecated": "top_p",
    "top_k is deprecated": "top_k",
    "`top_k` is deprecated": "top_k",
}


def _retry_kwargs_after_deprecation(
    kwargs: Dict[str, Any], exc: BaseException,
) -> Optional[Dict[str, Any]]:
    """If ``exc`` is the Anthropic deprecation 400 for a sampling
    field we recognise, return a copy of ``kwargs`` with that field
    removed. ``None`` means *don't retry* — let the caller re-raise.

    Defends against future model deprecations the static prefix list
    in ``_TEMPERATURE_DEPRECATED_PREFIXES`` doesn't know about yet.
    Only retries once per send (the caller guarantees this by not
    calling us recursively); if the retry also 400s the outer
    handler classifies and raises.
    """
    msg = str(getattr(exc, "message", "") or exc)
    msg_lower = msg.lower()
    for needle, key in _DEPRECATION_MSG_TO_KWARG_KEY.items():
        if needle in msg_lower and key in kwargs:
            retry = dict(kwargs)
            retry.pop(key, None)
            return retry
    return None


class AnthropicClient(BaseClient):
    """Real Anthropic API client using the official SDK."""

    provider = "anthropic"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
        supports_system_prompt=True,
        supports_structured_output=False,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=False,
        requires_workspace=False,
        streaming_granularity="token",
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
            import anthropic

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        try:
            raw_response = await client.messages.create(**kwargs)
            return self._parse_response(raw_response)
        except Exception as e:
            # Retry-on-deprecation safety net. The static prefix list
            # in ``_TEMPERATURE_DEPRECATED_PREFIXES`` will go stale as
            # Anthropic deprecates more sampling params for more
            # models. When the API explicitly tells us a sampling
            # param is the problem, strip it and retry once. Beats
            # a hard error on a model whose prefix we don't know yet.
            retry_kwargs = _retry_kwargs_after_deprecation(kwargs, e)
            if retry_kwargs is not None:
                logger.info(
                    "anthropic: retrying %s after deprecation 400 with "
                    "%r dropped (model=%r)",
                    purpose or "messages.create",
                    sorted(set(kwargs) - set(retry_kwargs)),
                    retry_kwargs.get("model"),
                )
                try:
                    raw_response = await client.messages.create(**retry_kwargs)
                    return self._parse_response(raw_response)
                except Exception as inner:
                    raise self._classify_error(inner) from inner
            raise self._classify_error(e) from e

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
        """Streaming call via the SDK's high-level ``messages.stream()`` helper.

        NOTE: do not pass ``stream=True`` in kwargs — that helper handles it.
        """
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

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield {"type": "text_delta", "text": text}

                final = await stream.get_final_message()
                yield {
                    "type": "message_complete",
                    "response": self._parse_response(final),
                }
        except Exception as e:
            # Same retry-on-deprecation safety net as ``_send``. The
            # SDK validates kwargs eagerly inside the ``stream``
            # context manager, so the deprecation 400 surfaces before
            # any tokens reach the caller — safe to retry once with
            # the offending field dropped.
            retry_kwargs = _retry_kwargs_after_deprecation(kwargs, e)
            if retry_kwargs is not None:
                logger.info(
                    "anthropic: retrying %s after deprecation 400 with "
                    "%r dropped (model=%r)",
                    purpose or "messages.stream",
                    sorted(set(kwargs) - set(retry_kwargs)),
                    retry_kwargs.get("model"),
                )
                try:
                    async with client.messages.stream(**retry_kwargs) as stream:
                        async for text in stream.text_stream:
                            yield {"type": "text_delta", "text": text}
                        final = await stream.get_final_message()
                        yield {
                            "type": "message_complete",
                            "response": self._parse_response(final),
                        }
                    return
                except Exception as inner:
                    raise self._classify_error(inner) from inner
            raise self._classify_error(e) from e

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        # Strip executor-internal keys (e.g. ``_meta`` on image blocks added by
        # the s01 normalizer for downstream provenance) and lower unsupported
        # block types (``file``) into safe fallbacks. Without this the
        # Anthropic Messages API rejects requests with
        # ``messages.0.content.0.image._meta: Extra inputs are not permitted``.
        sanitized_messages = canonical_messages_to_anthropic(request.messages)

        # Alias resolution — see ``_ANTHROPIC_MODEL_ALIASES`` docstring.
        # Pure function; no SDK call yet, so this is cheap.
        resolved_model = _resolve_anthropic_model(request.model)

        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": sanitized_messages,
            "max_tokens": request.max_tokens,
        }

        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            kwargs["top_k"] = request.top_k
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        if request.thinking:
            kwargs["thinking"] = request.thinking
        if request.metadata:
            kwargs["metadata"] = request.metadata

        # Extended-thinking sampling-param compatibility — see the
        # ``_THINKING_INCOMPATIBLE_SAMPLING_KEYS`` block at module top.
        # Anthropic rejects ``temperature``/``top_p``/``top_k`` when
        # ``thinking`` is set; drop them silently at the boundary so
        # an env with both ``thinking_enabled=True`` and an explicit
        # ``temperature`` (the common combo Geny ships) still works.
        if "thinking" in kwargs:
            for key in _THINKING_INCOMPATIBLE_SAMPLING_KEYS:
                if key in kwargs:
                    dropped = kwargs.pop(key)
                    logger.info(
                        "anthropic: dropped %r=%r — extended thinking "
                        "is enabled and the Messages API rejects this "
                        "sampling param",
                        key, dropped,
                    )

        # Model-level unconditional rejection — see
        # ``_TEMPERATURE_DEPRECATED_PREFIXES`` at module top. Opus 4.7
        # refuses ``temperature`` regardless of whether ``thinking`` is
        # set; without this drop, ``AdaptiveModelRouter`` promoting a
        # thinking-enabled call to Opus 4.7 would still 400.
        if _model_rejects_sampling_params(resolved_model):
            for key in _THINKING_INCOMPATIBLE_SAMPLING_KEYS:
                if key in kwargs:
                    dropped = kwargs.pop(key)
                    logger.info(
                        "anthropic: dropped %r=%r — model %r refuses "
                        "this sampling param unconditionally",
                        key, dropped, resolved_model,
                    )

        return kwargs

    def _parse_response(self, raw: Any) -> APIResponse:
        content_blocks: List[ContentBlock] = []

        for block in raw.content:
            if block.type == "text":
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=block.text,
                        raw={"type": "text", "text": block.text},
                    )
                )
            elif block.type == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                        raw={
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        },
                    )
                )
            elif block.type == "thinking":
                content_blocks.append(
                    ContentBlock(
                        type="thinking",
                        thinking_text=block.thinking,
                        raw={"type": "thinking", "thinking": block.thinking},
                    )
                )

        if raw.usage:
            usage = TokenUsage(
                input_tokens=getattr(raw.usage, "input_tokens", 0),
                output_tokens=getattr(raw.usage, "output_tokens", 0),
                cache_creation_input_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0),
                cache_read_input_tokens=getattr(raw.usage, "cache_read_input_tokens", 0),
            )
        else:
            usage = TokenUsage()

        return APIResponse(
            content=content_blocks,
            stop_reason=raw.stop_reason or "",
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=raw,
        )

    def _classify_error(self, e: Exception) -> APIError:
        import anthropic

        if isinstance(e, anthropic.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, anthropic.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, anthropic.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, anthropic.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, anthropic.BadRequestError):
            msg = str(e).lower()
            if "token" in msg or "context" in msg:
                return APIError(
                    str(e), category=ErrorCategory.TOKEN_LIMIT, status_code=400, cause=e
                )
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, status_code=400, cause=e)
        if isinstance(e, anthropic.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
