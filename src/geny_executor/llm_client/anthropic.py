"""Anthropic Messages API client.

Near-verbatim port of the former :class:`AnthropicProvider` in
``stages/s06_api/artifact/default/providers.py``, restructured to
inherit from :class:`BaseClient` and expose a :class:`ClientCapabilities`
profile.
"""

from __future__ import annotations

import logging
import re
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


# ── ``thinking.type=enabled`` → ``adaptive`` migration ──────────────
#
# Opus 4.7 (verified 2026-06-04) rejects ``thinking.type=enabled``
# and requires ``thinking.type=adaptive`` instead — the old
# enabled-with-fixed-budget shape isn't valid for the new generation
# of thinking-native models. The API error reads:
#
#   ``"thinking.type.enabled" is not supported for this model.
#     Use "thinking.type.adaptive" and "output_config.effort"``.
#
# Under ``adaptive``, the model picks its own budget; the legacy
# ``budget_tokens`` field is rejected as an extra input
# (``thinking.adaptive.budget_tokens: Extra inputs are not permitted``).
# Effort is *optional* — calls with bare ``{"type":"adaptive"}`` work.
# Translate at the boundary so callers that ship the v1 thinking
# shape continue to work against v2 models.
_THINKING_ADAPTIVE_ONLY_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
)


def _model_requires_adaptive_thinking(model: str) -> bool:
    """True iff ``model`` only accepts ``thinking.type=adaptive``."""
    return any(
        model.startswith(prefix) for prefix in _THINKING_ADAPTIVE_ONLY_PREFIXES
    )


def _translate_thinking_to_adaptive(thinking: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v1 (``type=enabled``) thinking dict to the v2
    (``type=adaptive``) shape Opus 4.7 demands.

      * ``type`` flips to ``"adaptive"``.
      * ``budget_tokens`` is dropped — the API rejects it under
        adaptive (``thinking.adaptive.budget_tokens: Extra inputs are
        not permitted``).
      * Any other unrelated keys (``display`` etc.) pass through.

    The kwarg ``output_config.effort`` is *not* added — bare
    ``{"type":"adaptive"}`` works against the live API; the API
    picks a default effort. Hosts that want to pin effort can do so
    explicitly by setting it on ``model_config`` and threading it
    through future plumbing.
    """
    out = dict(thinking)
    out["type"] = "adaptive"
    out.pop("budget_tokens", None)
    return out


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
    """If ``exc`` is an Anthropic 400 we can self-heal, return a
    rebuilt kwargs. ``None`` means *don't retry* — let the caller
    re-raise.

    Two recognised classes today:

      1. **Sampling-param deprecation** — the API message names a
         specific field (``temperature``, ``top_p``, ``top_k``) as
         deprecated. Strip the field and retry.
      2. **Thinking v1 → v2 migration** — the API rejects
         ``thinking.type=enabled`` and asks for
         ``thinking.type.adaptive``. Translate via
         :func:`_translate_thinking_to_adaptive` and retry.

    Defends against future model rollouts our static prefix lists
    don't know about yet. Caller guarantees one retry per send (we
    never recurse); a retry that also 400s gets classified + raised
    by the outer handler.
    """
    msg = str(getattr(exc, "message", "") or exc)
    msg_lower = msg.lower()

    # Class 2 — thinking v1→v2 migration. Run first because the
    # diagnostic is structural (the request shape, not just one
    # missing field).
    if (
        "thinking.type.enabled" in msg_lower
        or "thinking.type.adaptive" in msg_lower
    ) and isinstance(kwargs.get("thinking"), dict):
        thinking = kwargs["thinking"]
        if thinking.get("type") == "enabled":
            retry = dict(kwargs)
            retry["thinking"] = _translate_thinking_to_adaptive(thinking)
            return retry

    # Class 1 — sampling-param deprecation.
    for needle, key in _DEPRECATION_MSG_TO_KWARG_KEY.items():
        if needle in msg_lower and key in kwargs:
            retry = dict(kwargs)
            retry.pop(key, None)
            return retry
    return None


# ── 400 disambiguation: TOKEN_LIMIT vs BAD_REQUEST ──────────────────
#
# The 2.1.x heuristic was ``'token' in msg or 'context' in msg`` — which
# routed *param-shape* 400s into TOKEN_LIMIT. Concretely: the drift
# message this very module documents
# (``thinking.adaptive.budget_tokens: Extra inputs are not permitted``)
# contains the substring ``token``, so the next thinking-shape drift
# would have been diagnosed as "reduce your context" instead of "your
# request shape is stale" (audit §3.4). The two categories drive very
# different recovery: TOKEN_LIMIT tells s06 retry logic to compact the
# conversation; BAD_REQUEST is fatal and should bubble immediately.
#
# Classify TOKEN_LIMIT only on phrases the live API actually anchors
# overflow errors with, and short-circuit anything that *looks like a
# request-validation message* (a named param path or a deprecation
# notice) into BAD_REQUEST first.
_TOKEN_LIMIT_ANCHORS: tuple[str, ...] = (
    "maximum context length",
    "prompt is too long",
    "input length exceeds",
    "too many tokens",
    # The live API's combined-budget phrasing:
    # ``input length and `max_tokens` exceed context limit: …``
    "exceed context limit",
)

#: Substrings that mark a request-*validation* 400 (pydantic-style param
#: rejection or a deprecation notice) — never a context overflow.
_PARAM_SHAPE_NEEDLES: tuple[str, ...] = (
    ": extra inputs",
    "extra inputs are not permitted",
    "is deprecated",
)

#: A dotted parameter path (``thinking.adaptive.budget_tokens:``,
#: ``messages.0.content.0.image._meta:``) followed by a colon — the shape
#: the API uses to name the offending field in validation errors.
_PARAM_PATH_RE = re.compile(r"\b[a-z0-9_]+(?:\.[a-z0-9_\[\]]+)+\s*:")


def _classify_bad_request_message(msg_lower: str) -> ErrorCategory:
    """Pure classifier for an Anthropic ``BadRequestError`` message.

    Param-shape evidence wins over token-limit anchors on purpose: a
    validation message that happens to mention budget *tokens* must not
    be mistaken for an overflow (the misdiagnosis the audit flagged).
    """
    if any(needle in msg_lower for needle in _PARAM_SHAPE_NEEDLES):
        return ErrorCategory.BAD_REQUEST
    if _PARAM_PATH_RE.search(msg_lower):
        return ErrorCategory.BAD_REQUEST
    if any(anchor in msg_lower for anchor in _TOKEN_LIMIT_ANCHORS):
        return ErrorCategory.TOKEN_LIMIT
    return ErrorCategory.BAD_REQUEST


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

    _sdk_module = "anthropic"

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

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Route the 2.1.2/2.1.3 deprecation safety net through the
        :class:`BaseClient` heal hook.

        The mechanism (rebuild kwargs from the 400 message, retry once)
        was born here and got promoted to ``BaseClient`` in 2.2.0 so
        OpenAI/Google grow the same reflex; the Anthropic-specific
        needle table stays in :func:`_retry_kwargs_after_deprecation`
        (also kept module-level — the 2.1.x boundary tests pin it).
        """
        return _retry_kwargs_after_deprecation(kwargs, exc)

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        # Retry-on-heal safety net (``_heal_request_kwargs``). The static
        # prefix list in ``_TEMPERATURE_DEPRECATED_PREFIXES`` will go
        # stale as Anthropic deprecates more sampling params for more
        # models. When the API explicitly tells us a sampling param is
        # the problem, strip it and retry once. Beats a hard error on a
        # model whose prefix we don't know yet.
        raw_response = await self._invoke_with_heal(
            client.messages.create, kwargs, purpose=purpose or "messages.create"
        )
        return self._parse_response(raw_response)

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
            # Same retry-on-heal safety net as ``_send``, routed through
            # the BaseClient hook. The SDK validates kwargs eagerly
            # inside the ``stream`` context manager, so the heal-able
            # 400 surfaces before any tokens reach the caller — safe to
            # retry once with the rebuilt kwargs. (Generators can't use
            # ``_invoke_with_heal`` directly; the structure stays
            # hand-rolled, the policy is shared.)
            retry_kwargs = self._heal_request_kwargs(kwargs, e)
            if retry_kwargs is not None:
                try:
                    async with client.messages.stream(**retry_kwargs) as stream:
                        async for text in stream.text_stream:
                            yield {"type": "text_delta", "text": text}
                        final = await stream.get_final_message()
                        self._report_drift_healed(
                            kwargs, retry_kwargs, e,
                            purpose=purpose or "messages.stream",
                        )
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

        # Thinking-shape migration — see
        # ``_THINKING_ADAPTIVE_ONLY_PREFIXES`` at module top. Opus 4.7
        # rejects ``thinking.type=enabled``; flip to ``adaptive`` (and
        # drop the now-invalid ``budget_tokens``) at the boundary so
        # ``ModelConfig`` callers that still emit the v1 shape
        # continue to work against v2 models.
        if (
            "thinking" in kwargs
            and isinstance(kwargs["thinking"], dict)
            and kwargs["thinking"].get("type") == "enabled"
            and _model_requires_adaptive_thinking(resolved_model)
        ):
            before = kwargs["thinking"]
            kwargs["thinking"] = _translate_thinking_to_adaptive(before)
            logger.info(
                "anthropic: translated thinking.type=enabled → adaptive "
                "(model=%r, dropped legacy budget_tokens=%r)",
                resolved_model, before.get("budget_tokens"),
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

        # ``raw`` is the provenance channel (the CLI client already ships
        # a dict here with ``cli_version``). ``response`` carries the SDK
        # object for callers that need vendor-specific fields.
        provenance = self._provenance()
        provenance["response"] = raw

        return APIResponse(
            content=content_blocks,
            stop_reason=raw.stop_reason or "",
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=provenance,
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
            # TOKEN_LIMIT only on anchored overflow phrases; param-shape
            # validation messages (which routinely mention "tokens") are
            # BAD_REQUEST — see ``_classify_bad_request_message``.
            category = _classify_bad_request_message(str(e).lower())
            return APIError(str(e), category=category, status_code=400, cause=e)
        if isinstance(e, anthropic.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
