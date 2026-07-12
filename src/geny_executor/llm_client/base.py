"""Base class for every LLM client.

Implementations adapt a vendor SDK to the canonical :class:`APIRequest` /
:class:`APIResponse` shape. Every :class:`BaseClient` MUST:

- Accept a :class:`ModelConfig` + canonical messages and run the vendor
  call without the caller needing to know which vendor is in use.
- Drop unsupported fields rather than raising, emitting a
  ``llm_client.feature_unsupported`` event on ``event_sink`` if one was
  provided. Fields declared in ``capabilities.drops`` are additionally
  stripped + reported via ``llm_client.parameter_dropped`` (2.2.0 —
  the list was decorative through 2.1.x, audit §3.5).
- Translate vendor exceptions into
  :class:`geny_executor.core.errors.APIError` with a populated
  :class:`ErrorCategory` so upstream retry/classify logic does not need
  to branch on vendor.
- Self-heal vendor drift where the error names the problem: the
  ``_heal_request_kwargs`` hook + ``_invoke_with_heal`` wrapper retry a
  rebuilt request exactly once and report via
  ``llm_client.drift_healed`` + WARNING (2.2.0 — generalized from the
  Anthropic 2.1.2/2.1.3 deprecation net).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.types import APIRequest, APIResponse


logger = logging.getLogger(__name__)


#: Lazily resolved ``<package>.__version__`` strings, cached per package so
#: the provenance stamp on every response costs one import for the lifetime
#: of the process. All four 2.1.x boundary incidents were version skew and
#: the post-mortems had to reconstruct "which SDK was installed?" from pip
#: freeze archaeology — record it on the response instead.
_SDK_VERSION_CACHE: Dict[str, str] = {}


def _resolve_sdk_version(package: str) -> str:
    """Best-effort ``__version__`` lookup for an installed SDK module.

    Never raises — a missing or broken package degrades to ``"unknown"``
    (and the vendor call itself will surface the real ImportError with a
    much better message). Cached forever: SDK versions cannot change
    mid-process.
    """
    if not package:
        return "unknown"
    cached = _SDK_VERSION_CACHE.get(package)
    if cached is not None:
        return cached
    version = "unknown"
    try:
        import importlib

        module = importlib.import_module(package)
        version = str(getattr(module, "__version__", "unknown") or "unknown")
    except Exception:  # noqa: BLE001 — provenance must never fail a call
        version = "unknown"
    _SDK_VERSION_CACHE[package] = version
    return version


@dataclass(frozen=True)
class ClientCapabilities:
    """Feature flags a client advertises.

    Stage code inspects these before sending fields not every vendor
    supports. Unsupported fields are silently dropped and the client
    emits a ``llm_client.feature_unsupported`` event.
    """

    supports_thinking: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_tool_choice: bool = False
    supports_stop_sequences: bool = True
    supports_top_k: bool = False
    supports_system_prompt: bool = True

    # --- Extended capabilities (CLI backends + JSON schema + sessions) ---

    #: JSON-schema / json_object structured-output support.
    supports_structured_output: bool = False

    #: Vendor-side session id resume (e.g. claude --session-id / --resume).
    supports_session_continuity: bool = False

    #: Vendor accepts MCP server configuration passthrough.
    supports_mcp_passthrough: bool = False

    #: Vendor enforces a USD budget cap on the call (e.g. --max-budget-usd).
    supports_budget_limit: bool = False

    #: Token usage fields are populated on the response.
    supports_token_usage: bool = True

    #: Cost (usage.cost_usd) is populated on the response.
    supports_cost_usage: bool = False

    #: Implementation strategy hint — client spawns a subprocess.
    is_subprocess: bool = False

    #: Client requires a working directory / workspace path.
    requires_workspace: bool = False

    #: Streaming granularity: "token" | "message" | "none".
    streaming_granularity: str = "token"

    #: Fields this client drops when present on the request. As of 2.2.0
    #: this list is *authoritative*, not documentation: ``_build_request``
    #: strips every listed field from the outgoing request and emits one
    #: ``llm_client.parameter_dropped`` event per stripped field. The list
    #: spent 2.1.x as a decoy (declared, serialized, consumed by nothing —
    #: audit §3.5), which meant a manifest-pinned ``temperature`` on the
    #: CLI backend was ignored in total silence.
    drops: tuple[str, ...] = field(default=())

    def supports(self, feature: str) -> bool:
        """Lookup ``supports_<feature>`` flag by string name."""
        return bool(getattr(self, f"supports_{feature}", False))


class BaseClient(ABC):
    """Abstract LLM client. Concrete subclasses live in this package."""

    #: Provider name (stable identifier used by :class:`ClientRegistry`).
    provider: str = ""

    #: Capabilities advertised by this client. Subclasses override.
    capabilities: ClientCapabilities = ClientCapabilities()

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers
        self._event_sink = event_sink

    # ── High-level surface used by stages ───────────────────────────────

    async def create_message(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Send a non-streaming request built from a :class:`ModelConfig`.

        ``response_format`` is the canonical structured-output request
        (``{"type": "json_schema", "json_schema": {...}}`` or
        ``{"type": "json_object"}``). Clients that enforce it natively do
        (Claude Code CLI → ``--json-schema``); others carry it as an
        advisory field — callers should still validate/parse the reply.
        """
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            response_format=response_format,
        )
        return await self._send(request, purpose=purpose)

    async def create_message_stream(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming variant. Default: fall back to non-streaming.

        Concrete clients override to use vendor streams.
        """
        response = await self.create_message(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            purpose=purpose,
        )
        yield {"type": "message_complete", "response": response}

    # ── Low-level surface — kept for s06_api parity during PR-3→PR-4 bridge

    @abstractmethod
    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        """Send a pre-built :class:`APIRequest`. Subclass implements vendor call."""

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_request(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
        stream: bool,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> APIRequest:
        """Assemble a canonical :class:`APIRequest`.

        Emits ``llm_client.feature_unsupported`` events for any field in
        ``model_config`` that this client drops.
        """
        request = APIRequest(
            model=model_config.model,
            messages=list(messages),
            max_tokens=model_config.max_tokens,
            system=system,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k if self.capabilities.supports_top_k else None,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=(
                list(model_config.stop_sequences) if model_config.stop_sequences else None
            ),
            stream=stream,
        )
        if response_format:
            request.response_format = dict(response_format)
            if not self.capabilities.supports_structured_output:
                self._emit_unsupported("response_format")

        if model_config.thinking_enabled:
            if self.capabilities.supports_thinking:
                thinking: Dict[str, Any] = {"type": model_config.thinking_type}
                if model_config.thinking_type == "enabled":
                    thinking["budget_tokens"] = model_config.thinking_budget_tokens
                if model_config.thinking_display:
                    thinking["display"] = model_config.thinking_display
                request.thinking = thinking
            else:
                self._emit_unsupported("thinking_enabled")

        if model_config.top_k is not None and not self.capabilities.supports_top_k:
            self._emit_unsupported("top_k")

        if tool_choice and not self.capabilities.supports_tool_choice:
            self._emit_unsupported("tool_choice")

        # Stop-sequences negotiation — clients that drop stop_sequences
        # signal it explicitly. (Not silently honored by all CLI backends.)
        if model_config.stop_sequences and not self.capabilities.supports_stop_sequences:
            self._emit_unsupported("stop_sequences")
            request.stop_sequences = None

        self._apply_declared_drops(
            request,
            model_config=model_config,
            tools=tools,
            tool_choice=tool_choice,
        )

        return request

    # Maps a ``capabilities.drops`` entry to the APIRequest attribute that
    # carries it. Keyed by the *declaration* vocabulary (ModelConfig field
    # names — ``thinking_enabled``, not ``thinking``) because that is what
    # every shipped drops tuple already uses.
    _DROP_FIELD_TO_REQUEST_ATTR: Dict[str, str] = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_tokens": "max_tokens",
        "stop_sequences": "stop_sequences",
        "thinking_enabled": "thinking",
        "tools": "tools",
        "tool_choice": "tool_choice",
    }

    # Capability flag that overrides a declared drop. ``drops`` tuples are
    # written against a class's CONSERVATIVE defaults; instance-level
    # capability upgrades (``VLLMClient.configure_capabilities(
    # supports_tools=True)``) replace the flags but not the tuple, so the
    # 2.2.0 authoritative-drops enforcement read a stale declaration and
    # stripped fields the instance genuinely supports — a 2.1.x→2.2.0
    # regression (review B3). A drop is skipped when the instance flag
    # says the feature is supported; fields with no capability flag
    # (temperature / top_p / max_tokens) always honour the declaration.
    _DROP_FIELD_TO_CAPABILITY: Dict[str, str] = {
        "tools": "supports_tools",
        "tool_choice": "supports_tool_choice",
        "thinking_enabled": "supports_thinking",
        "top_k": "supports_top_k",
        "stop_sequences": "supports_stop_sequences",
    }

    def _apply_declared_drops(
        self,
        request: APIRequest,
        *,
        model_config: ModelConfig,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
    ) -> None:
        """Enforce ``capabilities.drops`` — strip + report, never silence.

        ``drops`` was dead metadata through 2.1.x (audit §3.5): the CLI
        backend declared it drops ``temperature``/``max_tokens`` and then
        nothing consumed the declaration, so an operator who pinned a
        temperature in the environment manifest saw a green check and no
        behaviour change. This converts the declaration into negotiation:
        every declared field that the caller actually supplied is stripped
        from the outgoing request and reported via one
        ``llm_client.parameter_dropped`` event carrying the discarded value.

        The pre-existing capability gates above (thinking/top_k/tool_choice/
        stop_sequences) keep emitting ``llm_client.feature_unsupported`` —
        hosts already key on those, and the two events answer different
        questions ("this client can't" vs "this value went nowhere"). Each
        field is stripped at most once and reported at most once per event
        type, so no double emission.

        Reads ``self.capabilities`` (not ``type(self).capabilities``) so
        instance-level upgrades — ``VLLMClient.configure_capabilities`` on a
        deployment whose model genuinely supports tools — can also amend the
        drops list without subclassing. And because those upgrades replace
        the capability FLAGS without rewriting the drops tuple, the
        effective drop set is capability-aware (review B3): a declared
        drop whose matching ``supports_*`` flag is True on the instance
        is skipped, so ``configure_capabilities(supports_tools=True)``
        restores tools exactly as its docstring promises.
        """
        declared = self.capabilities.drops
        if not declared:
            return

        sources: Dict[str, Any] = {
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "max_tokens": model_config.max_tokens,
            "stop_sequences": model_config.stop_sequences,
            "thinking_enabled": model_config.thinking_enabled,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        seen: set[str] = set()
        for field_name in declared:
            if field_name in seen:
                continue  # duplicate declaration — strip/report once
            seen.add(field_name)
            capability_flag = self._DROP_FIELD_TO_CAPABILITY.get(field_name)
            if capability_flag and getattr(self.capabilities, capability_flag, False):
                # The INSTANCE says it supports this feature — the drop
                # declaration is stale relative to a capability upgrade.
                continue
            attr = self._DROP_FIELD_TO_REQUEST_ATTR.get(field_name)
            if attr is None:
                # Unknown vocabulary (future capability name, typo in a
                # subclass). Nothing to strip on the request; stay quiet
                # rather than spam — the conformance suite is the place
                # that catches stale declarations.
                continue
            value = sources.get(field_name)
            setattr(request, attr, None)
            if self._drop_value_present(value):
                self._emit_parameter_dropped(field_name, value)

    @staticmethod
    def _drop_value_present(value: Any) -> bool:
        """Was the field meaningfully supplied? ``0.0`` temperature counts
        (it is an explicit sampling choice); ``False``/empty/None do not."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (list, tuple, dict, set)) and not value:
            return False
        return True

    def supports(self, feature: str) -> bool:
        """Capability lookup helper — proxy to ``self.capabilities.supports``."""
        return self.capabilities.supports(feature)

    def _emit_unsupported(self, field_name: str) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            {
                "type": "llm_client.feature_unsupported",
                "provider": self.provider,
                "field": field_name,
            }
        )

    def _emit_parameter_dropped(self, field_name: str, value: Any) -> None:
        """Report a value discarded by ``capabilities.drops`` negotiation."""
        if self._event_sink is None:
            return
        self._event_sink(
            {
                "type": "llm_client.parameter_dropped",
                "provider": self.provider,
                "field": field_name,
                "value": value,
            }
        )

    # ── Vendor-drift self-heal (generalized from AnthropicClient, 2.2.0) ─
    #
    # Every provider keeps a static compatibility table of some kind
    # (deprecated sampling params, renamed token-cap kwargs, …) and every
    # static table goes stale the day the vendor ships a new model family.
    # The 2.1.2–2.1.3 incidents established the pattern that works: when
    # the vendor 400 *names the problem*, rebuild the request and retry
    # exactly once. That mechanism was Anthropic-local; it now lives here
    # so OpenAI's ``max_tokens → max_completion_tokens`` rename (already
    # real in prod per the 2026-06-09 audit) gets the same safety net.

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Provider hook: given the vendor-call kwargs that just failed and
        the exception, return rebuilt kwargs to retry ONCE with, or ``None``
        to let the caller classify + re-raise.

        Contract for implementations:
          * Pure — never mutate ``kwargs``; return a fresh dict.
          * Conservative — only heal when the error message *names* the
            offending field/shape. A guess that retries a hopeless request
            doubles latency and cost on every failure.
          * Idempotent-safe — callers guarantee a single retry per send, so
            a heal whose retry also fails surfaces the second error.
        """
        return None

    async def _invoke_with_heal(
        self,
        vendor_call: Callable[..., Awaitable[Any]],
        kwargs: Dict[str, Any],
        *,
        purpose: str = "",
    ) -> Any:
        """Retry-once wrapper around an awaitable vendor call.

        Runs ``vendor_call(**kwargs)``; on failure consults
        :meth:`_heal_request_kwargs` and retries exactly once with the
        rebuilt kwargs. Both failure paths raise through
        :meth:`_classify_error` so callers keep the canonical
        ``APIError`` contract. A successful heal is reported via
        :meth:`_report_drift_healed` — loudly, because it means a static
        compatibility table is stale and the next deploy should fix it
        proactively instead of paying the extra round-trip forever.
        """
        try:
            return await vendor_call(**kwargs)
        except Exception as e:
            retry_kwargs = self._heal_request_kwargs(kwargs, e)
            if retry_kwargs is None:
                raise self._classify_error(e) from e
            try:
                result = await vendor_call(**retry_kwargs)
            except Exception as inner:
                raise self._classify_error(inner) from inner
            self._report_drift_healed(kwargs, retry_kwargs, e, purpose=purpose)
            return result

    def _report_drift_healed(
        self,
        kwargs: Dict[str, Any],
        retry_kwargs: Dict[str, Any],
        exc: BaseException,
        *,
        purpose: str = "",
    ) -> None:
        """Emit ``llm_client.drift_healed`` + a WARNING for a heal that the
        vendor accepted.

        WARNING, not INFO, on purpose: the retry masked the failure from
        the caller, and INFO is exactly how the 2.1.x masked-degradation
        incidents stayed invisible for weeks. Operators must learn that a
        static prefix/needle table is stale while the heal is still
        papering over it.
        """
        message = str(getattr(exc, "message", "") or exc)
        model = retry_kwargs.get("model", kwargs.get("model", ""))
        healed_fields = sorted(set(kwargs) - set(retry_kwargs))
        healed_fields += sorted(
            k for k in retry_kwargs if k in kwargs and retry_kwargs[k] != kwargs[k]
        )
        for field_name in healed_fields:
            logger.warning(
                "%s: request self-healed after vendor drift — %r rebuilt and "
                "retried once (model=%r, purpose=%r). The static "
                "compatibility tables are stale; original error: %s",
                self.provider,
                field_name,
                model,
                purpose,
                message,
            )
            if self._event_sink is not None:
                self._event_sink(
                    {
                        "type": "llm_client.drift_healed",
                        "provider": self.provider,
                        "model": model,
                        "field": field_name,
                        "message": message,
                    }
                )

    def _classify_error(self, e: Exception) -> APIError:
        """Translate a vendor exception into a canonical :class:`APIError`.

        Base fallback so :meth:`_invoke_with_heal` works for any subclass;
        SDK clients override with their vendor's typed exception chain.
        """
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)

    # ── Response provenance ──────────────────────────────────────────────

    #: Importable module whose ``__version__`` identifies the SDK speaking
    #: to this vendor (``"anthropic"``, ``"openai"``, ``"google.genai"``).
    #: Subclasses set it so :meth:`_provenance` can stamp responses.
    _sdk_module: str = ""

    def _provenance(self) -> Dict[str, Any]:
        """``{'provider': ..., 'sdk_version': ...}`` for ``APIResponse.raw``.

        Every 2.1.x boundary incident was version skew; this is the
        cheapest possible handshake — record which adapter + SDK produced
        the response so post-mortems stop guessing.
        """
        return {
            "provider": self.provider,
            "sdk_version": _resolve_sdk_version(self._sdk_module),
        }

    def configure(self, **kwargs: Any) -> None:
        """Apply provider-specific runtime configuration."""
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        """Best-effort pre-warm so turn 1 doesn't pay the cold start.

        TTFT program (2.50.0, findings C2/C3): the first call of a
        session pays whatever the backend defers — SDK client build,
        DNS + TCP + TLS to the vendor, the CLI's ``--version`` probe.
        Subclasses override to move that cost here (typically a cheap
        ``GET /models``); hosts call :meth:`Pipeline.warmup` right after
        session build, before the user's first message.

        Contract: never raises, returns False on failure, and leaves the
        client in the same logical state as before — warmup is purely an
        accelerator; a failed warmup means turn 1 behaves exactly as it
        does today.
        """
        del timeout_s
        return True
