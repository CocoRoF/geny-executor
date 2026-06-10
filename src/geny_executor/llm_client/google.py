"""Google Gemini API client.

Ported from the former :class:`GoogleProvider` in
``stages/s06_api/artifact/google/providers.py``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators import (
    blocks_to_text,
    canonical_messages_to_google,
    canonical_thinking_to_google,
    canonical_tool_choice_to_google,
    canonical_tools_to_google,
    normalize_stop_reason,
)
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


class GoogleClient(BaseClient):
    """Google Gemini generateContent API client.

    Requires: ``pip install geny-executor[google]``
    """

    provider = "google"
    _sdk_module = "google.genai"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
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
        drops=("thinking_enabled",),
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
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "Google client requires the 'google-genai' package. "
                    "Install with: pip install geny-executor[google]"
                ) from e
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        try:
            raw = await client.aio.models.generate_content(**kwargs)
            return self._parse_response(raw, request.model)
        except Exception as e:
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

        accumulated_text = ""
        accumulated_blocks: List[ContentBlock] = []
        finish_reason = ""
        usage_data: Optional[Any] = None

        try:
            async for chunk in await client.aio.models.generate_content_stream(**kwargs):
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage_data = chunk.usage_metadata

                if not hasattr(candidate, "content") or not candidate.content:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        is_thought = getattr(part, "thought", False)
                        if is_thought:
                            accumulated_blocks.append(
                                ContentBlock(type="thinking", thinking_text=part.text)
                            )
                        else:
                            accumulated_text += part.text
                            yield {"type": "text_delta", "text": part.text}

                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        fc_id = getattr(fc, "id", "") or ""
                        fc_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                        accumulated_blocks.append(
                            ContentBlock(
                                type="tool_use",
                                tool_use_id=fc_id,
                                tool_name=fc.name,
                                tool_input=fc_args,
                                raw={
                                    "type": "tool_use",
                                    "id": fc_id,
                                    "name": fc.name,
                                    "input": fc_args,
                                },
                            )
                        )

        except Exception as e:
            raise self._classify_error(e) from e

        blocks: List[ContentBlock] = []
        if accumulated_text:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=accumulated_text,
                    raw={"type": "text", "text": accumulated_text},
                )
            )
        blocks.extend(accumulated_blocks)

        usage = self._parse_usage(usage_data)

        response = APIResponse(
            content=blocks,
            stop_reason=normalize_stop_reason(finish_reason, "google"),
            usage=usage,
            model=request.model,
            raw=self._provenance(),
        )
        yield {"type": "message_complete", "response": response}

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        contents = canonical_messages_to_google(request.messages)

        config: Dict[str, Any] = {}
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.top_k is not None:
            config["top_k"] = request.top_k
        if request.stop_sequences:
            config["stop_sequences"] = request.stop_sequences

        if request.thinking:
            thinking_config = canonical_thinking_to_google(request.thinking)
            if thinking_config:
                config["thinking_config"] = thinking_config

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "contents": contents,
        }
        if config:
            kwargs["config"] = config

        if request.system:
            sys_text = blocks_to_text(request.system)
            if sys_text:
                kwargs["config"] = kwargs.get("config", {})
                kwargs["config"]["system_instruction"] = sys_text

        if request.tools:
            kwargs["config"] = kwargs.get("config", {})
            kwargs["config"]["tools"] = canonical_tools_to_google(request.tools)
        if request.tool_choice:
            kwargs["config"] = kwargs.get("config", {})
            kwargs["config"]["tool_config"] = canonical_tool_choice_to_google(request.tool_choice)

        return kwargs

    def _parse_response(self, raw: Any, model: str) -> APIResponse:
        # ``raw`` on the canonical response is the provenance channel —
        # see ``BaseClient._provenance``. The SDK object rides along
        # under ``response``.
        provenance = self._provenance()
        provenance["response"] = raw

        if not raw.candidates:
            return APIResponse(
                content=[ContentBlock(type="text", text="")],
                stop_reason="end_turn",
                usage=self._parse_usage(getattr(raw, "usage_metadata", None)),
                model=model,
                raw=provenance,
            )

        candidate = raw.candidates[0]
        blocks: List[ContentBlock] = []

        if hasattr(candidate, "content") and candidate.content:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    is_thought = getattr(part, "thought", False)
                    if is_thought:
                        blocks.append(ContentBlock(type="thinking", thinking_text=part.text))
                    else:
                        blocks.append(
                            ContentBlock(
                                type="text",
                                text=part.text,
                                raw={"type": "text", "text": part.text},
                            )
                        )
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    fc_id = getattr(fc, "id", "") or ""
                    fc_args = dict(fc.args) if hasattr(fc, "args") and fc.args else {}
                    blocks.append(
                        ContentBlock(
                            type="tool_use",
                            tool_use_id=fc_id,
                            tool_name=fc.name,
                            tool_input=fc_args,
                            raw={
                                "type": "tool_use",
                                "id": fc_id,
                                "name": fc.name,
                                "input": fc_args,
                            },
                        )
                    )

        finish = str(getattr(candidate, "finish_reason", "STOP"))
        stop_reason = normalize_stop_reason(finish, "google")
        usage = self._parse_usage(getattr(raw, "usage_metadata", None))

        return APIResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=model,
            raw=provenance,
        )

    def _parse_usage(self, usage_meta: Any) -> TokenUsage:
        if usage_meta is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        )

    def _classify_error(self, e: Exception) -> APIError:
        """Vendor exception → canonical :class:`APIError`.

        2.1.x classified by substring over ``str(e)`` — which routed any
        message *containing* ``"400"`` (e.g. a 500 whose body echoes a
        nested code, or a model name with "400" in it) into BAD_REQUEST
        (audit §1-4: "next incident, first in line"). Typed checks first:

          1. ``google.genai.errors`` — what the installed SDK actually
             raises. ``APIError`` carries ``.code`` (HTTP status int) and
             ``.status`` (gRPC-style string like ``RESOURCE_EXHAUSTED``);
             ``ClientError``/``ServerError`` partition 4xx/5xx.
          2. ``google.api_core.exceptions`` — not a google-genai
             dependency, but present whenever Vertex/grpc extras are
             installed, and some transports surface those types instead.
             Import is guarded; absence is normal.
          3. The old substring heuristic survives as a genuinely-last
             resort for non-SDK exceptions (httpx transport errors,
             asyncio timeouts) that carry no structure at all.
        """
        if isinstance(e, APIError):
            return e

        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover — SDK is a hard dep of this client
            genai_errors = None

        if genai_errors is not None and isinstance(e, genai_errors.APIError):
            code = getattr(e, "code", None)
            status = str(getattr(e, "status", "") or "").upper()

            if code == 429 or status == "RESOURCE_EXHAUSTED":
                return APIError(
                    str(e), category=ErrorCategory.RATE_LIMITED, status_code=code, cause=e
                )
            if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                return APIError(
                    str(e), category=ErrorCategory.AUTH, status_code=code, cause=e
                )
            if code == 504 or status == "DEADLINE_EXCEEDED":
                return APIError(
                    str(e), category=ErrorCategory.TIMEOUT, status_code=code, cause=e
                )
            if code == 503 or status == "UNAVAILABLE":
                return APIError(
                    str(e), category=ErrorCategory.SERVER_ERROR, status_code=code, cause=e
                )
            if code == 400 or status in ("INVALID_ARGUMENT", "FAILED_PRECONDITION"):
                return APIError(
                    str(e), category=ErrorCategory.BAD_REQUEST, status_code=code, cause=e
                )
            # Partition by class/range when the specific code is novel.
            if isinstance(e, genai_errors.ServerError) or (
                isinstance(code, int) and 500 <= code < 600
            ):
                return APIError(
                    str(e), category=ErrorCategory.SERVER_ERROR, status_code=code, cause=e
                )
            if isinstance(e, genai_errors.ClientError) or (
                isinstance(code, int) and 400 <= code < 500
            ):
                return APIError(
                    str(e), category=ErrorCategory.BAD_REQUEST, status_code=code, cause=e
                )

        api_core_category = self._classify_api_core(e)
        if api_core_category is not None:
            return APIError(str(e), category=api_core_category, cause=e)

        return self._classify_by_message(e)

    @staticmethod
    def _classify_api_core(e: Exception) -> Optional[ErrorCategory]:
        """isinstance chain over ``google.api_core`` typed exceptions.

        Returns ``None`` when the package isn't installed or the
        exception isn't one of its types.
        """
        try:
            from google.api_core import exceptions as gac_exceptions
        except ImportError:
            return None

        if isinstance(e, gac_exceptions.ResourceExhausted):
            return ErrorCategory.RATE_LIMITED
        if isinstance(
            e, (gac_exceptions.Unauthenticated, gac_exceptions.PermissionDenied)
        ):
            return ErrorCategory.AUTH
        if isinstance(e, gac_exceptions.DeadlineExceeded):
            return ErrorCategory.TIMEOUT
        if isinstance(e, gac_exceptions.InvalidArgument):
            return ErrorCategory.BAD_REQUEST
        if isinstance(
            e, (gac_exceptions.ServiceUnavailable, gac_exceptions.InternalServerError)
        ):
            return ErrorCategory.SERVER_ERROR
        return None

    def _classify_by_message(self, e: Exception) -> APIError:
        """Genuinely-last-resort substring classification.

        Only reached for exceptions with no usable type information.
        Server-side checks run before the ``400`` needle so a 5xx whose
        body happens to echo "400" no longer lands in BAD_REQUEST.
        """
        error_str = str(e).lower()

        if "resource exhausted" in error_str or "429" in error_str:
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if "deadline exceeded" in error_str or "timeout" in error_str:
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if "unauthenticated" in error_str or "401" in error_str or "api key" in error_str:
            return APIError(str(e), category=ErrorCategory.AUTH, cause=e)
        if "unavailable" in error_str or "503" in error_str or "500" in error_str:
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, cause=e)
        if "invalid argument" in error_str or "400" in error_str:
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, cause=e)
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
