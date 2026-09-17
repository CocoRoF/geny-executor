"""The OpenAI Responses wire, for everyone who speaks it.

Two clients talk to the Responses API: a ChatGPT/Codex subscription (OAuth
tokens, chatgpt.com) and an OpenAI API key (api.openai.com). Everything
between the canonical request and the canonical response is identical — the
input translation, the tool shape, the SSE event names, the reasoning block —
so it lives here once.

It matters that it is once. The reason this file exists is that the
Chat Completions client and this one had each been taught, separately, which
model families reject which parameter — and the third time that knowledge was
applied to only one of them, an agent on a correctly configured production
account could not take a single turn.

Why the Responses API at all: Chat Completions refuses function tools together
with reasoning on the gpt-5 family, and an agent always has tools. The
documented answer is either to give up reasoning (``reasoning_effort:
"none"``) or to use this wire. Giving up reasoning on every tool-using turn of
a reasoning agent is not a trade worth making.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client._failover import Notify, classify_text, strip_cache_markers
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.text_tool_protocol import system_text
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

#: Reasoning efforts the Responses API accepts.
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

RESPONSES_CAPABILITIES = ClientCapabilities(
    supports_thinking=True,
    supports_tools=True,
    supports_streaming=True,
    supports_tool_choice=False,
    supports_stop_sequences=False,
    supports_top_k=False,
    supports_system_prompt=True,
    supports_token_usage=True,
    streaming_granularity="token",
    drops=("temperature", "top_p", "top_k", "stop_sequences", "max_tokens", "tool_choice"),
)


# ── canonical → Responses ─────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _image_part(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    src = block.get("source") or {}
    if src.get("type") == "base64" and src.get("data"):
        return {"type": "input_image",
                "image_url": f"data:{src.get('media_type', 'image/png')};base64,{src['data']}"}
    if src.get("type") == "url" and src.get("url"):
        return {"type": "input_image", "image_url": src["url"]}
    return None


def _result_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str) if content is not None else ""


def to_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            if content:
                part = "output_text" if role == "assistant" else "input_text"
                items.append({"type": "message", "role": role if role in ("user", "assistant") else "user",
                              "content": [{"type": part, "text": content}]})
            continue
        parts: List[Dict[str, Any]] = []

        def flush() -> None:
            if parts:
                items.append({"type": "message", "role": "assistant" if role == "assistant" else "user",
                              "content": list(parts)})
                parts.clear()

        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and block.get("text"):
                parts.append({"type": "output_text" if role == "assistant" else "input_text",
                              "text": str(block["text"])})
            elif kind == "image" and role != "assistant":
                image = _image_part(block)
                if image:
                    parts.append(image)
            elif kind == "tool_use":
                flush()
                items.append({
                    "type": "function_call",
                    "call_id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                })
            elif kind == "tool_result":
                flush()
                items.append({
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id") or ""),
                    "output": _result_output(block.get("content")),
                })
        flush()
    return items


def to_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for tool in tools or []:
        if not tool.get("name"):
            continue
        out.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            "strict": False,
        })
    return out




class ResponsesClient(BaseClient):
    """Everything the Responses wire does, minus who you are.

    A subclass supplies the endpoint and the credentials — and, when the
    credentials can expire, a way to renew them mid-stream.
    """

    capabilities = RESPONSES_CAPABILITIES

    #: Where ``/responses`` lives.
    base_url_default = "https://api.openai.com/v1"
    #: Used in error messages, so a failure names the account it came from.
    label = "OpenAI"

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        *,
        account_id: str = "",
        account_label: str = "",
        effort: Optional[str] = None,
        timeout_s: float = 600.0,
        notify: Optional[Notify] = None,
        session_id: Optional[str] = None,
        event_sink: Any = None,
        transport: Any = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, default_headers=default_headers,
                         event_sink=event_sink)
        self._account_id = account_id
        self._account_label = account_label
        self._effort = effort if effort in EFFORTS else None
        self._timeout_s = float(timeout_s or 600.0)
        self._notify = notify
        self._session_id = session_id or uuid.uuid4().hex
        self._base = (base_url or self.base_url_default).rstrip("/")
        self._transport = transport

    # ── subclass surface ─────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        """Who this client is. Called before every attempt."""
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _renew(self) -> bool:
        """Refresh expiring credentials. ``False`` when there is nothing to
        renew — an API key never expires, so the base class says no."""
        return False

    async def _prepare(self) -> None:
        """Last chance before a request: raise when this client cannot work,
        and renew credentials that are about to expire."""
        if not self._api_key:
            raise APIError(f"{self.label}: no API key configured.", category=ErrorCategory.AUTH)

    # ── plumbing ─────────────────────────────────────────────────────

    def _notify_host(self, payload: Dict[str, Any]) -> None:
        if self._notify is None:
            return
        try:
            self._notify({"accountId": self._account_id, **payload})
        except Exception:  # noqa: BLE001 — a notification never breaks a turn
            pass

    def _client(self) -> httpx.AsyncClient:
        kwargs: Dict[str, Any] = {"timeout": httpx.Timeout(self._timeout_s, connect=30.0)}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(self._default_headers or {}),
            **self._auth_headers(),
        }
        return headers

    def _body(self, request: APIRequest) -> dict[str, Any]:
        instructions = system_text(strip_cache_markers(request.system)) or "You are a helpful assistant."
        tools = to_tools(request.tools)
        body: dict[str, Any] = {
            "model": request.model,
            "instructions": instructions,
            "input": to_input(strip_cache_markers(request.messages)),
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "geny_" + _sha256(
                self._session_id + "\0" + instructions),
        }
        if tools:
            body.update(tools=tools, tool_choice="auto", parallel_tool_calls=True)
        effort = self._effort_for(request)
        if effort:
            body["reasoning"] = {"effort": effort, "summary": "auto"}
        return body

    def _effort_for(self, request: APIRequest) -> Optional[str]:
        """How hard to think on this turn.

        The account's configured effort wins; otherwise a turn that asked for
        thinking at all gets the middle setting.
        """
        if self._effort:
            return self._effort
        thinking = request.thinking or {}
        if str(thinking.get("type", "")).lower() not in ("", "disabled", "off"):
            return "medium"
        return None


    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        response = None
        async for chunk in self._stream(request):
            if chunk.get("type") == "message_complete":
                response = chunk["response"]
        if response is None:
            raise APIError(f"{self.label} stream ended without a response",
                           category=ErrorCategory.NETWORK)
        return response

    async def create_message_stream(
        self, *, model_config: Any, messages: List[Dict[str, Any]], system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        request = self._build_request(model_config=model_config, messages=messages, system=system,
                                      tools=tools, tool_choice=tool_choice, stream=True)
        async for chunk in self._stream(request):
            yield chunk

    async def _stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        await self._prepare()
        body = self._body(request)
        started = time.monotonic()
        try:
            async for chunk in self._stream_once(body, started):
                yield chunk
        except APIError:
            raise
        except httpx.TimeoutException as exc:
            raise APIError(f"{self.label} timed out: {exc}",
                           category=ErrorCategory.TIMEOUT, cause=exc) from exc
        except httpx.TransportError as exc:
            raise APIError(f"{self.label} connection failed: {exc}",
                           category=ErrorCategory.NETWORK, cause=exc) from exc

    async def _stream_once(self, body: Dict[str, Any], started: float) -> AsyncIterator[Dict[str, Any]]:
        for attempt in (0, 1):
            async with self._client() as client:
                async with client.stream("POST", f"{self._base}/responses",
                                         headers=self._headers(), json=body) as resp:
                    # One renewal attempt, before anything has been yielded —
                    # after the first token a retry would replay half an answer.
                    if resp.status_code == 401 and attempt == 0 and await self._renew():
                        continue
                    if resp.status_code != 200:
                        raw = (await resp.aread()).decode("utf-8", "replace")
                        raise self._http_error(resp.status_code, raw)
                    async for chunk in self._consume(resp, started):
                        yield chunk
                    return

    def _http_error(self, status: int, raw: str) -> APIError:
        detail = raw
        try:
            data = json.loads(raw)
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or raw)
                if err.get("resets_in_seconds"):
                    self._notify_host({"kind": "rate_limit", "info": err})
            elif isinstance(data, dict) and data.get("detail"):
                detail = str(data["detail"])
        except Exception:  # noqa: BLE001 — a non-JSON body is the message
            pass
        label = f" [{self._account_label}]" if self._account_label else ""
        category = classify_text(detail, status)
        if status == 400 and "usage" in detail.lower() and "limit" in detail.lower():
            category = ErrorCategory.RATE_LIMITED
        return APIError(f"{self.label}{label} HTTP {status}: {detail[:600]}",
                        category=category, status_code=status)

    async def _consume(self, resp: httpx.Response, started: float) -> AsyncIterator[Dict[str, Any]]:
        text_parts: list[str] = []
        calls: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        usage = TokenUsage()
        model = ""
        stop = "end_turn"
        data_lines: list[str] = []

        async def events() -> AsyncIterator[dict[str, Any]]:
            async for line in resp.aiter_lines():
                if line == "":
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines.clear()
                        if payload.strip() == "[DONE]":
                            return
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                try:
                    yield json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    return

        completed = False
        async for ev in events():
            kind = ev.get("type", "")
            if kind == "response.output_text.delta":
                delta = str(ev.get("delta") or "")
                if delta:
                    text_parts.append(delta)
                    yield {"type": "text_delta", "text": delta}
            elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                delta = str(ev.get("delta") or "")
                if delta:
                    yield {"type": "thinking_delta", "text": delta}
            elif kind == "response.output_item.added":
                item = ev.get("item") or {}
                if item.get("type") == "function_call":
                    key = str(item.get("id") or item.get("call_id"))
                    calls[key] = {"call_id": item.get("call_id"), "name": item.get("name"),
                                  "arguments": item.get("arguments") or ""}
                    order.append(key)
            elif kind == "response.function_call_arguments.delta":
                key = str(ev.get("item_id"))
                if key in calls:
                    calls[key]["arguments"] += str(ev.get("delta") or "")
            elif kind == "response.output_item.done":
                item = ev.get("item") or {}
                if item.get("type") == "function_call":
                    key = str(item.get("id") or item.get("call_id"))
                    if key not in calls:
                        order.append(key)
                    calls[key] = {"call_id": item.get("call_id"), "name": item.get("name"),
                                  "arguments": item.get("arguments") or calls.get(key, {}).get("arguments", "")}
                elif item.get("type") == "message" and not text_parts:
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text" and part.get("text"):
                            text_parts.append(part["text"])
                            yield {"type": "text_delta", "text": part["text"]}
            elif kind in ("response.completed", "response.incomplete"):
                completed = True
                response = ev.get("response") or {}
                model = str(response.get("model") or "")
                u = response.get("usage") or {}
                cached = int(((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0)
                usage = TokenUsage(
                    input_tokens=max(0, int(u.get("input_tokens") or 0) - cached),
                    output_tokens=int(u.get("output_tokens") or 0),
                    cache_read_input_tokens=cached,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                if kind == "response.incomplete":
                    stop = "max_tokens"
            elif kind in ("response.failed", "error"):
                err = (ev.get("response") or {}).get("error") or ev.get("error") or ev
                message = str(err.get("message") if isinstance(err, dict) else err)
                code = str(err.get("code") if isinstance(err, dict) else "")
                category = classify_text(f"{code} {message}")
                if category == ErrorCategory.UNKNOWN:
                    category = ErrorCategory.SERVER_ERROR
                raise APIError(f"{self.label}: {message[:600]}", category=category)

        if not completed and not text_parts and not calls:
            raise APIError(f"{self.label} stream ended early", category=ErrorCategory.NETWORK)

        blocks: list[ContentBlock] = []
        text = "".join(text_parts)
        if text:
            blocks.append(ContentBlock(type="text", text=text, raw={"type": "text", "text": text}))
        for key in order:
            call = calls.get(key)
            if not call or not call.get("name"):
                continue
            try:
                args = json.loads(call.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"input": args}
            except json.JSONDecodeError:
                args = {"_raw": call.get("arguments")}
            call_id = str(call.get("call_id") or f"call_{uuid.uuid4().hex[:16]}")
            blocks.append(ContentBlock(type="tool_use", tool_use_id=call_id, tool_name=call["name"],
                                       tool_input=args,
                                       raw={"type": "tool_use", "id": call_id, "name": call["name"], "input": args}))
        if any(b.type == "tool_use" for b in blocks):
            stop = "tool_use"
        if not blocks:
            blocks.append(ContentBlock(type="text", text="", raw={"type": "text", "text": ""}))
        yield {"type": "message_complete",
               "response": APIResponse(content=blocks, stop_reason=stop, usage=usage, model=model,
                                       raw={"provider": self.provider, "account": self._account_id})}
