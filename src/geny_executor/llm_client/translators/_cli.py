"""Canonical ↔ CLI translation helpers.

Used by ``ClaudeCodeCLIClient`` (Phase B) and ``CopilotCLIClient`` (Phase C)
to:

  - Build vendor-specific argv lists from a canonical :class:`APIRequest`.
  - Assemble a canonical :class:`APIResponse` from CLI output.
  - Map streaming stream-json line types to canonical event dicts.

Claude Code helpers landed in Phase B1; ``gh copilot`` helpers
(``compose_copilot_prompt``, ``copilot_argv``, ``parse_plain_text_to_response``)
land here in Phase C1.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


# ---------------------------------------------------------------------------
# Claude Code: thinking budget → --effort string
# ---------------------------------------------------------------------------


def thinking_to_effort(thinking: Optional[Dict[str, Any]]) -> Optional[str]:
    """Map a canonical thinking dict to the ``--effort`` flag value.

    Buckets (rough heuristic mirroring vendor docs):
      budget <= 5k   → ``low``
      budget <= 15k  → ``medium``
      budget <= 32k  → ``high``
      budget <= 64k  → ``xhigh``
      else           → ``max``

    Returns ``None`` when thinking is None or its type is "disabled".
    """
    if not thinking:
        return None
    ttype = str(thinking.get("type", "")).lower()
    if ttype in {"", "disabled", "off"}:
        return None
    budget = int(thinking.get("budget_tokens", 0) or 0)
    if budget <= 5_000:
        return "low"
    if budget <= 15_000:
        return "medium"
    if budget <= 32_000:
        return "high"
    if budget <= 64_000:
        return "xhigh"
    return "max"


# ---------------------------------------------------------------------------
# Claude Code: argv builder
# ---------------------------------------------------------------------------


def claude_code_argv(
    request: APIRequest,
    *,
    bare_mode: bool = True,
    permission_mode: str = "default",
    max_budget_usd: Optional[float] = None,
    settings_path: Optional[str] = None,
    mcp_config: Any = None,
    allow_tools: Sequence[str] = (),
    disallow_tools: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> List[str]:
    """Build the argv list (excluding the binary) for one Claude Code call.

    The mapping is intentionally narrow — only flags Claude Code's
    ``--print`` mode honours. Fields the CLI does not accept (temperature,
    top_p, top_k, stop_sequences, tool_choice) are dropped silently by the
    caller via the standard capability-negotiation path.
    """
    argv: List[str] = ["--print"]

    # Output / input formats: always stream-json for streaming requests,
    # else json so we can parse a single object.
    if request.stream:
        argv += [
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
        ]
    else:
        argv += ["--output-format", "json"]

    if bare_mode:
        argv.append("--bare")

    # Model: alias or pinned id.
    if request.model:
        argv += ["--model", str(request.model)]

    # System prompt: --system-prompt fully replaces the CLI's default.
    if request.system:
        if isinstance(request.system, str):
            sys_text = request.system
        elif isinstance(request.system, list):
            # Anthropic-shaped system blocks — flatten text only.
            parts: List[str] = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            sys_text = "\n".join(parts)
        else:
            sys_text = str(request.system)
        if sys_text:
            argv += ["--system-prompt", sys_text]

    # Thinking → --effort
    effort = thinking_to_effort(request.thinking)
    if effort:
        argv += ["--effort", effort]

    # Tool allow/deny lists. We pass these as space-joined strings (the CLI
    # accepts comma- or space-separated input per its --help).
    if allow_tools:
        argv += ["--allowedTools", " ".join(allow_tools)]
    if disallow_tools:
        argv += ["--disallowedTools", " ".join(disallow_tools)]

    # Permission mode (passthrough).
    if permission_mode and permission_mode != "default":
        argv += ["--permission-mode", permission_mode]

    # Budget cap.
    if max_budget_usd is not None:
        argv += ["--max-budget-usd", str(max_budget_usd)]

    # Settings file (e.g. apiKeyHelper).
    if settings_path:
        argv += ["--settings", settings_path]

    # MCP config — precedence:
    #   1. ``request.mcp_config`` (per-request, set by host for
    #      session-scoped MCP wraps). Phase I: Geny synthesizes a
    #      per-session MCP config that bridges its tool registry to
    #      the CLI so the LLM can call host tools via MCP.
    #   2. ``mcp_config`` constructor kwarg (legacy per-client static
    #      config from the LLM-backends settings card).
    # Both flow to ``--mcp-config <json|path>``.
    effective_mcp_config: Any = (
        request.mcp_config if request.mcp_config is not None else mcp_config
    )
    has_host_mcp = bool(effective_mcp_config)
    if has_host_mcp:
        if isinstance(effective_mcp_config, str):
            argv += ["--mcp-config", effective_mcp_config]
        else:
            argv += [
                "--mcp-config",
                json.dumps(effective_mcp_config, ensure_ascii=False),
            ]
        # When the host exposes its own tool surface via MCP, disable
        # the CLI's built-in tool palette so the LLM cannot hallucinate
        # against ``Bash`` / ``Read`` / ``ToolSearch`` / etc. The CLI's
        # ``--tools ""`` literal disables the entire built-in set per
        # ``claude --help``. Caller-supplied ``allow_tools`` /
        # ``disallow_tools`` (legacy CLI-built-in filters) are also
        # forwarded earlier so a host that wants a mixed surface — MCP
        # tools + a curated subset of CLI built-ins — can opt back in.
        # ``--strict-mcp-config`` ignores any other MCP config sources
        # (user-level / project-level) so the per-session bridge is
        # the sole MCP surface the CLI sees.
        if not allow_tools:
            argv += ["--tools", ""]
        argv += ["--strict-mcp-config"]

    # JSON schema (structured output).
    if request.response_format:
        rf = request.response_format
        rftype = str(rf.get("type", "")).lower()
        if rftype == "json_schema" and "json_schema" in rf:
            argv += ["--json-schema", json.dumps(rf["json_schema"])]

    # Session continuity.
    if request.session_hint:
        sid = request.session_hint.get("session_id")
        if request.session_hint.get("resume") and sid:
            argv += ["--resume", str(sid)]
        elif sid:
            argv += ["--session-id", str(sid)]

    # Caller-supplied escape hatch.
    if extra_args:
        argv += list(extra_args)

    return argv


# ---------------------------------------------------------------------------
# Claude Code: stdin builder (stream-json input mode)
# ---------------------------------------------------------------------------


def _render_block_for_history(block: Any) -> str:
    """Render one Anthropic-style content block as readable text.

    Used by ``build_stream_json_stdin`` when collapsing multi-turn
    history into a single synthetic user envelope. Preserves enough
    fidelity (tool name + input, tool result text) for the LLM to
    reconstruct the conversation, while dropping shapes the CLI
    cannot ingest (thinking blocks, images→placeholder)."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return str(block)
    btype = str(block.get("type", ""))
    if btype == "text":
        return str(block.get("text", ""))
    if btype == "thinking":
        # Thinking traces from a prior provider don't replay on the
        # CLI — drop them. The CLI does its own ``--effort`` thinking
        # on the new turn.
        return ""
    if btype == "tool_use":
        name = block.get("name", "tool")
        try:
            input_json = json.dumps(
                block.get("input") or {}, ensure_ascii=False,
            )
        except (TypeError, ValueError):
            input_json = str(block.get("input"))
        return f"[Tool call: {name}({input_json})]"
    if btype == "tool_result":
        body = block.get("content")
        if isinstance(body, list):
            body = "\n".join(
                _render_block_for_history(b) for b in body
            ).strip()
        elif body is None:
            body = ""
        is_error = bool(block.get("is_error"))
        tag = "Tool error" if is_error else "Tool result"
        return f"[{tag}] {body}"
    if btype == "image":
        return "[image attachment]"
    return ""


def _render_content_for_history(content: Any) -> str:
    """Flatten a canonical ``content`` field (string or block list)
    into one display-ready text run for history-preamble use."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered = [
            _render_block_for_history(b) for b in content
        ]
        return "\n".join(s for s in rendered if s).strip()
    return str(content)


def build_stream_json_stdin(messages: List[Dict[str, Any]]) -> bytes:
    """Render canonical Anthropic-style messages as Claude Code
    stream-json stdin — **always as a single ``type:user`` envelope**.

    Claude Code CLI's ``--input-format stream-json`` strictly requires
    each envelope's ``message.role`` to be ``"user"``. The previous
    implementation forwarded the canonical role through (assistant /
    tool turns embedded in ``type:user`` envelopes with their original
    role kept) which the CLI rejects with::

        Error: Expected message role 'user', got 'assistant'

    For multi-turn pipelines (Geny's s06_api accumulates conversation
    history across loop iterations) we collapse the whole history into
    a single synthetic user envelope:

      - The latest user message becomes the bulk of the prompt.
      - All prior turns are rendered as a markdown preamble
        (``### User`` / ``### Assistant`` / tool calls + results).
      - The CLI receives one cohesive single-turn prompt with all
        relevant context — same input contract whether the host is
        running Geny's iterative loop or sending a one-shot query.

    The single-turn fast-path (one user message only) emits the
    canonical envelope unchanged so simple invocations stay byte-for-
    byte identical to the legacy path.
    """
    if not messages:
        return b""

    # Single-turn fast path — preserve the canonical envelope shape.
    if len(messages) == 1 and str(messages[0].get("role", "")) == "user":
        envelope = {
            "type": "user",
            "message": {"role": "user", "content": messages[0].get("content", "")},
        }
        return (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")

    # Multi-turn: flatten into a single synthetic user message. The
    # CLI's ``--bare`` mode treats this as a regular prompt; the LLM
    # reconstructs the conversation from the markdown structure.
    parts: List[str] = []
    last_user_idx = -1
    for i, m in enumerate(messages):
        if str(m.get("role", "")) == "user":
            last_user_idx = i

    for i, m in enumerate(messages):
        role = str(m.get("role", "user"))
        text = _render_content_for_history(m.get("content", ""))
        if not text and role != "assistant":
            continue
        if role == "user":
            # The final user turn is the "current input" — render it
            # without a header so it reads as the actual question.
            if i == last_user_idx:
                parts.append(text)
            else:
                parts.append(f"### User\n{text}")
        elif role == "assistant":
            if text:
                parts.append(f"### Assistant\n{text}")
        elif role == "tool":
            parts.append(f"### Tool result\n{text}")
        else:
            parts.append(f"### {role.capitalize()}\n{text}")

    preamble = ""
    current_input = parts[-1] if parts else ""
    if len(parts) > 1:
        preamble_parts = parts[:-1]
        preamble = (
            "## Conversation so far\n\n"
            + "\n\n".join(preamble_parts)
            + "\n\n## Current input\n"
        )

    flat = (preamble + current_input).strip()
    envelope = {
        "type": "user",
        "message": {"role": "user", "content": flat},
    }
    return (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Claude Code: stream-json line → canonical event
# ---------------------------------------------------------------------------


def stream_json_line_to_canonical_event(line_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one Claude Code stream-json line into a canonical event.

    Returns:
      - ``None`` for envelope lines (``system``, ``user``) that carry no
        emit-worthy delta.
      - ``{"type": ..., ...}`` for assistant deltas (text, thinking, tool
        use, input json), block stops, and the terminal ``message_complete``
        event with ``response: APIResponse``.
      - The raw ``error`` envelope when the CLI surfaces one.

    The exact stream-json line shape Claude Code emits evolves; this helper
    handles the contemporary subset (system / assistant message + content
    blocks / result). Unknown line types are reported as
    ``{"type": "cli_unknown", "raw": ...}`` so callers can log + ignore.
    """
    if not isinstance(line_obj, dict):
        return None
    if "__malformed__" in line_obj:
        return {"type": "cli_malformed", "raw": line_obj["__malformed__"]}

    ltype = str(line_obj.get("type", ""))
    if ltype == "system":
        return None  # session preamble — the assembler consumes separately
    if ltype == "user":
        return None  # echo of our input

    if ltype == "assistant":
        # Delta variants: {"delta": {...}} or {"message": {"content": [...]}}.
        delta = line_obj.get("delta") or {}
        dtype = str(delta.get("type", ""))
        if dtype == "text_delta":
            return {"type": "text_delta", "text": delta.get("text", "")}
        if dtype == "thinking_delta":
            return {"type": "thinking_delta", "text": delta.get("text", "")}
        if dtype == "input_json_delta":
            return {"type": "input_json_delta", "delta": delta.get("partial_json", "")}
        # block_start carries tool_use metadata
        if "content_block" in line_obj:
            cb = line_obj["content_block"]
            if isinstance(cb, dict) and cb.get("type") == "tool_use":
                return {
                    "type": "tool_use",
                    "id": cb.get("id"),
                    "name": cb.get("name"),
                    "input": cb.get("input") or {},
                }
        # Full-message form (Claude Code 2.x default): collapse the
        # entire content array to a single concatenated text_delta so
        # legacy single-event consumers see SOME text. Callers that
        # need per-block fidelity should use ``StreamJsonAccumulator``
        # directly.
        msg = line_obj.get("message") or {}
        if isinstance(msg, dict):
            parts: List[str] = []
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if text:
                        parts.append(text)
            if parts:
                return {"type": "text_delta", "text": "".join(parts)}
        return None

    if ltype == "content_block_stop":
        return {"type": "content_block_stop"}
    if ltype == "message_stop":
        return {"type": "message_complete"}

    if ltype == "result":
        return {"type": "result", "raw": line_obj}

    if ltype == "error":
        return {"type": "error", "raw": line_obj}

    return {"type": "cli_unknown", "raw": line_obj}


# ---------------------------------------------------------------------------
# Claude Code: assemble final APIResponse from JSON output
# ---------------------------------------------------------------------------


def parse_json_output_to_response(stdout: bytes, *, model: str) -> APIResponse:
    """Parse the single JSON object emitted by ``--output-format json``.

    Claude Code's ``json`` output is roughly::

        {
          "type": "result",
          "message_id": "...",
          "stop_reason": "end_turn",
          "content": [{"type": "text", "text": "..."}, ...],
          "usage": {"input_tokens": ..., "output_tokens": ..., "cost_usd": ...},
          "duration_ms": ...
        }
    """
    try:
        obj = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude Code json output not parseable: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("Claude Code json output is not an object")

    blocks: List[ContentBlock] = []
    for block in obj.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", "text"))
        if btype == "text":
            blocks.append(ContentBlock(type="text", text=block.get("text", "")))
        elif btype == "thinking":
            blocks.append(
                ContentBlock(type="thinking", thinking_text=block.get("text", ""))
            )
        elif btype == "tool_use":
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=block.get("id"),
                    tool_name=block.get("name"),
                    tool_input=block.get("input") or {},
                )
            )

    usage_in = obj.get("usage", {}) or {}
    usage = TokenUsage(
        input_tokens=int(usage_in.get("input_tokens", 0) or 0),
        output_tokens=int(usage_in.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(usage_in.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(usage_in.get("cache_read_input_tokens", 0) or 0),
        cost_usd=usage_in.get("cost_usd"),
        duration_ms=obj.get("duration_ms"),
    )

    return APIResponse(
        content=blocks,
        stop_reason=str(obj.get("stop_reason", "end_turn")),
        usage=usage,
        model=str(obj.get("model", model)),
        message_id=str(obj.get("message_id", "")),
        raw=obj,
    )


# ---------------------------------------------------------------------------
# Claude Code: assemble final APIResponse from a stream-json byte stream
# ---------------------------------------------------------------------------


class StreamJsonAccumulator:
    """Walk Claude Code stream-json lines and accumulate the final response.

    Handles both shapes the CLI emits (the shape varies by version + by
    ``--include-partial-messages``):

    1. **Delta form** (true streaming, ``--include-partial-messages`` on):
       ``{"type":"assistant","delta":{"type":"text_delta","text":"..."}}``
       — one delta per token-ish chunk; ``content_block_stop`` terminates a
       block.
    2. **Message form** (default + observed on claude_code 2.1.144):
       ``{"type":"assistant","message":{"content":[
           {"type":"text","text":"..."},
           {"type":"thinking","thinking":"..."},
           {"type":"tool_use","id":"...","name":"...","input":{...}},
         ],"stop_reason":"...","usage":{...}}}``
       — the full assistant message arrives in one envelope.

    The accumulator's ``feed(line)`` returns a list of canonical UI events
    ({"type":"text_delta", ...} etc.) that callers stream to consumers,
    while internally bookkeeping the state needed to call ``finalize()``
    for the terminal :class:`APIResponse`.
    """

    def __init__(self, model: str) -> None:
        self._text_buf: List[str] = []
        self._thinking_buf: List[str] = []
        self._tool_uses: List[Dict[str, Any]] = []
        self._current_tool: Optional[Dict[str, Any]] = None
        self._final_obj: Optional[Dict[str, Any]] = None
        self._message_id = ""
        self._stop_reason = "end_turn"
        self._resolved_model = model

    # ── Public ────────────────────────────────────────────────

    def feed(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update state from one stream-json line.

        Returns the list of canonical UI events the line produced
        (``text_delta`` / ``thinking_delta`` / ``tool_use`` / ...).
        Caller is responsible for yielding them to its own consumer.
        Empty list when the line is bookkeeping-only.
        """
        if not isinstance(line, dict) or "__malformed__" in line:
            return []
        ltype = str(line.get("type", ""))

        if ltype == "system":
            self._message_id = str(
                line.get("session_id") or line.get("message_id") or self._message_id
            )
            self._resolved_model = str(line.get("model") or self._resolved_model)
            return []

        if ltype == "assistant":
            return self._feed_assistant(line)

        if ltype == "content_block_stop":
            self._close_current_tool()
            return [{"type": "content_block_stop"}]

        if ltype == "message_stop":
            # Suppressed at this layer — the streaming caller emits one
            # populated ``message_complete`` after ``finalize()``.
            return []

        if ltype == "result":
            self._final_obj = line
            self._stop_reason = str(line.get("stop_reason", self._stop_reason))
            # ``message`` form puts stop_reason on the assistant envelope
            # too; keep whichever non-empty value won.
            return [{"type": "result", "raw": line}]

        if ltype == "error":
            return [{"type": "error", "raw": line}]

        return [{"type": "cli_unknown", "raw": line}]

    def finalize(self) -> APIResponse:
        """Build the canonical :class:`APIResponse` from accumulated state."""
        # Flush any unclosed tool — the message form often skips
        # ``content_block_stop`` entirely.
        self._close_current_tool()

        blocks: List[ContentBlock] = []
        if self._thinking_buf:
            blocks.append(
                ContentBlock(type="thinking", thinking_text="".join(self._thinking_buf))
            )
        if self._text_buf:
            blocks.append(ContentBlock(type="text", text="".join(self._text_buf)))
        for tu in self._tool_uses:
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=tu.get("id"),
                    tool_name=tu.get("name"),
                    tool_input=tu.get("input") or {},
                )
            )

        usage_in: Dict[str, Any] = (self._final_obj or {}).get("usage", {}) or {}
        usage = TokenUsage(
            input_tokens=int(usage_in.get("input_tokens", 0) or 0),
            output_tokens=int(usage_in.get("output_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                usage_in.get("cache_creation_input_tokens", 0) or 0
            ),
            cache_read_input_tokens=int(usage_in.get("cache_read_input_tokens", 0) or 0),
            cost_usd=usage_in.get("cost_usd")
            or (self._final_obj or {}).get("total_cost_usd"),
            duration_ms=(self._final_obj or {}).get("duration_ms"),
        )

        return APIResponse(
            content=blocks,
            stop_reason=self._stop_reason,
            usage=usage,
            model=self._resolved_model,
            message_id=self._message_id,
            raw=self._final_obj or {},
        )

    # ── Internals ─────────────────────────────────────────────

    def _feed_assistant(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Form 1 — delta (true streaming).
        delta = line.get("delta") or {}
        dtype = str(delta.get("type", ""))
        if dtype == "text_delta":
            text = str(delta.get("text", ""))
            self._text_buf.append(text)
            return [{"type": "text_delta", "text": text}] if text else []
        if dtype == "thinking_delta":
            text = str(delta.get("text", ""))
            self._thinking_buf.append(text)
            return [{"type": "thinking_delta", "text": text}] if text else []
        if dtype == "input_json_delta":
            partial = str(delta.get("partial_json", ""))
            if self._current_tool is not None:
                self._current_tool.setdefault("_partial_json", "")
                self._current_tool["_partial_json"] += partial
            return [{"type": "input_json_delta", "delta": partial}]
        cb = line.get("content_block")
        if isinstance(cb, dict) and cb.get("type") == "tool_use":
            self._current_tool = {
                "id": cb.get("id"),
                "name": cb.get("name"),
                "input": cb.get("input") or {},
            }
            return [
                {
                    "type": "tool_use",
                    "id": cb.get("id"),
                    "name": cb.get("name"),
                    "input": cb.get("input") or {},
                }
            ]

        # Form 2 — full message (default Claude Code 2.x output).
        message = line.get("message") or {}
        if isinstance(message, dict) and message.get("content"):
            return self._feed_message(message)

        return []

    def _feed_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a full assistant message envelope's content array.

        Emits synthetic per-block delta events so UI consumers see the
        same canonical shape they would with true streaming, then
        records the blocks for the eventual :class:`APIResponse`.
        """
        # Capture stop_reason / usage off the envelope if present —
        # the ``message`` form lets the assistant frame carry these
        # instead of waiting for the final ``result`` line.
        sr = message.get("stop_reason")
        if sr:
            self._stop_reason = str(sr)
        usage = message.get("usage")
        if isinstance(usage, dict) and self._final_obj is None:
            self._final_obj = {"usage": usage}
        # Skip synthetic "Not logged in" messages — Claude Code emits
        # them with ``error=authentication_failed`` and a placeholder
        # text block. Surface as an APIError-friendly error event so
        # callers raise instead of returning empty output.
        # (Detected on the outer ``line``, but ``message`` is the
        # carrier so we pass it through unchanged here.)

        events: List[Dict[str, Any]] = []
        content = message.get("content") or []
        if not isinstance(content, list):
            return events
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", ""))
            if btype == "text":
                text = str(block.get("text", ""))
                if text:
                    self._text_buf.append(text)
                    events.append({"type": "text_delta", "text": text})
            elif btype == "thinking":
                # Anthropic uses ``thinking`` field; some shims use ``text``.
                text = str(block.get("thinking") or block.get("text") or "")
                if text:
                    self._thinking_buf.append(text)
                    events.append({"type": "thinking_delta", "text": text})
            elif btype == "tool_use":
                tu = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input") or {},
                }
                self._tool_uses.append(tu)
                events.append(
                    {
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"],
                    }
                )
        return events

    def _close_current_tool(self) -> None:
        if self._current_tool is None:
            return
        partial = self._current_tool.pop("_partial_json", "")
        if partial and not self._current_tool.get("input"):
            try:
                self._current_tool["input"] = json.loads(partial)
            except json.JSONDecodeError:
                self._current_tool["input"] = {"_raw": partial}
        self._tool_uses.append(self._current_tool)
        self._current_tool = None


async def assemble_response_from_stream_json(
    stream: AsyncIterator[bytes],
    *,
    model: str,
) -> APIResponse:
    """Drain a stream-json output and return a canonical APIResponse.

    Used by ``ClaudeCodeCLIClient._send`` when ``request.stream=True``.
    Thin wrapper around :class:`StreamJsonAccumulator` so the
    streaming + non-streaming consumer paths share one parser — Claude
    Code's stream-json shape (delta vs full-message) varies by CLI
    version and ``--include-partial-messages``, and we never want the
    two paths to drift again.
    """
    from geny_executor.llm_client._cli_runtime import parse_stream_json_line

    accum = StreamJsonAccumulator(model=model)
    async for raw in stream:
        line = parse_stream_json_line(raw)
        if line is None:
            continue
        if "__malformed__" in line:
            continue
        # ``error`` envelopes from the CLI need to raise so the caller's
        # CLIProtocolError path runs — match the prior behaviour exactly.
        if str(line.get("type", "")) == "error":
            raise RuntimeError(
                f"Claude Code CLI reported error: {line.get('message') or line!r}"
            )
        accum.feed(line)

    return accum.finalize()


# ---------------------------------------------------------------------------
# Copilot CLI: prompt composition
# ---------------------------------------------------------------------------


def compose_copilot_prompt(system: Any, messages: List[Dict[str, Any]]) -> str:
    """Flatten a canonical (system + messages) into one ``-p`` argument.

    The Copilot CLI accepts a single prompt string. Conversation history
    is encoded as Markdown-style turns so the model can still see prior
    turns. The system prompt is prepended as a ``## System`` section
    when present.
    """
    parts: List[str] = []
    if system:
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            sys_text = "\n".join(
                str(b.get("text", "")) for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            sys_text = str(system)
        if sys_text:
            parts.append(f"## System\n{sys_text}")

    for m in messages:
        role = str(m.get("role", "user")).capitalize()
        content = m.get("content", "")
        if isinstance(content, list):
            chunks: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    chunks.append(str(block.get("text", "")))
                elif btype == "tool_result":
                    chunks.append(f"[tool_result]\n{block.get('content', '')}")
            content_text = "\n".join(chunks)
        else:
            content_text = str(content)
        if content_text:
            parts.append(f"## {role}\n{content_text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Copilot CLI: argv builder
# ---------------------------------------------------------------------------


def copilot_argv(
    *,
    allow_tools: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> List[str]:
    """Build the argv list for ``gh copilot`` (excluding the binary).

    The caller is expected to invoke the result as ``gh copilot ...`` —
    i.e. ``argv[0]`` is *not* prepended here. ``-p <prompt>`` is appended
    by the client after computing the prompt via
    :func:`compose_copilot_prompt`.

    Only the flags the CLI actually accepts are emitted:
      - ``-p``: single prompt (added by the client, not here)
      - ``--allow-tool '<scope>'``: repeated, one flag per scope
      - any ``extra_args`` for escape-hatch use.
    """
    argv: List[str] = ["copilot"]
    for scope in allow_tools:
        if scope:
            argv += ["--allow-tool", str(scope)]
    if extra_args:
        argv += list(extra_args)
    return argv


# ---------------------------------------------------------------------------
# Copilot CLI: stdout → APIResponse
# ---------------------------------------------------------------------------


def parse_plain_text_to_response(text: str, *, model: str = "default") -> APIResponse:
    """Wrap plain stdout text into a canonical :class:`APIResponse`.

    Copilot CLI does not return JSON in print mode, so we cannot recover
    structured usage / cost. The response carries the text in a single
    block, ``stop_reason="end_turn"``, and an empty TokenUsage with
    ``supports_token_usage=False`` advertised at the client level.
    """
    content_text = text.strip("\n")
    return APIResponse(
        content=[ContentBlock(type="text", text=content_text)],
        stop_reason="end_turn",
        usage=TokenUsage(),
        model=model,
        message_id="",
        raw=text,
    )
