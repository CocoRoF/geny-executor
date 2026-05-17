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

    # MCP config: accept dict (inline JSON), str path, or pre-serialized JSON.
    if mcp_config is not None:
        if isinstance(mcp_config, str):
            argv += ["--mcp-config", mcp_config]
        else:
            argv += ["--mcp-config", json.dumps(mcp_config)]

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


def build_stream_json_stdin(messages: List[Dict[str, Any]]) -> bytes:
    """Render canonical messages as Claude Code stream-json stdin.

    Claude Code's ``--input-format stream-json`` expects newline-delimited
    JSON envelopes of the shape::

        {"type": "user", "message": {"role": "user", "content": [...]}}

    Tool-results / assistant turns from prior multi-turn history flow as
    additional ``user``-typed entries with their original role embedded —
    the CLI reconstructs the conversation from the envelopes.
    """
    out_lines: List[str] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        envelope = {
            "type": "user",
            "message": {"role": role, "content": content},
        }
        out_lines.append(json.dumps(envelope))
    blob = "\n".join(out_lines)
    return (blob + "\n").encode("utf-8") if blob else b""


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


async def assemble_response_from_stream_json(
    stream: AsyncIterator[bytes],
    *,
    model: str,
) -> APIResponse:
    """Drain a stream-json output and return a canonical APIResponse.

    Used by ``ClaudeCodeCLIClient._send`` when ``request.stream=True``. The
    final ``result`` envelope carries usage + stop_reason; the accumulated
    text / thinking / tool_use blocks come from intermediate ``assistant``
    deltas.
    """
    text_buf: List[str] = []
    thinking_buf: List[str] = []
    tool_uses: List[Dict[str, Any]] = []
    current_tool: Optional[Dict[str, Any]] = None
    final_obj: Optional[Dict[str, Any]] = None
    message_id = ""
    stop_reason = "end_turn"
    resolved_model = model

    from geny_executor.llm_client._cli_runtime import parse_stream_json_line

    async for raw in stream:
        line = parse_stream_json_line(raw)
        if line is None:
            continue
        if "__malformed__" in line:
            # Skip malformed lines — caller's CLIProtocolError path
            # already runs on non-zero exits.
            continue
        ltype = str(line.get("type", ""))

        if ltype == "system":
            # The first system envelope carries session metadata.
            message_id = str(line.get("session_id") or line.get("message_id") or message_id)
            resolved_model = str(line.get("model") or resolved_model)
            continue

        if ltype == "assistant":
            delta = line.get("delta") or {}
            dtype = str(delta.get("type", ""))
            if dtype == "text_delta":
                text_buf.append(str(delta.get("text", "")))
                continue
            if dtype == "thinking_delta":
                thinking_buf.append(str(delta.get("text", "")))
                continue
            if dtype == "input_json_delta":
                if current_tool is not None:
                    current_tool.setdefault("_partial_json", "")
                    current_tool["_partial_json"] += str(delta.get("partial_json", ""))
                continue
            cb = line.get("content_block")
            if isinstance(cb, dict) and cb.get("type") == "tool_use":
                current_tool = {
                    "id": cb.get("id"),
                    "name": cb.get("name"),
                    "input": cb.get("input") or {},
                }
                continue

        if ltype == "content_block_stop":
            if current_tool is not None:
                # Finalise the in-flight tool block. If we accumulated a
                # partial_json buffer, try parsing it as the input.
                partial = current_tool.pop("_partial_json", "")
                if partial and not current_tool.get("input"):
                    try:
                        current_tool["input"] = json.loads(partial)
                    except json.JSONDecodeError:
                        current_tool["input"] = {"_raw": partial}
                tool_uses.append(current_tool)
                current_tool = None
            continue

        if ltype == "result":
            final_obj = line
            stop_reason = str(line.get("stop_reason", stop_reason))
            continue

        if ltype == "error":
            raise RuntimeError(
                f"Claude Code CLI reported error: {line.get('message') or line!r}"
            )

    blocks: List[ContentBlock] = []
    if thinking_buf:
        blocks.append(ContentBlock(type="thinking", thinking_text="".join(thinking_buf)))
    if text_buf:
        blocks.append(ContentBlock(type="text", text="".join(text_buf)))
    for tu in tool_uses:
        blocks.append(
            ContentBlock(
                type="tool_use",
                tool_use_id=tu.get("id"),
                tool_name=tu.get("name"),
                tool_input=tu.get("input") or {},
            )
        )

    usage_in: Dict[str, Any] = {}
    if final_obj:
        usage_in = final_obj.get("usage", {}) or {}
    usage = TokenUsage(
        input_tokens=int(usage_in.get("input_tokens", 0) or 0),
        output_tokens=int(usage_in.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(usage_in.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(usage_in.get("cache_read_input_tokens", 0) or 0),
        cost_usd=usage_in.get("cost_usd"),
        duration_ms=(final_obj or {}).get("duration_ms"),
    )

    return APIResponse(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        model=resolved_model,
        message_id=message_id,
        raw=final_obj,
    )


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
