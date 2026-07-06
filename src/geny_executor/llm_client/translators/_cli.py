"""Canonical ↔ CLI translation helpers.

Used by ``ClaudeCodeCLIClient`` (Phase B) to:

  - Build vendor-specific argv lists from a canonical :class:`APIRequest`.
  - Assemble a canonical :class:`APIResponse` from CLI output.
  - Map streaming stream-json line types to canonical event dicts.

The Phase-C ``gh copilot`` helpers (``compose_copilot_prompt``,
``copilot_argv``, ``parse_plain_text_to_response``) were removed in
2.0.6 along with the ``CopilotCLIClient`` itself — ``gh copilot``
does not support streaming, tools, or MCP, so it could not host the
pipeline's Stage-10 dispatch loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence

from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock


logger = logging.getLogger(__name__)


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
    auth_mode: str = "auto",
    has_api_key: bool = False,
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

    ``auth_mode`` / ``has_api_key`` replace the process-env sniff that
    earlier versions used to decide ``--bare`` (PR #868 history): the
    builder read ``os.environ["ANTHROPIC_API_KEY"]`` from the *parent*
    process — a variable the spawned CLI may never see (the runner scrubs
    the child env) and one that says nothing about the credentials the
    client was actually constructed with. The decision now flows from
    :class:`ClaudeCodeCLIClient`, which knows its own ``_api_key`` and
    declared ``auth_mode``; this builder is a pure function of its
    arguments again.
    """
    argv: List[str] = ["--print"]

    # Output / input formats: always stream-json for streaming requests,
    # else json so we can parse a single object.
    #
    # ``--verbose`` is required by Claude Code CLI ≥ 2.1.x whenever
    # ``--print`` is combined with ``--output-format=stream-json``;
    # without it the CLI exits 1 with:
    #
    #     Error: When using --print, --output-format=stream-json
    #     requires --verbose
    #
    # 2.0.6 emits it automatically alongside the stream-json switch so
    # hosts don't have to thread an opt-in flag through their settings.
    if request.stream:
        argv += [
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ]
    else:
        argv += ["--output-format", "json"]

    # ``--bare`` skips OAuth + keychain reads (per ``claude --help``:
    # "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via
    # --settings (OAuth and keychain are never read)"). That's correct
    # for the API-key auth path but **wrong** for the subscription
    # OAuth path — passing ``--bare`` without an API key crashes every
    # subscription user with "Not logged in · Please run /login".
    #
    # Resolution rules (2.2.0):
    #   - ``auth_mode="api_key"``      → the host vouches for an API key
    #     reaching the child env → ``--bare`` allowed.
    #   - ``auth_mode="oauth"`` / ``"setup_token"`` → subscription-style
    #     credential on disk → never ``--bare``.
    #   - ``auth_mode="auto"`` (default) → resolves to ``api_key`` iff
    #     the client itself holds a non-empty key (``has_api_key``).
    # ``bare_mode=False`` keeps its historical meaning: never emit
    # ``--bare``, even on the API-key path.
    resolved_auth = auth_mode
    if resolved_auth not in ("api_key", "oauth", "setup_token"):
        resolved_auth = "api_key" if has_api_key else "oauth"
    if bare_mode and resolved_auth == "api_key":
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
    effective_mcp_config: Any = request.mcp_config if request.mcp_config is not None else mcp_config
    has_host_mcp = bool(effective_mcp_config)
    if has_host_mcp:
        if isinstance(effective_mcp_config, str):
            argv += ["--mcp-config", effective_mcp_config]
        else:
            argv += [
                "--mcp-config",
                json.dumps(effective_mcp_config, ensure_ascii=False),
            ]
        # ``--strict-mcp-config`` ignores any other MCP config sources
        # (user-level / project-level) so the per-session bridge is
        # the sole MCP surface the CLI sees. The CLI's *built-in* tool
        # palette (``Bash`` / ``Read`` / ``Write`` / ``Edit`` / …)
        # stays available alongside the MCP surface — earlier
        # executor versions auto-emitted ``--tools ""`` here to
        # disable it (a defensive measure against the LLM
        # hallucinating against unknown built-ins), but in practice
        # most hosts *want* both surfaces (e.g. a Sub-Worker writing
        # files via ``Write`` while delegating to MCP-wrapped host
        # tools). Hosts that prefer the old MCP-only behaviour can
        # pass ``extra_args=("--tools", "")`` explicitly.
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

    # Prompt delivery.
    #   * Streaming  → messages travel via stream-json STDIN
    #     (``build_stream_json_stdin``), so nothing goes in argv here.
    #   * Non-streaming (``--print --output-format json``) has NO stdin wired
    #     by ``_send``, so the prompt MUST be the trailing positional argument
    #     (``claude --print --output-format json --model X "<prompt>"``).
    #     Without this, the CLI exits 1 with "input must be provided either
    #     through stdin or as a prompt argument when using --print" — which
    #     broke every non-streaming ``create_message`` (e.g. offline memory
    #     summarisation) while streaming sessions worked.
    prompt_text = flatten_messages_to_prompt(request.messages) if not request.stream else ""

    # Caller-supplied escape hatch — emitted BEFORE the ``--`` separator so
    # any flags it carries (e.g. ``--tools ""``) are parsed as options, not
    # swept into the positional prompt below.
    if extra_args:
        argv += list(extra_args)

    # The prompt is the trailing positional, but ``--allowedTools`` /
    # ``--disallowedTools`` are *variadic* (``nargs="+"``) — without a
    # separator the CLI greedily swallows the prompt tokens as extra tool
    # names ("permission deny rule '<word>' matches no known tool"). The
    # POSIX ``--`` end-of-options marker forces everything after it to be
    # positional, so the prompt survives verbatim regardless of which
    # variadic flags precede it.
    if prompt_text:
        argv += ["--", prompt_text]

    return argv


# ---------------------------------------------------------------------------
# Claude Code: stdin builder (stream-json input mode)
# ---------------------------------------------------------------------------


def messages_have_images(messages: List[Dict[str, Any]]) -> bool:
    """True when any message carries an Anthropic-style image block.

    Used by the client to decide the wire mode: the ``--print`` positional
    prompt is text-only, so requests with images must travel as stream-json
    stdin (the CLI ingests base64 image blocks there)."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "image" for b in content
        ):
            return True
    return False


def _image_blocks_of(content: Any) -> List[Dict[str, Any]]:
    """Anthropic-style image blocks contained in one message's content."""
    if not isinstance(content, list):
        return []
    return [
        b for b in content if isinstance(b, dict) and b.get("type") == "image"
    ]


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
                block.get("input") or {},
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            input_json = str(block.get("input"))
        return f"[Tool call: {name}({input_json})]"
    if btype == "tool_result":
        body = block.get("content")
        if isinstance(body, list):
            body = "\n".join(_render_block_for_history(b) for b in body).strip()
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
        rendered = [_render_block_for_history(b) for b in content]
        return "\n".join(s for s in rendered if s).strip()
    return str(content)


def flatten_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """Flatten canonical messages into a single plain-text prompt.

    The non-streaming counterpart to :func:`build_stream_json_stdin` (which
    wraps the equivalent text in a stream-json envelope): used for the
    ``--print`` positional prompt argument. Single-turn user messages render
    their content directly; multi-turn history collapses into a
    "Conversation so far / Current input" structure mirroring the streaming
    flatten, so non-stream and stream see the same prompt contract.
    """
    if not messages:
        return ""
    if len(messages) == 1 and str(messages[0].get("role", "")) == "user":
        return _render_content_for_history(messages[0].get("content", "")).strip()

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
            parts.append(text if i == last_user_idx else f"### User\n{text}")
        elif role == "assistant":
            if text:
                parts.append(f"### Assistant\n{text}")
        elif role == "tool":
            parts.append(f"### Tool result\n{text}")
        else:
            parts.append(f"### {role.capitalize()}\n{text}")

    current_input = parts[-1] if parts else ""
    preamble = ""
    if len(parts) > 1:
        preamble = (
            "## Conversation so far\n\n"
            + "\n\n".join(parts[:-1])
            + "\n\n## Current input\n"
        )
    return (preamble + current_input).strip()


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
            "## Conversation so far\n\n" + "\n\n".join(preamble_parts) + "\n\n## Current input\n"
        )

    flat = (preamble + current_input).strip()

    # The CURRENT turn's images must reach the model as real content blocks —
    # the CLI's stream-json input ingests base64 image blocks natively, and
    # flattening them to "[image attachment]" silently blinded every
    # multi-turn CLI session to chat images and screen-observation frames.
    # Older turns keep the text placeholder (replaying stale frames would
    # bloat every request for no recall value).
    current_images = (
        _image_blocks_of(messages[last_user_idx].get("content"))
        if last_user_idx >= 0
        else []
    )
    content: Any
    if current_images:
        content = [*current_images, {"type": "text", "text": flat}]
    else:
        content = flat

    envelope = {
        "type": "user",
        "message": {"role": "user", "content": content},
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
    if ltype == "rate_limit_event":
        # Quota telemetry the CLI emits before requesting (verified in the
        # 2.1.149/2.1.162 golden captures under tests/llm_client/golden/).
        # Bookkeeping-only — never text-bearing, never an error by itself.
        return None

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
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if text:
                        parts.append(text)
            if parts:
                return {"type": "text_delta", "text": "".join(parts)}
        return None

    if ltype == "stream_event":
        # Claude Code CLI 2.1.x wrapper around the Anthropic Messages SSE
        # event shape. Each ``stream_event`` line carries a single
        # ``event`` dict whose ``type`` is one of message_start /
        # content_block_start / content_block_delta / content_block_stop
        # / message_delta / message_stop. This is the *only* form that
        # actually streams token-by-token under
        # ``--include-partial-messages``; without handling it, the
        # parser falls through to the terminal ``assistant`` envelope
        # (full-message form) which collapses everything into a single
        # delta — exactly the "no streaming visible in the UI" bug.
        ev = line_obj.get("event") or {}
        if not isinstance(ev, dict):
            return None
        etype = str(ev.get("type", ""))
        if etype == "content_block_delta":
            delta = ev.get("delta") or {}
            dtype = str(delta.get("type", ""))
            if dtype == "text_delta":
                return {"type": "text_delta", "text": delta.get("text", "")}
            if dtype == "thinking_delta":
                # The Anthropic SSE shape carries the chunk under
                # ``thinking`` (verified: golden capture, CLI 2.1.149);
                # some shims used ``text``. Reading only ``text`` here
                # silently dropped every thinking token — accept both.
                return {
                    "type": "thinking_delta",
                    "text": delta.get("thinking") or delta.get("text", ""),
                }
            if dtype == "input_json_delta":
                return {"type": "input_json_delta", "delta": delta.get("partial_json", "")}
            return None
        if etype == "content_block_start":
            cb = ev.get("content_block") or {}
            if isinstance(cb, dict) and cb.get("type") == "tool_use":
                return {
                    "type": "tool_use",
                    "id": cb.get("id"),
                    "name": cb.get("name"),
                    "input": cb.get("input") or {},
                }
            return None
        if etype == "content_block_stop":
            return {"type": "content_block_stop"}
        # message_start / message_delta / message_stop carry usage +
        # stop_reason metadata. Not text-bearing; let the accumulator
        # record them silently.
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
# Claude Code: result-envelope field extraction (shared by both parsers)
# ---------------------------------------------------------------------------


def _envelope_usage(obj: Mapping[str, Any]) -> TokenUsage:
    """Extract :class:`TokenUsage` from a Claude Code result envelope.

    The real CLI puts token counts under ``usage`` but the **cost** at the
    *top level* as ``total_cost_usd`` (verified against the recorded
    transcripts in ``tests/llm_client/golden/``); the invented pre-2.2.0
    shape expected ``usage.cost_usd``. Both are read here, in one place,
    because the streaming accumulator and the non-streaming parser used
    to each carry their own copy of this logic and drifted (audit §3.4:
    the non-streaming path never read ``total_cost_usd`` at all, so every
    ``--output-format json`` call priced at $None while the same CLI's
    stream-json output priced correctly).
    """
    usage_in = obj.get("usage", {}) or {}
    cost = usage_in.get("cost_usd")
    if cost is None:
        cost = obj.get("total_cost_usd")
    return TokenUsage(
        input_tokens=int(usage_in.get("input_tokens", 0) or 0),
        output_tokens=int(usage_in.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(usage_in.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(usage_in.get("cache_read_input_tokens", 0) or 0),
        cost_usd=cost,
        duration_ms=obj.get("duration_ms"),
    )


def _envelope_stop_reason(obj: Mapping[str, Any], default: str = "end_turn") -> str:
    """Stop-reason off a result envelope, with a shared default."""
    return str(obj.get("stop_reason") or default)


# ---------------------------------------------------------------------------
# Claude Code: assemble final APIResponse from JSON output
# ---------------------------------------------------------------------------


def parse_json_output_to_response(stdout: bytes, *, model: str) -> APIResponse:
    """Parse the single JSON object emitted by ``--output-format json``.

    The **real** envelope (recorded from Claude Code 2.1.149, see
    ``tests/llm_client/golden/cli-2.1.149-json.json``) carries the
    assistant text as a *top-level string*, not a content array::

        {
          "type": "result", "subtype": "success", "is_error": false,
          "result": "Hi there, friend!",
          "stop_reason": "end_turn",
          "session_id": "...",
          "total_cost_usd": 0.1507...,
          "usage": {"input_tokens": ..., "output_tokens": ..., ...},
          "duration_ms": 4900, "num_turns": 1, ...
        }

    Earlier versions of this parser expected an invented shape with a
    ``content[]`` block array and ``usage.cost_usd`` — so real CLI output
    parsed to an *empty* response with no cost (audit §3.4). The
    ``content[]`` branch is kept for back-compat with hosts that feed
    pre-rendered envelopes through this function, but when it is absent
    the top-level ``result`` string becomes the text block. Cost / usage
    / stop_reason extraction is shared with
    :meth:`StreamJsonAccumulator.finalize` via :func:`_envelope_usage` so
    the two parsers cannot diverge again.
    """
    try:
        obj = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude Code json output not parseable: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("Claude Code json output is not an object")

    # ``tool_use`` blocks in the json output are intentionally dropped
    # for the same reason as ``StreamJsonAccumulator.finalize`` — the
    # CLI handles tool dispatch internally and host pipelines should
    # see only the final assistant text. See ``finalize``'s docstring
    # for the full rationale.
    blocks: List[ContentBlock] = []
    for block in obj.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", "text"))
        if btype == "text":
            blocks.append(ContentBlock(type="text", text=block.get("text", "")))
        elif btype == "thinking":
            blocks.append(ContentBlock(type="thinking", thinking_text=block.get("text", "")))

    # Real-envelope path: no content[] array, the assistant text lives in
    # the top-level ``result`` string.
    if not blocks and isinstance(obj.get("result"), str) and obj["result"]:
        blocks.append(ContentBlock(type="text", text=obj["result"]))

    return APIResponse(
        content=blocks,
        stop_reason=_envelope_stop_reason(obj),
        usage=_envelope_usage(obj),
        model=str(obj.get("model", model) or model),
        # The real envelope has no ``message_id`` — mirror the streaming
        # accumulator, which falls back to the CLI session id so hosts
        # get a stable correlation handle either way.
        message_id=str(obj.get("message_id") or obj.get("session_id") or ""),
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

    Wire-shape telemetry (2.2.0, audit §2.2)
    ----------------------------------------
    Unknown / malformed lines are *counted*, sampled (bounded), and
    surfaced — not just tagged and forgotten. The v2.1.4 incident
    happened precisely because the parser tolerated an unfamiliar wire
    shape (``stream_event``) by falling back to the terminal envelope:
    nobody was told, and streaming silently degraded for weeks. The
    detection cost was already paid; this class now spends it:

    - first unknown line per instance → one rate-limited
      ``logger.warning`` naming the unknown ``type`` and the CLI version
      (when the caller supplied one),
    - counts + first few raw samples are merged into
      ``APIResponse.raw`` at ``finalize()`` so post-hoc diagnosis has the
      evidence inline,
    - callers (``ClaudeCodeCLIClient``) read the public count properties
      to emit ``llm_client.unknown_wire_shape`` events and to enforce
      ``strict_wire=True`` CI canaries.
    """

    #: Bound on retained raw samples — enough for a bug report, small
    #: enough that a hostile/buggy stream cannot balloon memory.
    _SAMPLE_LIMIT = 3
    #: Per-sample truncation (chars).
    _SAMPLE_CHARS = 200

    def __init__(self, model: str, *, cli_version: str = "") -> None:
        self._text_buf: List[str] = []
        self._thinking_buf: List[str] = []
        self._tool_uses: List[Dict[str, Any]] = []
        self._current_tool: Optional[Dict[str, Any]] = None
        self._final_obj: Optional[Dict[str, Any]] = None
        self._message_id = ""
        self._stop_reason = "end_turn"
        self._resolved_model = model
        self._cli_version = cli_version
        self._unknown_count = 0
        self._malformed_count = 0
        self._first_unknown_type: Optional[str] = None
        self._unknown_samples: List[str] = []
        self._warned_unknown = False

    # ── Public ────────────────────────────────────────────────

    @property
    def unknown_line_count(self) -> int:
        """Lines whose ``type`` the parser did not recognise."""
        return self._unknown_count

    @property
    def malformed_line_count(self) -> int:
        """Lines that were not valid JSON objects at all."""
        return self._malformed_count

    @property
    def first_unknown_type(self) -> Optional[str]:
        """The ``type`` value of the first unknown line, if any."""
        return self._first_unknown_type

    @property
    def unknown_samples(self) -> List[str]:
        """Bounded raw samples of unknown / malformed lines."""
        return list(self._unknown_samples)

    def feed(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update state from one stream-json line.

        Returns the list of canonical UI events the line produced
        (``text_delta`` / ``thinking_delta`` / ``tool_use`` / ...).
        Caller is responsible for yielding them to its own consumer.
        Empty list when the line is bookkeeping-only.
        """
        if not isinstance(line, dict):
            return []
        if "__malformed__" in line:
            self._malformed_count += 1
            self._record_unknown(kind="malformed", sample=str(line["__malformed__"]))
            return []
        ltype = str(line.get("type", ""))

        if ltype == "system":
            self._message_id = str(
                line.get("session_id") or line.get("message_id") or self._message_id
            )
            self._resolved_model = str(line.get("model") or self._resolved_model)
            return []

        if ltype == "rate_limit_event":
            # Quota telemetry the CLI emits before requesting (verified in
            # the golden captures). Bookkeeping-only — deliberately known
            # so it never inflates the unknown-shape counters.
            return []

        if ltype == "user":
            return self._feed_user(line)

        if ltype == "assistant":
            return self._feed_assistant(line)

        if ltype == "stream_event":
            return self._feed_stream_event(line)

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

        self._unknown_count += 1
        if self._first_unknown_type is None:
            self._first_unknown_type = ltype or "<missing>"
        self._record_unknown(
            kind="unknown", sample=json.dumps(line, ensure_ascii=False, default=str)
        )
        return [{"type": "cli_unknown", "raw": line}]

    def finalize(self) -> APIResponse:
        """Build the canonical :class:`APIResponse` from accumulated state.

        ``tool_use`` blocks observed during streaming are intentionally
        **dropped** from the assembled response. Claude Code CLI 2.1.x
        runs its agentic loop *internally* (LLM → tool → LLM → tool →
        …); each intermediate turn arrives as its own ``"assistant"``
        envelope and the accumulator collects every block from every
        envelope into the shared buffers below. The CLI has already
        dispatched those tool calls (via its own built-ins or via the
        host's MCP bridge) and emitted the matching ``"user"``
        ``tool_result`` envelopes in the same stream — so including the
        ``tool_use`` blocks in the terminal :class:`APIResponse` would
        push host pipelines (Geny's Stage 10, the canonical reference
        consumer) into trying to re-dispatch tools they have no
        registration for, producing instant ``ERROR (0 ms) — No
        output`` ghost failures for every CLI tool call. Per the Phase
        I design contract:

            Stage 10 receives that assistant message, sees no
            ``tool_use`` blocks (they were executed inside the CLI),
            and naturally no-ops.

        Hosts that *do* want the raw tool_use record can still recover
        it from the per-line stream events the accumulator yields
        through ``feed()`` (each ``tool_use`` block produces a
        ``{"type": "tool_use", "id": ..., "name": ..., "input": ...}``
        event).
        """
        # Flush any unclosed tool — the message form often skips
        # ``content_block_stop`` entirely. We still call this so the
        # accumulator's internal state is consistent for callers that
        # rely on ``_tool_uses`` directly; only the *response* blocks
        # below skip them.
        self._close_current_tool()

        blocks: List[ContentBlock] = []
        if self._thinking_buf:
            blocks.append(ContentBlock(type="thinking", thinking_text="".join(self._thinking_buf)))
        if self._text_buf:
            blocks.append(ContentBlock(type="text", text="".join(self._text_buf)))

        # ``raw`` starts as a *copy* of the terminal result envelope —
        # wire telemetry is merged in (not clobbered over it) so the
        # original CLI fields stay readable for hosts that inspect raw.
        raw_out: Dict[str, Any] = dict(self._final_obj or {})
        if self._unknown_count or self._malformed_count:
            raw_out["unknown_line_count"] = self._unknown_count
            raw_out["malformed_line_count"] = self._malformed_count
            raw_out["first_unknown_type"] = self._first_unknown_type
            raw_out["unknown_samples"] = list(self._unknown_samples)

        return APIResponse(
            content=blocks,
            stop_reason=_envelope_stop_reason(self._final_obj or {}, default=self._stop_reason),
            usage=_envelope_usage(self._final_obj or {}),
            model=self._resolved_model,
            message_id=self._message_id,
            raw=raw_out,
        )

    # ── Internals ─────────────────────────────────────────────

    def _record_unknown(self, *, kind: str, sample: str) -> None:
        """Bounded sample retention + once-per-instance warning.

        One warning per accumulator instance (≈ one per CLI call), not
        per line: a wire change typically affects *every* line of a
        shape, and a per-line warning would flood logs at token rate —
        the same failure mode the embedding 401-spam incident had.
        """
        if len(self._unknown_samples) < self._SAMPLE_LIMIT:
            self._unknown_samples.append(sample[: self._SAMPLE_CHARS])
        if not self._warned_unknown:
            self._warned_unknown = True
            logger.warning(
                "Claude Code stream-json contained an unrecognised line "
                "(kind=%s, type=%s, cli_version=%s). The line was tolerated, "
                "but this usually means the CLI wire format moved — "
                "sample: %.200s",
                kind,
                self._first_unknown_type or "<n/a>",
                self._cli_version or "unknown",
                sample,
            )

    def _feed_user(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Surface CLI-executed tool results from ``user`` envelopes.

        Claude Code CLI 2.1.x runs its agentic loop internally; after
        executing a tool it emits the matching ``tool_result`` as a
        ``{"type": "user"}`` envelope in the same stream (the
        :meth:`finalize` docstring has relied on this wire fact since
        Phase I). Two flavours arrive here:

        - the echo of OUR OWN stdin input (plain string / text-block
          content) — bookkeeping-only, no event;
        - ``tool_result`` blocks for tools the CLI dispatched itself —
          each becomes a canonical ``{"type": "tool_result", ...}``
          event so consumers (Stage 6's ``api.tool_result`` state
          event, host tool timelines) can show what the CLI actually
          ran, paired with the ``tool_use`` event that preceded it.

        Handling ``user`` explicitly also fixes a telemetry bug
        (2.2.0): the line type was missing from ``feed``'s dispatch
        entirely, so every tool-using CLI session inflated
        ``unknown_line_count`` — one ``llm_client.unknown_wire_shape``
        warning per call and a spurious hard failure under
        ``strict_wire=True``, for a line shape the parser's own
        docstrings describe as expected.
        """
        message = line.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        events: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    events.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": block.get("content"),
                            "is_error": bool(block.get("is_error", False)),
                        }
                    )
        return events

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

    def _feed_stream_event(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single Claude Code CLI ``stream_event`` line.

        Claude Code CLI 2.1.x wraps the Anthropic Messages SSE event
        shape inside a ``{"type":"stream_event","event":{...}}`` envelope
        when ``--include-partial-messages`` is on. The accumulator turns
        each one into the same canonical UI event the legacy ``assistant``
        delta form produced, so downstream consumers (Geny's session
        logger streaming pipe, Stage 10's tool dispatch, anything else
        that reads ``feed()`` output) get token-level deltas without
        having to learn the new wire format.

        Mappings:
          - ``message_start``        → record id + usage; no UI event
          - ``content_block_start``  → tool_use carries the tool record
          - ``content_block_delta``  → text_delta / thinking_delta /
                                       input_json_delta
          - ``content_block_stop``   → close current tool; emit stop
          - ``message_delta``        → record stop_reason; no UI event
          - ``message_stop``         → no UI event (caller emits the
                                       final ``message_complete`` after
                                       ``finalize()``)
        """
        ev = line.get("event") or {}
        if not isinstance(ev, dict):
            return []
        etype = str(ev.get("type", ""))

        if etype == "message_start":
            msg = ev.get("message") or {}
            if isinstance(msg, dict):
                self._message_id = str(msg.get("id") or self._message_id)
                self._resolved_model = str(msg.get("model") or self._resolved_model)
                usage = msg.get("usage")
                if isinstance(usage, dict) and self._final_obj is None:
                    self._final_obj = {"usage": usage}
            return []

        if etype == "content_block_start":
            cb = ev.get("content_block") or {}
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
            return []

        if etype == "content_block_delta":
            delta = ev.get("delta") or {}
            dtype = str(delta.get("type", ""))
            if dtype == "text_delta":
                text = str(delta.get("text", ""))
                if text:
                    self._text_buf.append(text)
                    return [{"type": "text_delta", "text": text}]
                return []
            if dtype == "thinking_delta":
                # Wire key is ``thinking`` on the real CLI (Anthropic SSE
                # shape — verified in the 2.1.149 golden capture), ``text``
                # on older shims. Reading only ``text`` dropped every
                # thinking token and the terminal envelope then re-recorded
                # the full thinking block, masking the loss entirely.
                text = str(delta.get("thinking") or delta.get("text") or "")
                if text:
                    self._thinking_buf.append(text)
                    return [{"type": "thinking_delta", "text": text}]
                return []
            if dtype == "input_json_delta":
                partial = str(delta.get("partial_json", ""))
                if self._current_tool is not None:
                    self._current_tool.setdefault("_partial_json", "")
                    self._current_tool["_partial_json"] += partial
                return [{"type": "input_json_delta", "delta": partial}]
            return []

        if etype == "content_block_stop":
            self._close_current_tool()
            return [{"type": "content_block_stop"}]

        if etype == "message_delta":
            delta = ev.get("delta") or {}
            if isinstance(delta, dict):
                sr = delta.get("stop_reason")
                if sr:
                    self._stop_reason = str(sr)
            # ``message_delta`` carries the *cumulative* usage (the
            # message_start snapshot only has a couple of output tokens),
            # so newer wins — verified against the 2.1.149 golden capture
            # where message_start says output_tokens=7 and message_delta
            # says 112. The terminal ``result`` line still overrides both.
            usage = ev.get("usage")
            if isinstance(usage, dict):
                if self._final_obj is None:
                    self._final_obj = {"usage": usage}
                else:
                    self._final_obj["usage"] = usage
            return []

        if etype == "message_stop":
            return []

        return []

    def _feed_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a full assistant message envelope's content array.

        Emits synthetic per-block delta events so UI consumers see the
        same canonical shape they would with true streaming, then
        records the blocks for the eventual :class:`APIResponse`.

        ``stream_event``-form coexistence — Claude Code CLI 2.1.x with
        ``--include-partial-messages`` emits BOTH per-token
        ``stream_event`` lines AND a terminal ``assistant`` envelope
        containing the same text in full. If we've already accumulated
        text/thinking via the ``stream_event`` deltas, the envelope is
        a duplicate and re-recording it would double every assistant
        message. ``tool_use`` blocks are kept either way — tool calls
        arrive via ``content_block_start``, not via deltas, so the
        envelope is the canonical record for them.
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

        already_streamed = bool(self._text_buf) or bool(self._thinking_buf)

        events: List[Dict[str, Any]] = []
        content = message.get("content") or []
        if not isinstance(content, list):
            return events
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", ""))
            if btype == "text":
                if already_streamed:
                    continue
                text = str(block.get("text", ""))
                if text:
                    self._text_buf.append(text)
                    events.append({"type": "text_delta", "text": text})
            elif btype == "thinking":
                if already_streamed:
                    continue
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
    cli_version: str = "",
) -> APIResponse:
    """Drain a stream-json output and return a canonical APIResponse.

    Used by ``ClaudeCodeCLIClient._send`` when ``request.stream=True``.
    Thin wrapper around :class:`StreamJsonAccumulator` so the
    streaming + non-streaming consumer paths share one parser — Claude
    Code's stream-json shape (delta vs full-message) varies by CLI
    version and ``--include-partial-messages``, and we never want the
    two paths to drift again.

    Malformed lines flow *into* the accumulator (which counts and
    samples them) instead of being skipped at this layer — pre-2.2.0
    they were dropped here before the telemetry could see them, which
    is exactly the masking channel audit §2.2 calls out. Counts surface
    via ``APIResponse.raw`` (``unknown_line_count`` etc.).
    """
    from geny_executor.llm_client._cli_runtime import parse_stream_json_line

    accum = StreamJsonAccumulator(model=model, cli_version=cli_version)
    async for raw in stream:
        line = parse_stream_json_line(raw)
        if line is None:
            continue
        # ``error`` envelopes from the CLI need to raise so the caller's
        # CLIProtocolError path runs — match the prior behaviour exactly.
        # (Malformed lines have no ``type`` key, so they fall through to
        # ``feed`` for counting.)
        if str(line.get("type", "")) == "error":
            raise RuntimeError(f"Claude Code CLI reported error: {line.get('message') or line!r}")
        accum.feed(line)

    return accum.finalize()


# Copilot CLI helpers (compose_copilot_prompt / copilot_argv /
# parse_plain_text_to_response) were removed in 2.0.6. ``gh copilot``
# is one-shot text-in / text-out with no streaming, no tool round-trip,
# and no MCP support, so it could not host the pipeline's Stage-10
# dispatch loop. The ``CopilotCLIClient`` and its registry entry are
# also gone — see the matching commit message.
