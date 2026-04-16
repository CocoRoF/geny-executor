"""Cross-provider translation utilities for Stage 6 (API).

Converts between the geny-executor canonical format (Anthropic-style)
and provider-native formats for OpenAI and Google Gemini.

The canonical format is used throughout the pipeline:
  - Messages: List[Dict] with role="user"|"assistant", content=str|List[Dict]
  - Tools: List[Dict] with name, description, input_schema
  - Tool calls: {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
  - Tool results: {"type": "tool_result", "tool_use_id": ..., "content": ...}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stop Reason Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Provider-native → Canonical
OPENAI_STOP_REASON: Dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "content_filter",
}

GOOGLE_STOP_REASON: Dict[str, str] = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "FINISH_REASON_UNSPECIFIED": "end_turn",
}


def normalize_stop_reason(reason: str, provider: str) -> str:
    """Convert provider-specific stop reason to canonical format."""
    if provider == "openai":
        return OPENAI_STOP_REASON.get(reason, reason)
    if provider == "google":
        return GOOGLE_STOP_REASON.get(reason, reason)
    return reason  # Anthropic is already canonical


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Definition Conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def canonical_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonical tool defs → OpenAI function tools.

    Canonical: {"name", "description", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def canonical_tools_to_google(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonical tool defs → Google function declarations.

    Canonical: {"name", "description", "input_schema": {...}}
    Google:    [{"functionDeclarations": [{"name", "description", "parameters": {...}}]}]
    """
    return [
        {
            "functionDeclarations": [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        }
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Choice Conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def canonical_tool_choice_to_openai(
    choice: Optional[Dict[str, Any]],
) -> Union[str, Dict[str, Any]]:
    """Canonical (Anthropic) tool_choice → OpenAI tool_choice.

    Canonical: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "fn"}
    OpenAI:    "auto" | "required" | "none" | {"type": "function", "function": {"name": "fn"}}
    """
    if choice is None:
        return "auto"
    t = choice.get("type", "auto")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


def canonical_tool_choice_to_google(
    choice: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Canonical (Anthropic) tool_choice → Google toolConfig.

    Canonical: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "fn"}
    Google:    {"functionCallingConfig": {"mode": "AUTO|ANY|NONE", "allowedFunctionNames": [...]}}
    """
    if choice is None:
        return {"functionCallingConfig": {"mode": "AUTO"}}
    t = choice.get("type", "auto")
    mode_map = {"auto": "AUTO", "any": "ANY", "none": "NONE"}
    mode = mode_map.get(t, "AUTO")
    config: Dict[str, Any] = {"functionCallingConfig": {"mode": mode}}
    if t == "tool":
        config["functionCallingConfig"]["mode"] = "ANY"
        config["functionCallingConfig"]["allowedFunctionNames"] = [choice["name"]]
    return config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Message Content Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def blocks_to_text(content: Any) -> str:
    """Extract plain text from canonical content (str or List[Dict])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # skip thinking blocks
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)


def split_tool_results(
    content: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split canonical content blocks into (tool_results, other_blocks)."""
    tool_results: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_results.append(block)
        else:
            other.append(block)
    return tool_results, other


def split_tool_uses(
    content: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split canonical content blocks into (text_blocks, tool_use_blocks)."""
    text_blocks: List[Dict[str, Any]] = []
    tool_uses: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_uses.append(block)
        else:
            text_blocks.append(block)
    return text_blocks, tool_uses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI Message Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def canonical_messages_to_openai(
    messages: List[Dict[str, Any]],
    system: Any = "",
) -> List[Dict[str, Any]]:
    """Canonical messages + system → OpenAI messages.

    Key transformations:
      - system → prepend as {"role": "developer"} message
      - assistant tool_use blocks → message.tool_calls array
      - user tool_result blocks → separate {"role": "tool"} messages
    """
    result: List[Dict[str, Any]] = []

    # System prompt → developer role message
    if system:
        sys_text = blocks_to_text(system)
        if sys_text:
            result.append({"role": "developer", "content": sys_text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                tool_results, other = split_tool_results(content)
                # tool_result blocks → separate "tool" role messages
                for tr in tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": str(tr.get("content", "")),
                        }
                    )
                # remaining blocks → user message
                if other:
                    result.append({"role": "user", "content": blocks_to_text(other)})
            else:
                result.append({"role": "user", "content": str(content)})

        elif role == "assistant":
            if isinstance(content, list):
                text_blocks, tool_uses = split_tool_uses(content)
                msg_dict: Dict[str, Any] = {"role": "assistant"}
                text = blocks_to_text(text_blocks)
                if text:
                    msg_dict["content"] = text
                if tool_uses:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {}), ensure_ascii=False),
                            },
                        }
                        for tc in tool_uses
                    ]
                result.append(msg_dict)
            else:
                result.append({"role": "assistant", "content": str(content)})

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Google Message Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def canonical_messages_to_google(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Canonical messages → Google Gemini contents.

    Key transformations:
      - role "assistant" → role "model"
      - tool_use blocks → functionCall parts
      - tool_result blocks → functionResponse parts
      - text blocks → text parts
    """
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Google uses "model" instead of "assistant"
        g_role = "model" if role == "assistant" else "user"

        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    parts.append({"text": str(block)})
                    continue

                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append({"text": text})
                elif btype == "tool_use":
                    parts.append(
                        {
                            "functionCall": {
                                "name": block.get("name", ""),
                                "args": block.get("input", {}),
                                "id": block.get("id", ""),
                            }
                        }
                    )
                elif btype == "tool_result":
                    parts.append(
                        {
                            "functionResponse": {
                                "name": block.get("name", ""),
                                "id": block.get("tool_use_id", ""),
                                "response": {"result": str(block.get("content", ""))},
                            }
                        }
                    )

        if parts:
            contents.append({"role": g_role, "parts": parts})

    return contents


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thinking / Reasoning Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def canonical_thinking_to_openai(thinking: Optional[Dict[str, Any]]) -> Optional[str]:
    """Canonical thinking config → OpenAI reasoning_effort.

    Canonical: {"type": "enabled", "budget_tokens": N} | {"type": "adaptive"}
    OpenAI:    reasoning_effort = "low" | "medium" | "high"

    Mapping by budget_tokens:
      < 5000     → "low"
      5000~20000 → "medium"
      > 20000    → "high"
      adaptive   → "medium"
    """
    if thinking is None:
        return None

    t = thinking.get("type", "disabled")
    if t == "disabled":
        return None
    if t == "adaptive":
        return "medium"

    budget = thinking.get("budget_tokens", 10000)
    if budget < 5000:
        return "low"
    if budget <= 20000:
        return "medium"
    return "high"


def canonical_thinking_to_google(thinking: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Canonical thinking config → Google thinkingConfig.

    Canonical: {"type": "enabled", "budget_tokens": N} | {"type": "adaptive"}
    Google:    {"thinkingLevel": "low"|"medium"|"high"} or {"thinkingBudget": N}
    """
    if thinking is None:
        return None

    t = thinking.get("type", "disabled")
    if t == "disabled":
        return None
    if t == "adaptive":
        return {"includeThoughts": True}

    budget = thinking.get("budget_tokens")
    if budget:
        if budget < 5000:
            return {"thinkingLevel": "low", "includeThoughts": True}
        if budget <= 20000:
            return {"thinkingLevel": "medium", "includeThoughts": True}
        return {"thinkingLevel": "high", "includeThoughts": True}
    return {"includeThoughts": True}
