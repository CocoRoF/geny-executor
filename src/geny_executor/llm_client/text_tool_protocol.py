"""Text tool-call protocol for token-only backends.

A backend that can only *generate text* — the `claude` CLI with every
built-in tool switched off — still has to drive the pipeline's tool loop.
The harness does it by contract instead of by API: the tool schemas and a
strict call format go into the system prompt, the model answers with
`<tool_call>` blocks, and this module turns those blocks back into canonical
`tool_use` content, so Stage 10 executes them exactly as it would for a
native tool-calling API. The CLI never runs anything itself.

Everything here is pure (no I/O) so it can be tested without a model.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

PROTOCOL_HEADER = """\
# Tool use (Geny harness)

You are the reasoning engine of an agent harness. You do not execute tools
yourself — the harness executes them and shows you the results.

To call a tool, write a block in EXACTLY this form (one JSON object per block,
with the tool's parameters nested inside "arguments"):

<tool_call>
{"name": "ToolName", "arguments": {"param": "value"}}
</tool_call>

For example, to read a file:

<tool_call>
{"name": "Read", "arguments": {"file_path": "/abs/path/notes.txt"}}
</tool_call>

Rules:
- `arguments` must be a JSON object that satisfies the tool's input_schema.
- You may emit several <tool_call> blocks in a row for independent calls.
- After your last <tool_call> block, END YOUR MESSAGE immediately. Do not
  describe the outcome or claim success — you have not seen the result yet.
  The harness replies with <tool_result> blocks in the next message.
- Only call tools listed below. If no tool is needed, answer normally and do
  not mention this protocol.
- Text you write before a tool call is shown to the user, so keep it brief.
"""


def new_tool_id() -> str:
    return f"toolu_geny_{uuid.uuid4().hex[:20]}"


def _tool_entry(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name"),
        "description": tool.get("description") or "",
        "input_schema": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
    }


def render_system_prompt(system: Any, tools: Optional[list[dict[str, Any]]]) -> str:
    """The engine's system prompt (string or block list) plus, when there
    are tools, the calling protocol and the catalogue."""
    base = system_text(system)
    if not tools:
        return base
    catalogue = json.dumps([_tool_entry(t) for t in tools if t.get("name")], ensure_ascii=False)
    parts = [p for p in (base, PROTOCOL_HEADER, "## Available tools\n<tools>\n" + catalogue + "\n</tools>") if p]
    return "\n\n".join(parts)


def system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        out = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                out.append(block)
        return "\n\n".join(p for p in out if p)
    return str(system or "")


# ── transcript ────────────────────────────────────────────────────────
def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _render_blocks(role: str, content: Any) -> tuple[str, list[dict[str, Any]]]:
    """(text, images) for one canonical message."""
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text") or ""))
        elif kind == "tool_use":
            call = {"name": block.get("name"), "arguments": block.get("input") or {}}
            texts.append(f"{OPEN}\n{json.dumps(call, ensure_ascii=False)}\n{CLOSE}")
        elif kind == "tool_result":
            status = ' is_error="true"' if block.get("is_error") else ""
            texts.append(
                f'<tool_result id="{block.get("tool_use_id", "")}"{status}>\n'
                f"{_result_text(block.get('content'))}\n</tool_result>"
            )
        elif kind == "image":
            src = block.get("source")
            if isinstance(src, dict):
                images.append({"type": "image", "source": src})
            texts.append("[image attached]")
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            texts.append(_result_text([block]))
    return "\n".join(t for t in texts if t), images


def render_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical history → the content blocks of ONE stream-json user
    envelope (the CLI accepts only `user` envelopes). The latest message is
    the live input; earlier ones become a tagged transcript the model reads
    as its own conversation. Images of the latest message are kept as real
    image blocks."""
    if not messages:
        return [{"type": "text", "text": ""}]
    *history, last = messages
    last_text, images = _render_blocks(str(last.get("role", "user")), last.get("content", ""))

    if not history and str(last.get("role")) == "user":
        return [*images, {"type": "text", "text": last_text}]

    lines = ["<conversation>"]
    for message in history:
        role = str(message.get("role", "user"))
        text, _ = _render_blocks(role, message.get("content", ""))
        if not text:
            continue
        label = "user" if role == "user" else "assistant"
        lines.append(f'<turn role="{label}">\n{text}\n</turn>')
    lines.append("</conversation>")
    head = "\n".join(lines)
    if str(last.get("role")) == "assistant":
        tail = f"{last_text}\n\nContinue your previous assistant message."
    elif "<tool_result" in last_text:
        tail = (
            "The harness executed your tool calls:\n\n" + last_text +
            "\n\nContinue as the assistant: call more tools or give the final answer."
        )
    else:
        tail = last_text
    return [*images, {"type": "text", "text": f"{head}\n\n{tail}"}]


# ── parsing ───────────────────────────────────────────────────────────
@dataclass
class ParsedCall:
    id: str
    name: str
    arguments: dict[str, Any]


def _loads_lenient(raw: str) -> Optional[Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # the first complete object, ignoring what follows — models add a stray
    # closing brace or a word after the JSON (observed with haiku)
    start = raw.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
            return obj
        except Exception:
            return None
    return None


def parse_call(body: str, known: Optional[set[str]] = None) -> Optional[ParsedCall]:
    data = _loads_lenient(body)
    if not isinstance(data, dict):
        return None
    # tolerate OpenAI-shaped {"function": {"name", "arguments": "<json>"}}
    if isinstance(data.get("function"), dict):
        data = {**data["function"], "id": data.get("id")}
    name = data.get("name") or data.get("tool") or data.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    if any(k in data for k in ("arguments", "input", "parameters", "args")):
        args = data.get("arguments", data.get("input", data.get("parameters", data.get("args", {}))))
    else:
        # arguments written flat next to the name: {"name": "Write", "file_path": …}
        args = {k: v for k, v in data.items() if k not in ("name", "tool", "tool_name", "id", "type")}
    if isinstance(args, str):
        parsed = _loads_lenient(args)
        args = parsed if isinstance(parsed, dict) else {"input": args}
    if not isinstance(args, dict):
        args = {}
    if known and name not in known:
        # a near-miss in case is still the tool the model meant
        lowered = {k.lower(): k for k in known}
        name = lowered.get(name.lower(), name)
    return ParsedCall(id=new_tool_id(), name=name, arguments=args)


@dataclass
class StreamSplitter:
    """Incremental splitter: visible text flows out as it arrives, anything
    inside `<tool_call>…</tool_call>` is withheld and parsed.

    `finished` turns True once at least one call closed and the model then
    starts writing something that is not another call — by protocol that
    is the model narrating a result it does not have, so the caller stops
    reading (and stops paying for) the rest.
    """

    known: Optional[set[str]] = None
    calls: list[ParsedCall] = field(default_factory=list)
    visible: list[str] = field(default_factory=list)
    finished: bool = False
    malformed: int = 0
    _buf: str = ""
    _inside: bool = False

    def feed(self, text: str) -> str:
        """Returns the visible text released by this chunk."""
        if self.finished or not text:
            return ""
        self._buf += text
        out: list[str] = []
        while self._buf and not self.finished:
            if self._inside:
                # the first closing tag may sit INSIDE the JSON (an argument
                # that mentions it): take the first one whose body parses
                call = None
                search = 0
                while True:
                    end = self._buf.find(CLOSE, search)
                    if end < 0:
                        break
                    call = parse_call(self._buf[:end], self.known)
                    if call is not None:
                        self._buf = self._buf[end + len(CLOSE):]
                        break
                    search = end + len(CLOSE)
                if call is None:
                    break  # need more text (or close() decides)
                self._inside = False
                self.calls.append(call)
                continue
            start = self._buf.find(OPEN)
            if start >= 0:
                before = self._buf[:start]
                if self.calls and before.strip():
                    self.finished = True
                    self._buf = ""
                    break
                if not self.calls:
                    out.append(before)
                self._buf = self._buf[start + len(OPEN):]
                self._inside = True
                continue
            # keep a tail that could be the start of "<tool_call>"
            keep = _partial_prefix(self._buf, OPEN)
            release = self._buf[: len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep:]
            if self.calls:
                if release.strip():
                    self.finished = True
                    self._buf = ""
            else:
                out.append(release)
            break
        released = "".join(out)
        if released:
            self.visible.append(released)
        return released

    def close(self) -> str:
        """End of stream: flush what is left. An unterminated call is
        parsed if its JSON is complete (models sometimes drop the closing
        tag at the very end)."""
        tail = ""
        if self._inside:
            body = self._buf
            end = body.rfind(CLOSE)
            call = parse_call(body[:end] if end >= 0 else body, self.known)
            if call is not None:
                self.calls.append(call)
            elif not self.calls:
                # prose that merely mentions the tag: give the text back
                tail = OPEN + body
                self.malformed += 1
            else:
                self.malformed += 1
        elif not self.finished and not self.calls:
            tail = self._buf
        self._buf = ""
        self._inside = False
        if tail:
            self.visible.append(tail)
        return tail

    @property
    def text(self) -> str:
        return "".join(self.visible).strip()


def _partial_prefix(buf: str, token: str) -> int:
    """Length of the longest suffix of `buf` that is a prefix of `token`."""
    for size in range(min(len(buf), len(token) - 1), 0, -1):
        if token.startswith(buf[-size:]):
            return size
    return 0


def tool_names(tools: Optional[Iterable[dict[str, Any]]]) -> set[str]:
    return {str(t.get("name")) for t in tools or [] if t.get("name")}
