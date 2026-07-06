"""CLI vision wire (2.45.0): image blocks must actually reach the model.

Two regressions this locks down:

* Non-streaming ``create_message`` built a text-only ``--print`` positional
  prompt, so every vision call (screen-observation captioning, whiteboard
  describe) lost its image and the model answered "I don't see an image…".
  Requests carrying image blocks now switch to the stream-json wire, which
  ingests base64 images natively (verified against claude CLI 2.1.185).

* Multi-turn ``build_stream_json_stdin`` flattened the CURRENT turn's
  content through the history renderer, turning image blocks into the
  literal text ``[image attachment]`` — silently blinding every multi-turn
  CLI session (the persona) to chat images and screen frames. The last
  user message's images now ride along as real content blocks.
"""
import json

import pytest

from geny_executor.llm_client.base import APIRequest
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient
from geny_executor.llm_client.translators._cli import (
    build_stream_json_stdin,
    messages_have_images,
)


def _img(data: str = "AAAA") -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _txt(text: str) -> dict:
    return {"type": "text", "text": text}


# ── messages_have_images ─────────────────────────────────────────────


def test_messages_have_images_detects_blocks():
    assert messages_have_images([{"role": "user", "content": [_img(), _txt("hi")]}])


def test_messages_have_images_false_for_text():
    assert not messages_have_images([{"role": "user", "content": "hello"}])
    assert not messages_have_images([{"role": "user", "content": [_txt("hi")]}])


# ── build_stream_json_stdin: current-turn image preservation ─────────


def _envelope(messages) -> dict:
    return json.loads(build_stream_json_stdin(messages).decode("utf-8"))


def test_single_turn_passthrough_keeps_blocks():
    env = _envelope([{"role": "user", "content": [_img("XYZ"), _txt("color?")]}])
    types = [b["type"] for b in env["message"]["content"]]
    assert types == ["image", "text"]
    assert env["message"]["content"][0]["source"]["data"] == "XYZ"


def test_multi_turn_keeps_last_user_images_as_blocks():
    env = _envelope(
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [_img("CURRENT"), _txt("what is on screen?")]},
        ]
    )
    content = env["message"]["content"]
    assert isinstance(content, list)
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["data"] == "CURRENT"
    # Flattened history rides in the text block alongside the image.
    text = "\n".join(b["text"] for b in content if b["type"] == "text")
    assert "### Assistant" in text
    assert "what is on screen?" in text


def test_multi_turn_older_images_stay_placeholders():
    env = _envelope(
        [
            {"role": "user", "content": [_img("OLD"), _txt("earlier frame")]},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "text only now"},
        ]
    )
    content = env["message"]["content"]
    # No image in the last user turn → plain flattened string, with the old
    # frame reduced to its placeholder.
    assert isinstance(content, str)
    assert "OLD" not in content
    assert "[image attachment]" in content


def test_multi_turn_text_only_unchanged():
    env = _envelope(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert isinstance(env["message"]["content"], str)


# ── client: non-stream + images → stream-json wire ───────────────────


class _CaptureRunner:
    """Stub runner recording the wire mode ``_send`` chose."""

    def __init__(self):
        self.argv = None
        self.stdin = b""
        self.streamed = False

    async def run_oneshot(self, argv, stdin=None):  # pragma: no cover - guard
        self.argv = argv
        raise AssertionError("image request must not use the one-shot wire")

    def stream(self, argv, stdin_iter=None):
        self.argv = argv
        self.streamed = True

        async def _gen():
            if stdin_iter is not None:
                async for chunk in stdin_iter:
                    self.stdin += chunk
            assistant = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A red square."}],
                },
                "session_id": "s",
            }
            done = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "A red square.",
                "session_id": "s",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            yield (json.dumps(assistant) + "\n").encode("utf-8")
            yield (json.dumps(done) + "\n").encode("utf-8")

        return _gen()


@pytest.mark.asyncio
async def test_nonstream_image_request_rides_stream_wire(monkeypatch):
    client = ClaudeCodeCLIClient(binary_path="/bin/true", api_key="sk-test")
    runner = _CaptureRunner()
    monkeypatch.setattr(client, "_make_runner", lambda: runner)

    async def _ver():
        return "test"

    monkeypatch.setattr(client, "_ensure_cli_version", _ver)

    request = APIRequest(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": [_img("PIXELS"), _txt("describe")]}],
        stream=False,
    )
    response = await client._send(request)

    assert runner.streamed, "non-stream vision call must switch to stream-json"
    assert "--input-format" in runner.argv
    assert "PIXELS" in runner.stdin.decode("utf-8")
    assert response.text == "A red square."


@pytest.mark.asyncio
async def test_nonstream_text_request_keeps_oneshot_wire(monkeypatch):
    client = ClaudeCodeCLIClient(binary_path="/bin/true", api_key="sk-test")

    class _OneshotRunner:
        def __init__(self):
            self.argv = None

        async def run_oneshot(self, argv, stdin=None):
            self.argv = argv

            class _R:
                returncode = 0
                stdout = json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "ok",
                        "session_id": "s",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode("utf-8")
                stderr = b""

            return _R()

        def stream(self, argv, stdin_iter=None):  # pragma: no cover - guard
            raise AssertionError("text-only non-stream must stay one-shot")

    runner = _OneshotRunner()
    monkeypatch.setattr(client, "_make_runner", lambda: runner)

    async def _ver():
        return "test"

    monkeypatch.setattr(client, "_ensure_cli_version", _ver)

    request = APIRequest(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "say ok"}],
        stream=False,
    )
    response = await client._send(request)
    assert response.text == "ok"
    assert "--output-format" in runner.argv
