"""A conversation outlives the model answering it — including its images.

An image attached while Claude was answering can reach a text-only local
model on the very next turn, through a failover or a model switch. Three
things could happen there and only one is honest:

  * send it anyway — the wire rejects the request and the turn dies for a
    reason the user cannot act on;
  * strip it silently — the model answers confidently about a picture it
    never received;
  * say an image was here and could not be shown.

This pins the third, and pins that the backends which CAN see still get the
real image.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.base import BaseClient, ClientCapabilities

IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
}


def _messages() -> List[Dict[str, Any]]:
    return [{"role": "user", "content": [IMAGE, {"type": "text", "text": "what is this?"}]}]


def _tool_result_messages() -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "captured"},
                IMAGE,
            ],
        }
    ]


class _Client(BaseClient):
    provider = "probe"

    def __init__(self, caps: ClientCapabilities, sink: list) -> None:
        super().__init__(api_key="k", event_sink=sink.append)
        self.capabilities = caps

    async def _send(self, request, *, purpose: str = ""):  # pragma: no cover
        raise NotImplementedError

    async def create_message_stream(self, **_kw):  # pragma: no cover
        raise NotImplementedError
        yield {}

    def build(self, messages):
        return self._build_request(
            model_config=ModelConfig(model="m"), messages=messages, system="", tools=None,
            tool_choice=None, stream=False,
        )


def _blocks(request) -> List[Dict[str, Any]]:
    return request.messages[0]["content"]


class TestABackendThatCannotSee:
    def test_the_image_becomes_a_note_not_a_silent_deletion(self) -> None:
        sink: list = []
        client = _Client(ClientCapabilities(supports_vision=False), sink)
        blocks = _blocks(client.build(_messages()))
        assert all(b.get("type") != "image" for b in blocks)
        note = next(b for b in blocks if b["type"] == "text" and "image" in b["text"])
        assert "cannot see" in note["text"]

    def test_the_question_survives(self) -> None:
        client = _Client(ClientCapabilities(supports_vision=False), [])
        blocks = _blocks(client.build(_messages()))
        assert any(b.get("text") == "what is this?" for b in blocks)

    def test_the_host_is_told(self) -> None:
        """Same contract as every other declared drop: visible, not silent."""
        sink: list = []
        client = _Client(ClientCapabilities(supports_vision=False), sink)
        client.build(_messages())
        assert any("image" in str(event) for event in sink), sink


class TestABackendThatCanSee:
    def test_the_image_is_untouched(self) -> None:
        client = _Client(
            ClientCapabilities(supports_vision=True, supports_vision_tool_results=True), []
        )
        blocks = _blocks(client.build(_messages()))
        assert blocks[0]["source"]["data"] == "iVBORw0KGgo="

    def test_nothing_is_reported_dropped(self) -> None:
        sink: list = []
        client = _Client(
            ClientCapabilities(supports_vision=True, supports_vision_tool_results=True), sink
        )
        client.build(_messages())
        assert not [e for e in sink if "image" in str(e)]


class TestSeeingAUserImageButNotAToolOne:
    """Some backends take a multimodal user message and reject list-type tool
    content outright — the two are separate declarations for that reason."""

    def test_the_user_image_survives_and_the_tool_image_does_not(self) -> None:
        caps = ClientCapabilities(supports_vision=True, supports_vision_tool_results=False)
        client = _Client(caps, [])
        assert _blocks(client.build(_messages()))[0]["type"] == "image"
        assert all(b.get("type") != "image" for b in _blocks(client.build(_tool_result_messages())))


class TestWhatTheShippedClientsDeclare:
    @pytest.mark.parametrize(
        "module,cls",
        [
            ("anthropic", "AnthropicClient"),
            ("openai", "OpenAIClient"),
            ("google", "GoogleClient"),
            ("claude_code_tokens", "ClaudeCodeTokenClient"),
            ("codex", "CodexResponsesClient"),
            ("router", "RouterClient"),
        ],
    )
    def test_the_vision_backends_say_so(self, module: str, cls: str) -> None:
        import importlib

        client_cls = getattr(importlib.import_module(f"geny_executor.llm_client.{module}"), cls)
        assert client_cls.capabilities.supports_vision is True

    def test_a_local_endpoint_stays_conservative(self) -> None:
        """Whether a served model can see is the operator's to declare —
        guessing yes is how an image reaches a text-only checkpoint."""
        from geny_executor.llm_client.vllm import VLLMClient

        assert VLLMClient.capabilities.supports_vision is False
