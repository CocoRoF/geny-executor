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
    """The shape a tool actually returns an image in.

    Canonical history nests it INSIDE the tool_result's own content list
    (``_canonical._tool_result_text_and_images`` reads it from there), not
    beside the tool_result. A scan that only walks the message's top-level
    blocks therefore finds nothing to lower on exactly the turn that needs
    it — a screenshot — and the raw image goes to a model that cannot see.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "captured"}, IMAGE],
                }
            ],
        }
    ]


def _tool_result_images(request) -> List[Dict[str, Any]]:
    """Every image block still nested in a tool_result of *request*."""
    found: List[Dict[str, Any]] = []
    for message in request.messages:
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                for inner in block.get("content") or []:
                    if isinstance(inner, dict) and inner.get("type") == "image":
                        found.append(inner)
    return found


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
        assert _tool_result_images(client.build(_tool_result_messages())) == []

    def test_the_tool_result_keeps_its_text_and_its_identity(self) -> None:
        """Lowering the picture must not lose what the tool SAID, nor the
        id that pairs the result with its call — an orphaned tool_result is
        a 400 on every backend."""
        caps = ClientCapabilities(supports_vision=True, supports_vision_tool_results=False)
        request = _Client(caps, []).build(_tool_result_messages())
        block = request.messages[0]["content"][0]
        assert block["tool_use_id"] == "t1"
        assert any(b.get("text") == "captured" for b in block["content"])
        assert any("cannot see" in (b.get("text") or "") for b in block["content"])

    def test_a_blind_model_loses_the_tool_image_too(self) -> None:
        """supports_vision=False is the stronger statement: it covers both
        places an image can sit, not only the one the user attached."""
        request = _Client(ClientCapabilities(supports_vision=False), []).build(
            _tool_result_messages()
        )
        assert _tool_result_images(request) == []

    def test_the_conversations_tool_image_survives_for_the_next_hop(self) -> None:
        history = _tool_result_messages()
        caps = ClientCapabilities(supports_vision=True, supports_vision_tool_results=False)
        _Client(caps, []).build(history)
        assert history[0]["content"][0]["content"][1]["type"] == "image", (
            "a hop that cannot see tool images destroyed it in the conversation"
        )


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


class TestTheDegradationEndsWithThisRequest:
    """The bug this guards: ``request.messages`` is a SHALLOW copy of the
    caller's list, so replacing a block in place edits the pipeline's
    canonical history. One turn on a text-only hop would then destroy the
    image for every LATER turn — including the ones on a model that could
    have seen it. Exactly backwards for a conversation meant to outlive the
    model answering it."""

    def test_the_callers_history_still_holds_the_image(self) -> None:
        history = [{"role": "user", "content": [IMAGE, {"type": "text", "text": "hi"}]}]
        client = _Client(ClientCapabilities(supports_vision=False), [])
        client.build(history)
        assert history[0]["content"][0]["type"] == "image", (
            "a text-only hop destroyed the image in the conversation"
        )

    def test_the_request_still_carries_the_placeholder(self) -> None:
        history = [{"role": "user", "content": [IMAGE, {"type": "text", "text": "hi"}]}]
        client = _Client(ClientCapabilities(supports_vision=False), [])
        request = client.build(history)
        assert request.messages[0]["content"][0]["type"] == "text"

    def test_a_vision_hop_on_the_next_turn_still_sees_it(self) -> None:
        """The whole point: degrading for one hop must not degrade the
        conversation."""
        history = [{"role": "user", "content": [IMAGE, {"type": "text", "text": "hi"}]}]
        _Client(ClientCapabilities(supports_vision=False), []).build(history)
        seeing = _Client(
            ClientCapabilities(supports_vision=True, supports_vision_tool_results=True), []
        ).build(history)
        assert seeing.messages[0]["content"][0]["source"]["data"] == IMAGE["source"]["data"]

    def test_untouched_messages_are_not_copied(self) -> None:
        """Copy only what changes — a deep copy of every turn would be paid
        on every request by every provider."""
        plain = {"role": "user", "content": [{"type": "text", "text": "no image here"}]}
        history = [plain, {"role": "user", "content": [IMAGE]}]
        request = _Client(ClientCapabilities(supports_vision=False), []).build(history)
        assert request.messages[0] is plain
