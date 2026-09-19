"""One conversation, any provider, at any point — the invariant behind
mid-conversation model switching.

An agent session outlives the model answering it. A user swaps Opus for a
ChatGPT plan mid-thread; a route fails over from a rate-limited Claude
account to a second one; a sub-agent runs on a cheaper model than its
parent. In every case the SAME canonical history is handed to a DIFFERENT
translator, and it has to survive the trip.

Three properties make that work, and each has broken in some harness
somewhere:

  1. No vendor session handle. Nothing carries ``previous_response_id`` or
     a thread id, so history is rebuilt from our own record every call and
     no provider owns the conversation.
  2. No cross-model thinking. Reasoning blocks are bound to the model that
     produced them; replaying one to another model is at best ignored and
     at worst a 400. Every translator drops them.
  3. Our tool ids, round-tripped. A ``tool_use`` emitted on one provider
     and its ``tool_result`` executed here must still pair when the NEXT
     turn runs somewhere else.
"""

from __future__ import annotations

import json

import pytest

# A history with every shape that has to survive: text, an image, a tool
# call made by one provider, its result from our Stage 10, and a thinking
# block from whichever model produced it.
HISTORY = [
    {"role": "user", "content": "look at this and list /tmp"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should list it.", "signature": "sig-from-opus"},
            {"type": "text", "text": "Listing."},
            {
                "type": "tool_use",
                "id": "toolu_geny_abc123",
                "name": "ListDir",
                "input": {"path": "/tmp"},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_geny_abc123", "content": "a.txt"}
        ],
    },
    {"role": "user", "content": "thanks"},
]


def _dumped(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class TestNoVendorSessionHandle:
    def test_the_responses_wire_sends_no_previous_response_id(self) -> None:
        """Codex rebuilds the thread from OUR history every call. A vendor
        session handle would mean a conversation could not leave Codex and
        come back — and could not be replayed on a second account."""
        from geny_executor.llm_client.responses_wire import to_input

        blob = _dumped(to_input(HISTORY))
        assert "previous_response_id" not in blob
        assert "thread_id" not in blob


class TestNoCrossModelThinking:
    def test_the_responses_wire_drops_thinking(self) -> None:
        from geny_executor.llm_client.responses_wire import to_input

        blob = _dumped(to_input(HISTORY))
        assert "sig-from-opus" not in blob, "a signature from another model was replayed"
        assert "I should list it." not in blob

    def test_the_anthropic_translator_drops_thinking(self) -> None:
        from geny_executor.llm_client.translators import canonical_messages_to_anthropic

        blob = _dumped(canonical_messages_to_anthropic(HISTORY))
        assert "sig-from-opus" not in blob, "a signature from another model was replayed"

    def test_the_openai_translator_drops_thinking(self) -> None:
        from geny_executor.llm_client.translators import canonical_messages_to_openai

        blob = _dumped(canonical_messages_to_openai(HISTORY))
        assert "sig-from-opus" not in blob

    def test_the_google_translator_drops_thinking(self) -> None:
        from geny_executor.llm_client.translators import canonical_messages_to_google

        blob = _dumped(canonical_messages_to_google(HISTORY))
        assert "sig-from-opus" not in blob

    def test_the_claude_code_protocol_drops_thinking(self) -> None:
        from geny_executor.llm_client import text_tool_protocol as tp

        blob = _dumped(tp.render_transcript(HISTORY))
        assert "sig-from-opus" not in blob


class TestOurToolIdsRoundTrip:
    """The call and its result must still refer to each other after the
    conversation moves to a provider that never issued the id."""

    def test_the_responses_wire_pairs_them(self) -> None:
        from geny_executor.llm_client.responses_wire import to_input

        items = to_input(HISTORY)
        calls = [i for i in items if i.get("type") == "function_call"]
        outs = [i for i in items if i.get("type") == "function_call_output"]
        assert [c["call_id"] for c in calls] == ["toolu_geny_abc123"]
        assert [o["call_id"] for o in outs] == ["toolu_geny_abc123"]

    def test_the_anthropic_translator_pairs_them(self) -> None:
        from geny_executor.llm_client.translators import canonical_messages_to_anthropic

        blob = _dumped(canonical_messages_to_anthropic(HISTORY))
        assert blob.count("toolu_geny_abc123") >= 2, "the call and its result came apart"

    def test_the_openai_translator_pairs_them(self) -> None:
        from geny_executor.llm_client.translators import canonical_messages_to_openai

        blob = _dumped(canonical_messages_to_openai(HISTORY))
        assert blob.count("toolu_geny_abc123") >= 2


class TestEveryProviderAcceptsTheSameHistory:
    @pytest.mark.parametrize(
        "translate",
        [
            pytest.param("responses", id="geny_codex / openai-responses"),
            pytest.param("anthropic", id="anthropic / geny_claude_code family"),
            pytest.param("openai", id="openai / vllm / ollama / lmstudio"),
            pytest.param("google", id="google"),
            pytest.param("claude_code", id="geny_claude_code text protocol"),
        ],
    )
    def test_it_renders_without_raising_and_keeps_the_user_text(self, translate: str) -> None:
        """A translator that raises on a history another provider produced
        is a session that dies the moment its model changes."""
        if translate == "responses":
            from geny_executor.llm_client.responses_wire import to_input as fn
        elif translate == "anthropic":
            from geny_executor.llm_client.translators import canonical_messages_to_anthropic as fn
        elif translate == "openai":
            from geny_executor.llm_client.translators import canonical_messages_to_openai as fn
        elif translate == "google":
            from geny_executor.llm_client.translators import canonical_messages_to_google as fn
        else:
            from geny_executor.llm_client import text_tool_protocol as tp

            fn = tp.render_transcript

        blob = _dumped(fn(HISTORY))
        assert "look at this and list /tmp" in blob
        assert "a.txt" in blob
