"""Driving the tool loop through a model that can only write text.

A token-only backend (the `claude` CLI with its tools switched off) cannot
emit native tool calls, so the protocol lives in the system prompt and the
model answers with `<tool_call>` blocks. The splitter has to do three things
at once, incrementally, while tokens arrive:

 · release ordinary text to the user as it streams,
 · withhold everything between the tags and parse it into a real call,
 · stop reading once the model starts narrating a result it has not seen.

That last one is the expensive failure: the protocol tells the model to end
its message after the last call, and a model that keeps going invents the
outcome. Cutting the stream there costs nothing and keeps the invention out
of the transcript.
"""

from __future__ import annotations

from geny_executor.llm_client import text_tool_protocol as tp


def _feed(chunks, known=None):
    splitter = tp.StreamSplitter(known=known)
    released = "".join(splitter.feed(c) for c in chunks)
    tail = splitter.close()
    return splitter, released + tail


class TestParsing:
    def test_a_plain_call(self) -> None:
        splitter, visible = _feed(['<tool_call>\n{"name":"Read","arguments":{"file_path":"/a"}}\n</tool_call>'])
        assert visible == ""
        assert [(c.name, c.arguments) for c in splitter.calls] == [("Read", {"file_path": "/a"})]

    def test_text_before_a_call_reaches_the_user(self) -> None:
        splitter, visible = _feed(['Reading it now.\n<tool_call>{"name":"Read","arguments":{}}</tool_call>'])
        assert visible.strip() == "Reading it now."
        assert len(splitter.calls) == 1

    def test_a_tag_split_across_chunks(self) -> None:
        splitter, visible = _feed(["Hi <tool", "_call>", '{"name":"Read",', '"arguments":{}}', "</tool_call>"])
        assert visible.strip() == "Hi"
        assert [c.name for c in splitter.calls] == ["Read"]

    def test_several_independent_calls(self) -> None:
        splitter, _ = _feed([
            '<tool_call>{"name":"Read","arguments":{"file_path":"/a"}}</tool_call>',
            '<tool_call>{"name":"Read","arguments":{"file_path":"/b"}}</tool_call>',
        ])
        assert [c.arguments["file_path"] for c in splitter.calls] == ["/a", "/b"]

    def test_call_ids_are_unique(self) -> None:
        splitter, _ = _feed([
            '<tool_call>{"name":"Read","arguments":{}}</tool_call>',
            '<tool_call>{"name":"Read","arguments":{}}</tool_call>',
        ])
        assert splitter.calls[0].id != splitter.calls[1].id

    def test_a_closing_tag_quoted_inside_an_argument(self) -> None:
        body = '{"name":"Write","arguments":{"content":"write </tool_call> here"}}'
        splitter, _ = _feed([f"<tool_call>{body}</tool_call>"])
        assert splitter.calls[0].arguments["content"] == "write </tool_call> here"

    def test_flat_arguments_next_to_the_name(self) -> None:
        splitter, _ = _feed(['<tool_call>{"name":"Write","file_path":"/a","content":"x"}</tool_call>'])
        assert splitter.calls[0].arguments == {"file_path": "/a", "content": "x"}

    def test_an_openai_shaped_call(self) -> None:
        splitter, _ = _feed([
            '<tool_call>{"function":{"name":"Read","arguments":"{\\"file_path\\":\\"/a\\"}"}}</tool_call>'
        ])
        assert splitter.calls[0].name == "Read"
        assert splitter.calls[0].arguments == {"file_path": "/a"}

    def test_a_fenced_body(self) -> None:
        splitter, _ = _feed(['<tool_call>\n```json\n{"name":"Read","arguments":{}}\n```\n</tool_call>'])
        assert splitter.calls[0].name == "Read"

    def test_a_case_near_miss_snaps_to_the_real_tool(self) -> None:
        splitter, _ = _feed(['<tool_call>{"name":"read","arguments":{}}</tool_call>'], known={"Read"})
        assert splitter.calls[0].name == "Read"

    def test_a_missing_closing_tag_at_end_of_stream(self) -> None:
        splitter, _ = _feed(['<tool_call>{"name":"Read","arguments":{}}'])
        assert [c.name for c in splitter.calls] == ["Read"]


class TestStoppingEarly:
    def test_narration_after_a_call_ends_the_stream(self) -> None:
        splitter = tp.StreamSplitter()
        splitter.feed('<tool_call>{"name":"Read","arguments":{}}</tool_call>')
        splitter.feed("The file contains a list of names.")
        assert splitter.finished is True
        assert "list of names" not in splitter.text

    def test_a_second_call_is_not_narration(self) -> None:
        splitter = tp.StreamSplitter()
        splitter.feed('<tool_call>{"name":"Read","arguments":{}}</tool_call>\n')
        splitter.feed('<tool_call>{"name":"Read","arguments":{}}</tool_call>')
        assert splitter.finished is False
        assert len(splitter.calls) == 2


class TestProseIsNotACall:
    def test_prose_mentioning_the_tag_is_given_back(self) -> None:
        splitter, visible = _feed(["To call a tool you write <tool_call> and then JSON."])
        assert splitter.calls == []
        assert "<tool_call>" in visible

    def test_a_malformed_body_after_a_real_call_is_counted_not_guessed(self) -> None:
        splitter = tp.StreamSplitter()
        splitter.feed('<tool_call>{"name":"Read","arguments":{}}</tool_call>')
        splitter.feed("<tool_call>not json at all")
        splitter.close()
        assert len(splitter.calls) == 1
        assert splitter.malformed == 1


class TestSystemPrompt:
    def test_the_catalogue_carries_every_tool(self) -> None:
        prompt = tp.render_system_prompt("Be helpful.", [
            {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}},
        ])
        assert "Be helpful." in prompt
        assert "<tools>" in prompt and '"Read"' in prompt

    def test_no_tools_means_no_protocol_noise(self) -> None:
        assert tp.render_system_prompt("Be helpful.", []) == "Be helpful."

    def test_block_shaped_system_prompts_are_flattened(self) -> None:
        text = tp.system_text([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
        assert text == "A\n\nB"


class TestTranscript:
    def test_a_first_user_message_is_sent_as_itself(self) -> None:
        blocks = tp.render_transcript([{"role": "user", "content": "hello"}])
        assert blocks == [{"type": "text", "text": "hello"}]

    def test_history_is_tagged_as_a_conversation(self) -> None:
        blocks = tp.render_transcript([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and now?"},
        ])
        text = blocks[-1]["text"]
        assert "<conversation>" in text and '<turn role="user">' in text
        assert text.endswith("and now?")

    def test_tool_results_come_back_as_the_harness_speaking(self) -> None:
        blocks = tp.render_transcript([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file body"}]},
        ])
        text = blocks[-1]["text"]
        assert "The harness executed your tool calls" in text
        assert "file body" in text

    def test_images_on_the_live_message_stay_real_images(self) -> None:
        blocks = tp.render_transcript([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                {"type": "text", "text": "what is this?"},
            ]},
        ])
        assert blocks[0]["type"] == "image"
        assert "what is this?" in blocks[-1]["text"]
