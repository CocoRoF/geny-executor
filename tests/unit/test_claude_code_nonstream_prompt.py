"""Non-streaming claude_code create_message must deliver the prompt (2.17.0).

Regression: non-stream `--print --output-format json` carried NO prompt (not in
stdin, not in argv) → CLI "input must be provided". Streaming worked. Now the
non-stream path appends the prompt as the trailing positional argument.
"""
from geny_executor.llm_client.base import APIRequest
from geny_executor.llm_client.translators._cli import (
    claude_code_argv, flatten_messages_to_prompt,
)


def _req(messages, stream):
    return APIRequest(model="claude-haiku-4-5-20251001", messages=messages, stream=stream)


def test_nonstream_appends_prompt_positional():
    argv = claude_code_argv(_req([{"role": "user", "content": "say ok"}], False))
    assert "--print" in argv
    assert argv[-1] == "say ok"            # trailing positional prompt
    assert "--input-format" not in argv    # non-stream = json output, no stdin


def test_streaming_does_not_add_positional_prompt():
    argv = claude_code_argv(_req([{"role": "user", "content": "say ok"}], True))
    assert "--input-format" in argv        # streaming → stdin delivers the prompt
    assert "say ok" not in argv            # not a positional arg in stream mode


def test_flatten_single_user():
    assert flatten_messages_to_prompt([{"role": "user", "content": "hello"}]) == "hello"


def test_flatten_multi_turn_structure():
    out = flatten_messages_to_prompt([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "now this"},
    ])
    assert "## Conversation so far" in out
    assert "### Assistant\nreply" in out
    assert out.rstrip().endswith("now this")  # latest user turn is the current input


def test_flatten_empty():
    assert flatten_messages_to_prompt([]) == ""
