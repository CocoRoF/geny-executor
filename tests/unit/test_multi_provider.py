"""Multi-provider tests — translation utilities, provider selection, pricing."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.stages.s06_api._translate import (
    normalize_stop_reason,
    canonical_tools_to_openai,
    canonical_tools_to_google,
    canonical_tool_choice_to_openai,
    canonical_tool_choice_to_google,
    canonical_messages_to_openai,
    canonical_messages_to_google,
    canonical_thinking_to_openai,
    canonical_thinking_to_google,
    blocks_to_text,
    split_tool_results,
    split_tool_uses,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stop Reason Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStopReason:
    def test_openai_stop_to_end_turn(self):
        assert normalize_stop_reason("stop", "openai") == "end_turn"

    def test_openai_tool_calls(self):
        assert normalize_stop_reason("tool_calls", "openai") == "tool_use"

    def test_openai_length(self):
        assert normalize_stop_reason("length", "openai") == "max_tokens"

    def test_google_stop(self):
        assert normalize_stop_reason("STOP", "google") == "end_turn"

    def test_google_max_tokens(self):
        assert normalize_stop_reason("MAX_TOKENS", "google") == "max_tokens"

    def test_google_safety(self):
        assert normalize_stop_reason("SAFETY", "google") == "content_filter"

    def test_anthropic_passthrough(self):
        assert normalize_stop_reason("end_turn", "anthropic") == "end_turn"

    def test_unknown_passthrough(self):
        assert normalize_stop_reason("custom_reason", "openai") == "custom_reason"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Definition Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolTranslation:
    CANONICAL_TOOLS = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    def test_to_openai(self):
        result = canonical_tools_to_openai(self.CANONICAL_TOOLS)
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["type"] == "object"
        assert "city" in tool["function"]["parameters"]["properties"]

    def test_to_google(self):
        result = canonical_tools_to_google(self.CANONICAL_TOOLS)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["parameters"]["type"] == "object"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Choice Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolChoice:
    def test_openai_auto(self):
        assert canonical_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_openai_any(self):
        assert canonical_tool_choice_to_openai({"type": "any"}) == "required"

    def test_openai_none(self):
        assert canonical_tool_choice_to_openai({"type": "none"}) == "none"

    def test_openai_specific(self):
        result = canonical_tool_choice_to_openai({"type": "tool", "name": "get_weather"})
        assert result["function"]["name"] == "get_weather"

    def test_openai_null(self):
        assert canonical_tool_choice_to_openai(None) == "auto"

    def test_google_auto(self):
        result = canonical_tool_choice_to_google({"type": "auto"})
        assert result["functionCallingConfig"]["mode"] == "AUTO"

    def test_google_any(self):
        result = canonical_tool_choice_to_google({"type": "any"})
        assert result["functionCallingConfig"]["mode"] == "ANY"

    def test_google_specific(self):
        result = canonical_tool_choice_to_google({"type": "tool", "name": "get_weather"})
        assert result["functionCallingConfig"]["mode"] == "ANY"
        assert "get_weather" in result["functionCallingConfig"]["allowedFunctionNames"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Message Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMessageTranslation:
    def test_openai_simple(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = canonical_messages_to_openai(messages, system="You are helpful.")
        assert result[0]["role"] == "developer"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_openai_tool_use_in_assistant(self):
        """assistant tool_use blocks → OpenAI tool_calls array."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Using tool"},
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/tmp"}},
                ],
            }
        ]
        result = canonical_messages_to_openai(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Using tool"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "t1"
        assert tc["function"]["name"] == "read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp"}

    def test_openai_tool_results(self):
        """user tool_result blocks → OpenAI 'tool' role messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
                ],
            }
        ]
        result = canonical_messages_to_openai(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "t1"
        assert msg["content"] == "file contents"

    def test_google_simple(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = canonical_messages_to_google(messages)
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Hello"}]
        assert result[1]["role"] == "model"
        assert result[1]["parts"] == [{"text": "Hi there"}]

    def test_google_tool_use(self):
        """assistant tool_use blocks → Google functionCall parts."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
                ],
            }
        ]
        result = canonical_messages_to_google(messages)
        assert result[0]["role"] == "model"
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["name"] == "search"
        assert fc["args"] == {"q": "test"}
        assert fc["id"] == "t1"

    def test_google_tool_result(self):
        """user tool_result blocks → Google functionResponse parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "name": "search",
                        "content": "results here",
                    },
                ],
            }
        ]
        result = canonical_messages_to_google(messages)
        fr = result[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "search"
        assert fr["id"] == "t1"
        assert fr["response"]["result"] == "results here"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thinking Translation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestThinkingTranslation:
    def test_openai_adaptive(self):
        assert canonical_thinking_to_openai({"type": "adaptive"}) == "medium"

    def test_openai_low_budget(self):
        assert canonical_thinking_to_openai({"type": "enabled", "budget_tokens": 2000}) == "low"

    def test_openai_high_budget(self):
        assert canonical_thinking_to_openai({"type": "enabled", "budget_tokens": 30000}) == "high"

    def test_openai_disabled(self):
        assert canonical_thinking_to_openai({"type": "disabled"}) is None

    def test_openai_none(self):
        assert canonical_thinking_to_openai(None) is None

    def test_google_adaptive(self):
        result = canonical_thinking_to_google({"type": "adaptive"})
        assert result["includeThoughts"] is True

    def test_google_high_budget(self):
        result = canonical_thinking_to_google({"type": "enabled", "budget_tokens": 25000})
        assert result["thinkingLevel"] == "high"
        assert result["includeThoughts"] is True

    def test_google_disabled(self):
        assert canonical_thinking_to_google({"type": "disabled"}) is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Content Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHelpers:
    def test_blocks_to_text_string(self):
        assert blocks_to_text("hello") == "hello"

    def test_blocks_to_text_blocks(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert blocks_to_text(blocks) == "hello\nworld"

    def test_blocks_to_text_skips_thinking(self):
        blocks = [{"type": "thinking"}, {"type": "text", "text": "answer"}]
        assert blocks_to_text(blocks) == "answer"

    def test_split_tool_results(self):
        content = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "hello"},
        ]
        results, other = split_tool_results(content)
        assert len(results) == 1
        assert len(other) == 1

    def test_split_tool_uses(self):
        content = [
            {"type": "text", "text": "Using tool"},
            {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
        ]
        text, tools = split_tool_uses(content)
        assert len(text) == 1
        assert len(tools) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Builder Auto-detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuilderInference:
    def test_openai_models_detected(self):
        from geny_executor.core.builder import PipelineBuilder

        for model in ["gpt-4.1", "gpt-4o-mini", "o3", "o4-mini"]:
            builder = PipelineBuilder("test", api_key="test", model=model)
            assert builder._infer_api_artifact() == "openai", f"Failed for {model}"

    def test_google_models_detected(self):
        from geny_executor.core.builder import PipelineBuilder

        for model in ["gemini-3-flash", "gemini-2.5-pro"]:
            builder = PipelineBuilder("test", api_key="test", model=model)
            assert builder._infer_api_artifact() == "google", f"Failed for {model}"

    def test_anthropic_models_default(self):
        from geny_executor.core.builder import PipelineBuilder

        for model in ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"]:
            builder = PipelineBuilder("test", api_key="test", model=model)
            assert builder._infer_api_artifact() is None, f"Failed for {model}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified Pricing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnifiedPricing:
    def test_anthropic_model(self):
        from geny_executor.stages.s07_token.artifact.default.pricing import (
            UnifiedPricingCalculator,
        )
        from geny_executor.core.state import TokenUsage

        calc = UnifiedPricingCalculator()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calc.calculate(usage, "claude-sonnet-4-6")
        assert cost == pytest.approx(3.0 + 15.0)  # $3 input + $15 output

    def test_openai_model(self):
        from geny_executor.stages.s07_token.artifact.default.pricing import (
            UnifiedPricingCalculator,
        )
        from geny_executor.core.state import TokenUsage

        calc = UnifiedPricingCalculator()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calc.calculate(usage, "gpt-4.1")
        assert cost == pytest.approx(2.0 + 8.0)  # $2 input + $8 output

    def test_google_model(self):
        from geny_executor.stages.s07_token.artifact.default.pricing import (
            UnifiedPricingCalculator,
        )
        from geny_executor.core.state import TokenUsage

        calc = UnifiedPricingCalculator()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calc.calculate(usage, "gemini-3-flash")
        assert cost == pytest.approx(0.50 + 3.0)  # $0.50 input + $3.0 output

    def test_unknown_model_zero(self):
        from geny_executor.stages.s07_token.artifact.default.pricing import (
            UnifiedPricingCalculator,
        )
        from geny_executor.core.state import TokenUsage

        calc = UnifiedPricingCalculator()
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert calc.calculate(usage, "unknown-model") == 0.0

    def test_anthropic_cache_pricing(self):
        from geny_executor.stages.s07_token.artifact.default.pricing import (
            UnifiedPricingCalculator,
        )
        from geny_executor.core.state import TokenUsage

        calc = UnifiedPricingCalculator()
        usage = TokenUsage(
            input_tokens=500_000,
            output_tokens=100_000,
            cache_creation_input_tokens=200_000,
            cache_read_input_tokens=300_000,
        )
        cost = calc.calculate(usage, "claude-sonnet-4-6")
        # regular_input = 500k - 300k (cache_read) = 200k → $0.6
        # output = 100k → $1.5
        # cache_write = 200k → $0.75
        # cache_read = 300k → $0.09
        expected = (
            (200_000 / 1e6 * 3.0)
            + (100_000 / 1e6 * 15.0)
            + (200_000 / 1e6 * 3.75)
            + (300_000 / 1e6 * 0.3)
        )
        assert cost == pytest.approx(expected)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache Bypass for non-Anthropic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCacheBypass:
    def test_anthropic_applies_cache(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s05_cache.artifact.default.strategies import (
            SystemCacheStrategy,
        )

        state = PipelineState(model="claude-sonnet-4-6", system="You are helpful.")
        strategy = SystemCacheStrategy()
        strategy.apply_cache_markers(state)
        # Should have converted to content blocks with cache_control
        assert isinstance(state.system, list)

    def test_openai_skips_cache(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s05_cache.artifact.default.strategies import (
            SystemCacheStrategy,
        )

        state = PipelineState(model="gpt-4.1", system="You are helpful.")
        strategy = SystemCacheStrategy()
        strategy.apply_cache_markers(state)
        # Should NOT modify — still a string
        assert state.system == "You are helpful."

    def test_google_skips_cache(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s05_cache.artifact.default.strategies import (
            AggressiveCacheStrategy,
        )

        state = PipelineState(model="gemini-3-flash", system="You are helpful.")
        strategy = AggressiveCacheStrategy()
        strategy.apply_cache_markers(state)
        assert state.system == "You are helpful."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Artifact System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestArtifactSystem:
    def test_list_api_artifacts(self):
        from geny_executor.core.artifact import list_artifacts

        artifacts = list_artifacts("s06_api")
        assert "default" in artifacts
        assert "openai" in artifacts
        assert "google" in artifacts
