"""The number that decides whether a long conversation survives.

``context_window_budget`` sizes proactive compaction and the Stage-4 headroom
guard. It ships at 200_000 — right for the frontier vendors, wrong by a factor
of six for a local server launched with 32k, where it means compaction never
fires and the request overflows before anything notices.

What is pinned here is that the number is *resolved* and never *guessed*: the
operator's declaration, then what the endpoint said, then a short table of
families whose window really is a property of the model — and then nothing,
because "we don't know" is something a host can tell a user and a confident
wrong number is not.
"""

from __future__ import annotations

import pytest

from geny_executor.llm_client.context_window import (
    DEFAULT_CONTEXT_WINDOW,
    binding_context_window,
    known_context_window,
    resolve_context_window,
)
from geny_executor.llm_client.model_discovery import _parse_openai_models


class TestPrecedence:
    def test_the_operator_wins(self) -> None:
        """Nobody else knows what sits behind a company gateway's address."""
        assert (
            resolve_context_window(declared=8192, discovered=131072, model="claude-sonnet-5")
            == 8192
        )

    def test_then_what_the_endpoint_said(self) -> None:
        assert resolve_context_window(discovered=32768, model="claude-sonnet-5") == 32768

    def test_then_what_we_know(self) -> None:
        assert resolve_context_window(model="claude-sonnet-5") == 200_000

    def test_and_otherwise_nothing(self) -> None:
        """Not a fallback guess. A caller that knows it is unknown can say so
        to the user; a number cannot."""
        assert resolve_context_window(model="some-model-nobody-shipped-yet") is None

    def test_a_zero_is_not_an_answer(self) -> None:
        assert resolve_context_window(declared=0, discovered=0, model="claude-sonnet-5") == 200_000


class TestWhatWeClaimToKnow:
    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"])
    def test_the_claude_family(self, model: str) -> None:
        assert known_context_window(model) == 200_000

    @pytest.mark.parametrize("alias", ["sonnet", "opus", "haiku", "fable"])
    def test_the_cli_aliases_resolve_to_the_family(self, alias: str) -> None:
        """A Claude Code account stores ``sonnet``, not a canonical id."""
        assert known_context_window(alias) == 200_000

    def test_an_aggregator_prefix_is_stripped(self) -> None:
        assert known_context_window("anthropic/claude-sonnet-5") == 200_000

    def test_the_longest_prefix_wins(self) -> None:
        assert known_context_window("gemini-2.5-pro") == 1_048_576

    def test_we_do_not_claim_to_know_the_open_families(self) -> None:
        """A llama's real window is whatever the server was started with —
        knowable only by asking it."""
        for model in ("llama3.1", "qwen3-coder", "mistral-large", "deepseek-v3"):
            assert known_context_window(model) is None, model

    def test_a_deployment_defined_provider_is_not_answered_from_the_table(self) -> None:
        """Even for an id the table recognises: on a self-hosted server the
        window belongs to the server, not to the weights."""
        assert known_context_window("claude-sonnet-5", provider="vllm") is None
        assert known_context_window("claude-sonnet-5", provider="ollama") is None

    def test_the_default_is_shared_with_the_pipeline(self) -> None:
        """Two copies of 200_000 that can drift is how a host ends up sizing
        compaction to a number the pipeline no longer uses."""
        from geny_executor.core.config import PipelineConfig

        assert PipelineConfig().context_window_budget == DEFAULT_CONTEXT_WINDOW


class TestARouteHoldsItsSmallestHop:
    """A route is one conversation that must be able to continue on any of its
    hops. Sizing to the primary means the failover — which happens exactly when
    things are already going wrong — walks into an overflow it cannot recover
    from."""

    def test_the_smallest_binds(self) -> None:
        assert binding_context_window([200_000, 32_768]) == 32_768

    def test_an_unknown_hop_cannot_raise_the_budget(self) -> None:
        assert binding_context_window([32_768, None]) == 32_768

    def test_all_unknown_is_unknown(self) -> None:
        assert binding_context_window([None, None]) is None

    def test_no_hops_is_unknown(self) -> None:
        assert binding_context_window([]) is None


class TestWhatTheEndpointStates:
    """The backends whose window we cannot know from a model id are exactly
    the ones that state it."""

    def test_an_aggregator_states_it(self) -> None:
        models = _parse_openai_models({"data": [{"id": "x", "context_length": 131072}]})
        assert models[0].context_window == 131072

    def test_a_vllm_server_states_what_it_was_launched_with(self) -> None:
        """``max_model_len`` is the running limit, usually below what the
        weights support — which is the number that actually binds."""
        models = _parse_openai_models({"data": [{"id": "q", "max_model_len": 32768}]})
        assert models[0].context_window == 32768

    def test_the_routed_provider_binds_before_the_model(self) -> None:
        """OpenRouter states both; the one that will actually serve the call
        is the one in ``top_provider``."""
        models = _parse_openai_models(
            {"data": [{"id": "m", "context_length": 200000,
                       "top_provider": {"context_length": 32768}}]}
        )
        assert models[0].context_window == 32768

    def test_a_backend_that_says_nothing_says_nothing(self) -> None:
        models = _parse_openai_models({"data": [{"id": "plain"}]})
        assert models[0].context_window is None

    def test_a_nonsense_value_is_not_believed(self) -> None:
        models = _parse_openai_models({"data": [{"id": "a", "context_length": "lots"},
                                                {"id": "b", "context_length": 0},
                                                {"id": "c", "context_length": -1}]})
        assert [m.context_window for m in models] == [None, None, None]

    def test_google_states_it_too(self) -> None:
        from geny_executor.llm_client.model_discovery import _parse_google_models

        models = _parse_google_models(
            {"models": [{"name": "models/gemini-2.5-pro", "inputTokenLimit": 1048576}]}
        )
        assert models[0].context_window == 1_048_576
