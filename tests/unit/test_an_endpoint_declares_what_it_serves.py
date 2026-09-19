"""One client class, endpoints with nothing in common behind it.

``custom`` is how Geny reaches OpenRouter *and* a laptop's llama.cpp; the
one ``vllm`` class serves a tool-calling Qwen and a plain text model. The
class has to declare the weakest of them, so whoever configured the endpoint
gets to amend it — at construction, from the account row.

The regression that made this necessary: defaulting ``supports_vision`` to
False was right for a local server and silently turned every image sent
through OpenRouter into "[an image was attached here]".
"""

from __future__ import annotations

import pytest

from geny_executor.core.pipeline import _creds_to_client_kwargs
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.credentials import ProviderCredentials


class _Probe(BaseClient):
    provider = "probe"
    capabilities = ClientCapabilities(supports_tools=False, drops=("tools",))

    async def _send(self, request, *, purpose: str = ""):  # pragma: no cover
        raise NotImplementedError

    async def create_message_stream(self, **_kw):  # pragma: no cover
        raise NotImplementedError
        yield {}


class TestADeclarationFromTheHost:
    def test_a_flag_is_raised(self) -> None:
        client = _Probe()
        client.apply_capability_overrides({"supports_vision": True})
        assert client.capabilities.supports_vision is True

    def test_a_flag_is_lowered(self) -> None:
        client = _Probe()
        client.apply_capability_overrides({"supports_streaming": False})
        assert client.capabilities.supports_streaming is False

    def test_json_truthiness_becomes_a_real_boolean(self) -> None:
        """The declaration arrives from a database row via JSON, where a
        1 and a "true" are both how someone spells yes."""
        client = _Probe()
        client.apply_capability_overrides({"supports_vision": 1})
        assert client.capabilities.supports_vision is True

    def test_a_non_boolean_field_is_not_coerced(self) -> None:
        client = _Probe()
        client.apply_capability_overrides({"streaming_granularity": "message"})
        assert client.capabilities.streaming_granularity == "message"

    def test_an_unknown_flag_is_ignored_not_raised(self) -> None:
        """It comes from config. A typo in a settings page must not take the
        turn down, and a flag from a newer library version must not either."""
        client = _Probe()
        client.apply_capability_overrides({"supports_telepathy": True, "supports_vision": True})
        assert client.capabilities.supports_vision is True

    def test_nothing_declared_changes_nothing(self) -> None:
        client = _Probe()
        before = client.capabilities
        client.apply_capability_overrides(None)
        client.apply_capability_overrides({})
        assert client.capabilities is before

    def test_only_this_instance_is_affected(self) -> None:
        """Two accounts on the same class, one an aggregator and one a local
        box — the second must not inherit the first's declaration."""
        one, two = _Probe(), _Probe()
        one.apply_capability_overrides({"supports_vision": True})
        assert two.capabilities.supports_vision is False
        assert _Probe.capabilities.supports_vision is False


class TestTheDeclarationBeatsTheDropsTuple:
    """``drops`` is written against the class's conservative defaults. An
    upgrade that did not also amend the drops would restore the flag and
    still strip the field — the flag would read as a lie."""

    def test_restoring_tools_really_sends_them(self) -> None:
        from geny_executor.core.config import ModelConfig

        client = _Probe()
        client.apply_capability_overrides({"supports_tools": True})
        request = client._build_request(
            model_config=ModelConfig(model="m"),
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[{"name": "Read", "input_schema": {}}],
            tool_choice=None,
            stream=False,
        )
        assert request.tools


class TestADeclarationHasToDoSomething:
    """The bug this guards is the whole reason the declaration exists: a flag
    that changes ``capabilities`` and nothing on the wire is a settings page
    that lies. Stripping used to be the ``drops`` tuple's job alone, and every
    shipped client that says no to tools also lists them there — so the hole
    was invisible until a DEPLOYMENT said no through ``capabilities=`` on a
    class whose tuple says nothing about tools."""

    @staticmethod
    def _request(client, *, tools=True, tool_choice=True):
        from geny_executor.core.config import ModelConfig

        return client._build_request(
            model_config=ModelConfig(model="m"),
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[{"name": "Read", "input_schema": {}}] if tools else None,
            tool_choice={"type": "auto"} if tool_choice else None,
            stream=False,
        )

    def test_saying_no_to_tools_stops_sending_them(self) -> None:
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        client = CustomOpenAIClient(
            base_url="http://box:8080/v1", capabilities={"supports_tools": False}
        )
        assert self._request(client).tools is None

    def test_the_tool_choice_goes_with_them(self) -> None:
        """A tool_choice with no tools is a 400 on every backend that checks."""
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        client = CustomOpenAIClient(
            base_url="http://box:8080/v1", capabilities={"supports_tools": False}
        )
        assert self._request(client).tool_choice is None

    def test_saying_no_to_tool_choice_alone_keeps_the_tools(self) -> None:
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        client = CustomOpenAIClient(
            base_url="http://box:8080/v1", capabilities={"supports_tool_choice": False}
        )
        request = self._request(client)
        assert request.tools and request.tool_choice is None

    def test_the_host_is_told(self) -> None:
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        sink: list = []
        client = CustomOpenAIClient(
            base_url="http://box:8080/v1",
            capabilities={"supports_tools": False},
            event_sink=sink.append,
        )
        self._request(client)
        assert any("tools" in str(event) for event in sink), sink

    def test_an_endpoint_that_says_nothing_still_gets_its_tools(self) -> None:
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        client = CustomOpenAIClient(base_url="http://box:8080/v1")
        assert self._request(client).tools


class TestVLLM:
    """A vLLM server is whatever model it loaded — which is why the class
    says no tools and why an account has to be able to say otherwise."""

    def test_the_class_still_assumes_nothing(self) -> None:
        from geny_executor.llm_client.vllm import VLLMClient

        assert VLLMClient.capabilities.supports_tools is False

    def test_a_deployment_can_say_it_calls_tools(self) -> None:
        from geny_executor.llm_client.vllm import VLLMClient

        client = VLLMClient(
            base_url="http://box:8000/v1",
            capabilities={"supports_tools": True, "supports_tool_choice": True},
        )
        assert client.capabilities.supports_tools is True
        assert client.capabilities.supports_tool_choice is True

    def test_the_account_declaration_reaches_the_constructor(self) -> None:
        """The gap this closes: a flag stored on the account that no code
        path carried to the client is a setting that does nothing."""
        kwargs = _creds_to_client_kwargs(
            "vllm",
            ProviderCredentials(
                api_key="k",
                base_url="http://box:8000/v1",
                extras={"capabilities": {"supports_tools": True}},
            ),
        )
        assert kwargs["capabilities"] == {"supports_tools": True}

    def test_an_account_that_declares_nothing_sends_nothing(self) -> None:
        kwargs = _creds_to_client_kwargs(
            "vllm", ProviderCredentials(api_key="k", base_url="http://box:8000/v1")
        )
        assert "capabilities" not in kwargs


class TestTheOpenAICompatibleFamily:
    @pytest.mark.parametrize("provider", ["custom", "ollama", "lmstudio"])
    def test_the_account_declaration_reaches_the_constructor(self, provider: str) -> None:
        kwargs = _creds_to_client_kwargs(
            provider,
            ProviderCredentials(
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                extras={"capabilities": {"supports_vision": True}},
            ),
        )
        assert kwargs["capabilities"] == {"supports_vision": True}

    def test_an_aggregator_gets_its_vision_back(self) -> None:
        """OpenRouter is reached as ``custom``. Without the declaration the
        shared class assumes a text-only local server."""
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        default = CustomOpenAIClient(base_url="https://openrouter.ai/api/v1")
        declared = CustomOpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            capabilities={"supports_vision": True},
        )
        assert default.capabilities.supports_vision is False
        assert declared.capabilities.supports_vision is True

    def test_declaring_one_flag_leaves_the_rest_alone(self) -> None:
        from geny_executor.llm_client.openai_compatible import CustomOpenAIClient

        client = CustomOpenAIClient(
            base_url="http://box:8080/v1", capabilities={"supports_vision": True}
        )
        assert client.capabilities.supports_tools is True
        assert client.capabilities.supports_structured_output is True


class TestEveryCapabilityAwareClientTakesTheKwarg:
    """Geny names these providers as the ones it may declare for. A client
    that lost the parameter would fail only on a live turn."""

    @pytest.mark.parametrize("provider", ["custom", "local", "ollama", "lmstudio", "vllm"])
    def test_the_constructor_accepts_it(self, provider: str) -> None:
        import inspect

        from geny_executor.llm_client.registry import ClientRegistry

        params = inspect.signature(ClientRegistry.get(provider).__init__).parameters
        assert "capabilities" in params
