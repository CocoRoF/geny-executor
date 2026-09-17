"""One conversation, several accounts — and what may not happen between them.

The router turns "which account answers this call" into a runtime decision
so a single manifest can serve a Claude subscription, a second Claude login
and a ChatGPT plan without the pipeline knowing. Three rules keep that from
becoming a silent-corruption machine:

 · Failover only BEFORE the first token. Once the model has streamed text to
   the user, retrying on another account would replay half an answer.
 · Failover only for failures another account can survive. A bad request is
   the request's fault and must reach the caller unchanged.
 · A failing account cools down process-wide, so sibling agents skip it
   instead of walking into the same wall.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.config import ModelConfig
from geny_executor.llm_client import _failover as failover
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.router import RouterClient
from geny_executor.llm_client.types import APIResponse, ContentBlock


class _Scripted(BaseClient):
    """A child that either streams a scripted reply or raises."""

    capabilities = ClientCapabilities(supports_tools=True, supports_streaming=True)

    def __init__(self, label: str, *, error: Optional[BaseException] = None,
                 text: str = "", error_after_text: bool = False) -> None:
        super().__init__()
        self.label = label
        self.error = error
        self.text = text or f"hello from {label}"
        self.error_after_text = error_after_text
        self.calls = 0
        self.models: List[str] = []

    async def create_message_stream(self, *, model_config: Any, messages: Any, system: Any = "",
                                    tools: Any = None, tool_choice: Any = None,
                                    purpose: str = "") -> AsyncIterator[Dict[str, Any]]:
        self.calls += 1
        self.models.append(getattr(model_config, "model", ""))
        if self.error is not None and not self.error_after_text:
            raise self.error
        yield {"type": "text_delta", "text": self.text}
        if self.error is not None:
            raise self.error
        yield {"type": "message_complete", "response": APIResponse(
            content=[ContentBlock(type="text", text=self.text, raw={})],
            stop_reason="end_turn", model=getattr(model_config, "model", ""),
        )}

    async def create_message(self, *, model_config: Any, messages: Any, system: Any = "",
                             tools: Any = None, tool_choice: Any = None, purpose: str = "",
                             response_format: Any = None) -> APIResponse:
        self.calls += 1
        self.models.append(getattr(model_config, "model", ""))
        if self.error is not None:
            raise self.error
        return APIResponse(content=[ContentBlock(type="text", text=self.text, raw={})],
                           stop_reason="end_turn", model=getattr(model_config, "model", ""))

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:  # pragma: no cover
        raise NotImplementedError


def _target(account: str, model: str = "m", provider: str = "anthropic") -> Dict[str, Any]:
    return {"accountId": account, "label": account, "kind": "claude_code",
            "engineProvider": provider, "model": model}


def _router(children: Dict[str, _Scripted], targets: List[Dict[str, Any]],
            notes: Optional[List[Dict[str, Any]]] = None) -> RouterClient:
    return RouterClient(
        targets=targets,
        notify=(notes.append if notes is not None else None),
        client_factory=lambda t: children[str(t["accountId"])],
    )


async def _drain(router: RouterClient) -> str:
    out = []
    async for chunk in router.create_message_stream(
        model_config=ModelConfig(model="ignored"), messages=[{"role": "user", "content": "hi"}],
    ):
        if chunk.get("type") == "text_delta":
            out.append(chunk["text"])
    return "".join(out)


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    for account in list(failover.snapshot()):
        failover.clear_cooldown(account)
    yield
    for account in list(failover.snapshot()):
        failover.clear_cooldown(account)


class TestRoute:
    def test_a_route_with_no_usable_hop_is_refused_at_construction(self) -> None:
        with pytest.raises(APIError) as caught:
            RouterClient(targets=[{"accountId": "a"}])   # no provider / model
        assert caught.value.category == ErrorCategory.AUTH

    @pytest.mark.asyncio
    async def test_each_hop_runs_with_its_own_model(self) -> None:
        children = {"a": _Scripted("a")}
        router = _router(children, [_target("a", model="claude-sonnet-4-5")])
        await _drain(router)
        assert children["a"].models == ["claude-sonnet-4-5"]


class TestFailover:
    @pytest.mark.asyncio
    async def test_a_rate_limited_account_hands_over_mid_turn(self) -> None:
        children = {
            "a": _Scripted("a", error=APIError("usage limit reached",
                                               category=ErrorCategory.RATE_LIMITED)),
            "b": _Scripted("b", text="second account speaking"),
        }
        notes: List[Dict[str, Any]] = []
        router = _router(children, [_target("a"), _target("b")], notes)
        assert await _drain(router) == "second account speaking"
        assert [n["kind"] for n in notes] == ["failover", "route"]
        assert router.last_target["accountId"] == "b"

    @pytest.mark.asyncio
    async def test_a_bad_request_is_the_request_s_fault(self) -> None:
        children = {
            "a": _Scripted("a", error=APIError("schema invalid",
                                               category=ErrorCategory.BAD_REQUEST)),
            "b": _Scripted("b"),
        }
        router = _router(children, [_target("a"), _target("b")])
        with pytest.raises(APIError) as caught:
            await _drain(router)
        assert caught.value.category == ErrorCategory.BAD_REQUEST
        assert children["b"].calls == 0

    @pytest.mark.asyncio
    async def test_no_failover_once_the_user_has_seen_text(self) -> None:
        children = {
            "a": _Scripted("a", text="half an ans", error_after_text=True,
                           error=APIError("dropped", category=ErrorCategory.NETWORK)),
            "b": _Scripted("b"),
        }
        router = _router(children, [_target("a"), _target("b")])
        with pytest.raises(APIError):
            await _drain(router)
        assert children["b"].calls == 0

    @pytest.mark.asyncio
    async def test_an_exhausted_route_is_terminal_not_retryable(self) -> None:
        """Every hop out of quota: Stage 6's backoff would otherwise replay
        the whole route three more times for nothing."""
        limit = APIError("usage limit", category=ErrorCategory.RATE_LIMITED)
        children = {"a": _Scripted("a", error=limit), "b": _Scripted("b", error=limit)}
        router = _router(children, [_target("a"), _target("b")])
        with pytest.raises(APIError) as caught:
            await _drain(router)
        assert caught.value.category == ErrorCategory.TERMINAL


class TestCooldown:
    @pytest.mark.asyncio
    async def test_a_failed_account_is_cooled_for_every_agent(self) -> None:
        children = {
            "a": _Scripted("a", error=APIError("usage limit",
                                               category=ErrorCategory.RATE_LIMITED)),
            "b": _Scripted("b"),
        }
        await _drain(_router(children, [_target("a"), _target("b")]))
        assert "a" in failover.snapshot()

        # a second agent on the same route skips the cooling account first
        others = {"a": _Scripted("a", text="from a"), "b": _Scripted("b", text="from b")}
        assert await _drain(_router(others, [_target("a"), _target("b")])) == "from b"
        assert others["a"].calls == 0

    @pytest.mark.asyncio
    async def test_a_success_clears_the_cooldown(self) -> None:
        failover.cool_down("a", 600, "earlier limit")
        children = {"a": _Scripted("a")}
        await _drain(_router(children, [_target("a")]))
        assert "a" not in failover.snapshot()

    @pytest.mark.asyncio
    async def test_every_hop_cooling_still_tries_rather_than_refusing(self) -> None:
        failover.cool_down("a", 600, "limit")
        failover.cool_down("b", 600, "limit")
        children = {"a": _Scripted("a", text="from a"), "b": _Scripted("b")}
        assert await _drain(_router(children, [_target("a"), _target("b")])) == "from a"


class TestCacheMarkers:
    @pytest.mark.asyncio
    async def test_anthropic_cache_markers_never_reach_another_wire(self) -> None:
        seen: Dict[str, Any] = {}

        class _Recorder(_Scripted):
            async def create_message_stream(self, *, model_config, messages, system="",
                                            tools=None, tool_choice=None, purpose=""):
                seen["messages"] = messages
                seen["system"] = system
                async for chunk in super().create_message_stream(
                    model_config=model_config, messages=messages, system=system,
                    tools=tools, tool_choice=tool_choice, purpose=purpose):
                    yield chunk

        children = {"a": _Recorder("a")}
        router = _router(children, [_target("a", provider="geny_codex")])
        async for _ in router.create_message_stream(
            model_config=ModelConfig(model="x"),
            system=[{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        ):
            pass
        assert "cache_control" not in str(seen["messages"])
        assert "cache_control" not in str(seen["system"])

    @pytest.mark.asyncio
    async def test_an_anthropic_primary_keeps_reporting_as_anthropic(self) -> None:
        """Stage 5 places its cache markers only when the client says it IS
        anthropic — the router has to forward that truth or prompt caching
        silently stops for every API-backed account."""
        assert _router({"a": _Scripted("a")}, [_target("a")]).provider == "anthropic"
        assert _router({"a": _Scripted("a")},
                       [_target("a", provider="geny_codex")]).provider == "geny_router"
