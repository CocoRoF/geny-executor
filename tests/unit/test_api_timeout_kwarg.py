"""2.2.0 Wave 1 — s06 timeout_ms threaded into the client call kwargs.

Audit "validated-but-inert" table: ``timeout_ms`` was accepted by the
stage schema, stored on the stage, serialized — and never reached the
client call. The wiring feeds ``timeout_ms`` to clients whose call
signature accepts it (named param or ``**kwargs``); clients that predate
the kwarg get an ``api.timeout_unsupported`` event instead of either a
silent drop (the old bug) or a TypeError (which would regress manifests
that set the previously-inert knob).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.llm_client import BaseClient
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock
from geny_executor.stages.s06_api.artifact.default.stage import APIStage


def _response() -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text="ok")],
        stop_reason="end_turn",
        model="m",
    )


class _TimeoutAwareClient(BaseClient):
    """Client whose high-level surface accepts timeout_ms (the post-wave
    client shape)."""

    provider = "timeout-aware"

    def __init__(self) -> None:
        super().__init__(api_key="k")
        self.seen_kwargs: List[Dict[str, Any]] = []

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return _response()

    async def create_message(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
        timeout_ms: Optional[int] = None,
    ) -> APIResponse:
        self.seen_kwargs.append(
            {
                "model_config": model_config,
                "timeout_ms": timeout_ms,
            }
        )
        return _response()


class _LegacyClient(BaseClient):
    """Client with the pre-wave signature — no timeout_ms, no **kwargs."""

    provider = "legacy"

    def __init__(self) -> None:
        super().__init__(api_key="k")
        self.calls = 0

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return _response()

    async def create_message(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> APIResponse:
        self.calls += 1
        return _response()


def _state(client: BaseClient) -> PipelineState:
    state = PipelineState(session_id="s")
    state.llm_client = client
    state.messages = [{"role": "user", "content": "hi"}]
    state.stream = False  # exercise the non-streaming call path
    return state


class TestTimeoutReachesCallKwargs:
    @pytest.mark.asyncio
    async def test_timeout_ms_passed_to_supporting_client(self):
        client = _TimeoutAwareClient()
        stage = APIStage(timeout_ms=12_345)

        await stage.execute("in", _state(client))

        assert client.seen_kwargs[0]["timeout_ms"] == 12_345

    @pytest.mark.asyncio
    async def test_no_timeout_configured_no_kwarg(self):
        client = _TimeoutAwareClient()
        stage = APIStage()

        await stage.execute("in", _state(client))

        assert client.seen_kwargs[0]["timeout_ms"] is None

    @pytest.mark.asyncio
    async def test_update_config_path(self):
        """The manifest path: timeout_ms arrives via update_config."""
        client = _TimeoutAwareClient()
        stage = APIStage()
        stage.update_config({"timeout_ms": 7_000})

        await stage.execute("in", _state(client))

        assert client.seen_kwargs[0]["timeout_ms"] == 7_000


class TestLegacyClientBackCompat:
    @pytest.mark.asyncio
    async def test_legacy_client_not_passed_timeout_and_event_emitted(self):
        client = _LegacyClient()
        stage = APIStage(timeout_ms=12_345)
        state = _state(client)

        await stage.execute("in", state)  # must not TypeError

        assert client.calls == 1
        unsupported = [e for e in state.events if e["type"] == "api.timeout_unsupported"]
        assert len(unsupported) == 1
        assert unsupported[0]["data"]["timeout_ms"] == 12_345

    @pytest.mark.asyncio
    async def test_legacy_client_without_timeout_no_event(self):
        client = _LegacyClient()
        stage = APIStage()
        state = _state(client)

        await stage.execute("in", state)

        assert not [e for e in state.events if e["type"] == "api.timeout_unsupported"]
