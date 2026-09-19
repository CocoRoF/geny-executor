"""One pipeline, many accounts.

The manifest names a single provider — `geny_router` — for every agent, and
this client decides per call which real backend answers: the agent's chosen
account and model first, then its fallbacks in order. History stays in the
pipeline's canonical (Anthropic-shaped) form, so switching the model mid
conversation, or failing over from a rate-limited Claude login to a second
Claude login or to Codex, continues the SAME conversation with the SAME
tools, memory, hooks and permission policy.

Failover is deliberately narrow: only before the first token of a call and
only for failures another account can plausibly fix (rate limit, auth,
outage, missing CLI). A bad request is the request's fault and goes to the
user. An account that failed is cooled down process-wide so sibling agents
skip it too.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client import _failover as failover
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.credentials import ProviderCredentials
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.types import APIRequest, APIResponse

_CONTENT = {"text_delta", "thinking_delta", "tool_use", "input_json_delta"}


def _label(target: dict[str, Any]) -> str:
    return f"{target.get('label') or target.get('kind')}·{target.get('model')}"


class RouterClient(BaseClient):
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_system_prompt=True,
        supports_token_usage=True,
        supports_cost_usage=True,
        streaming_granularity="token",
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        *,
        targets: Optional[List[dict[str, Any]]] = None,
        notify: Optional[failover.Notify] = None,
        session_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
        event_sink: Any = None,
        client_factory: Any = None,
        balance: bool = False,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._targets = [t for t in (targets or []) if t.get("engineProvider") and t.get("model")]
        if not self._targets:
            raise APIError(
                "No model account is routed to this agent — add one and select it.",
                category=ErrorCategory.AUTH,
            )
        self._notify = notify
        self._session_id = session_id
        self._timeout_s = timeout_s
        self._children: dict[int, BaseClient] = {}
        self._factory = client_factory
        # Spread turns across healthy hops instead of always asking the
        # first one. Off by default: a route's order is a choice the user
        # made, and silently round-robining it would be a different
        # product. Geny turns it on for a route of interchangeable
        # subscription accounts.
        self._balance = bool(balance)
        self.last_target: Optional[dict[str, Any]] = None

    @property
    def targets(self) -> List[dict[str, Any]]:
        """The route as it stands, in order."""
        return list(self._targets)

    def set_route(self, targets: List[dict[str, Any]]) -> None:
        """Point this client at a different route, between turns.

        This is what makes switching model mid-conversation cheap: the
        pipeline is not rebuilt, so the conversation keeps its history, tools,
        memory, hooks and permission policy, and the very next turn simply
        goes somewhere else. Children built for the old route are closed on
        the next ``aclose``; their handles are dropped here so a stale client
        cannot answer for a hop that is no longer in the route.

        An empty (or entirely unusable) route is refused rather than applied:
        leaving the session with no way to reach a model is worse than
        keeping the route it already had.
        """
        usable = [t for t in (targets or []) if t.get("engineProvider") and t.get("model")]
        if not usable:
            raise APIError(
                "That route has no usable account — the current one is kept.",
                category=ErrorCategory.BAD_REQUEST,
            )
        self._targets = usable
        self._children = {}
        self.last_target = None

    # the prompt-cache stage only places Anthropic markers when the client
    # says it IS anthropic; the router forwards that truth for the primary
    @property
    def provider(self) -> str:  # type: ignore[override]
        primary = self._targets[0]
        return "anthropic" if primary.get("engineProvider") == "anthropic" else "geny_router"

    def _notify_host(self, payload: dict[str, Any]) -> None:
        if self._notify is not None:
            try:
                self._notify(payload)
            except Exception:
                pass

    def _child(self, index: int) -> BaseClient:
        client = self._children.get(index)
        if client is not None:
            return client
        target = self._targets[index]
        if self._factory is not None:
            client = self._factory(target)
        else:
            client = build_client(
                target, notify=self._notify, session_id=self._session_id, timeout_s=self._timeout_s
            )
        self._children[index] = client
        return client

    def _order(self) -> list[int]:
        """Who to ask, in order: healthy hops first, cooling ones last.

        With ``balance`` on, hops that are INTERCHANGEABLE — same provider
        and same model, i.e. two logins to the same thing — take turns
        least-recently-used instead of the first one answering until it
        hits its cap. That is the difference between failover and load
        balancing: failover only reacts once an account is already
        exhausted, which for subscription plans is backwards, since the
        reason to hold two logins is that neither should reach its limit.

        Balancing deliberately does NOT reach across different providers
        or models. A route is a preference — a primary and its fallbacks —
        so spreading turns over it would make one conversation answer as
        Claude, then as GPT, then as Claude again. Equivalent hops are a
        pool; the route between pools is still an order.
        """
        ready: list[int] = []
        cooling: list[int] = []
        for i, target in enumerate(self._targets):
            (cooling if failover.cooling(str(target.get("accountId") or "")) else ready).append(i)
        if self._balance:

            def group(i: int) -> tuple[str, str]:
                t = self._targets[i]
                return (str(t.get("engineProvider") or ""), str(t.get("model") or ""))

            # the pool's place in the route is its best member's place, so
            # a primary pool stays ahead of a fallback pool
            rank: dict[tuple[str, str], int] = {}
            for i in ready:
                rank.setdefault(group(i), i)
            ready.sort(
                key=lambda i: (
                    rank[group(i)],
                    failover.last_used(str(self._targets[i].get("accountId") or "")),
                    i,
                )
            )
        # everything cooling: still try, earliest recovery first — refusing
        # outright would turn a stale cooldown into an outage
        cooling.sort(
            key=lambda i: (
                failover.cooling(str(self._targets[i].get("accountId") or "")) or (0, "")
            )[0]
        )
        return ready + cooling

    def _prepare(
        self, index: int, model_config: Any, messages: Any, system: Any, tools: Any
    ) -> tuple:
        target = self._targets[index]
        cfg = (
            replace(model_config, model=str(target["model"]))
            if model_config is not None
            else model_config
        )
        if target.get("engineProvider") != "anthropic":
            messages = failover.strip_cache_markers(messages)
            system = failover.strip_cache_markers(system)
            tools = failover.strip_cache_markers(tools)
        if cfg is not None and target.get("maxTokens"):
            cfg = replace(cfg, max_tokens=int(target["maxTokens"]))
        if cfg is not None and target.get("thinking") is False:
            cfg = replace(cfg, thinking_enabled=False)
        return cfg, messages, system, tools

    def _failed(self, index: int, exc: BaseException, remaining: int) -> None:
        target = self._targets[index]
        category = failover.category_of(exc)
        account = str(target.get("accountId") or "")
        failover.cool_down(account, failover.cooldown_for(category), str(exc)[:200])
        self._notify_host(
            {
                "kind": "failover",
                "accountId": account,
                "from": _label(target),
                "category": category.value,
                "error": str(exc)[:400],
                "remaining": remaining,
            }
        )

    def _chosen(self, index: int) -> None:
        target = self._targets[index]
        self.last_target = target
        account = str(target.get("accountId") or "")
        failover.clear_cooldown(account)
        # Only the hop that actually answered counts as used — a hop that
        # was tried and failed over is not "its turn taken".
        failover.mark_used(account)
        self._notify_host(
            {
                "kind": "route",
                "accountId": target.get("accountId"),
                "label": target.get("label"),
                "accountKind": target.get("kind"),
                "model": target.get("model"),
                "index": index,
            }
        )

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        order = self._order()
        last_exc: Optional[BaseException] = None
        for position, index in enumerate(order):
            cfg, msgs, sys_, tls = self._prepare(index, model_config, messages, system, tools)
            started = False
            try:
                child = self._child(index)
                stream = child.create_message_stream(
                    model_config=cfg,
                    messages=msgs,
                    system=sys_,
                    tools=tls,
                    tool_choice=tool_choice,
                    purpose=purpose,
                )
                async for chunk in stream:
                    if not started and chunk.get("type") in _CONTENT | {"message_complete"}:
                        started = True
                        self._chosen(index)
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001 — classified below
                last_exc = exc
                category = failover.category_of(exc)
                remaining = len(order) - position - 1
                if started or category not in failover.FAILOVER or remaining == 0:
                    if not started and category in failover.FAILOVER:
                        self._failed(index, exc, 0)
                    if not started and remaining == 0 and category in failover.EXHAUSTED_TERMINAL:
                        # every hop is out of quota / logged out: Stage 6's
                        # backoff would only replay the whole route 3 more times
                        raise APIError(
                            str(exc), category=ErrorCategory.TERMINAL, cause=exc
                        ) from exc
                    raise
                self._failed(index, exc, remaining)
        if last_exc is not None:
            raise last_exc

    async def create_message(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        order = self._order()
        for position, index in enumerate(order):
            cfg, msgs, sys_, tls = self._prepare(index, model_config, messages, system, tools)
            try:
                response = await self._child(index).create_message(
                    model_config=cfg,
                    messages=msgs,
                    system=sys_,
                    tools=tls,
                    tool_choice=tool_choice,
                    purpose=purpose,
                    response_format=response_format,
                )
                self._chosen(index)
                return response
            except Exception as exc:  # noqa: BLE001
                category = failover.category_of(exc)
                remaining = len(order) - position - 1
                if category not in failover.FAILOVER or remaining == 0:
                    if remaining == 0 and category in failover.EXHAUSTED_TERMINAL:
                        raise APIError(
                            str(exc), category=ErrorCategory.TERMINAL, cause=exc
                        ) from exc
                    raise
                self._failed(index, exc, remaining)
        raise APIError("no route", category=ErrorCategory.UNKNOWN)

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        raise NotImplementedError("RouterClient routes through create_message")

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        try:
            return await self._child(self._order()[0]).warmup(timeout_s=timeout_s)
        except Exception:
            return False

    async def aclose(self) -> None:
        for client in self._children.values():
            close = getattr(client, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
        self._children.clear()


# ── child construction ────────────────────────────────────────────────
def build_client(
    target: dict[str, Any],
    *,
    notify: Optional[failover.Notify],
    session_id: Optional[str],
    timeout_s: Optional[float],
) -> BaseClient:
    provider = str(target["engineProvider"])
    options = dict(target.get("options") or {})
    if provider in ("geny_claude_code", "geny_codex"):
        from geny_executor.llm_client.claude_code_tokens import ClaudeCodeTokenClient
        from geny_executor.llm_client.codex import CodexResponsesClient

        cls = ClaudeCodeTokenClient if provider == "geny_claude_code" else CodexResponsesClient
        kwargs: dict[str, Any] = {
            **options,
            "api_key": target.get("apiKey") or "",
            "base_url": target.get("baseUrl") or None,
            "account_id": target.get("accountId") or "",
            "account_label": target.get("label") or "",
            "notify": notify,
        }
        if provider == "geny_codex":
            kwargs["session_id"] = session_id
        if timeout_s and "timeout_s" not in kwargs:
            kwargs["timeout_s"] = timeout_s
        return cls(**kwargs)

    from geny_executor.core.pipeline import _creds_to_client_kwargs

    creds = ProviderCredentials(
        api_key=str(target.get("apiKey") or ""),
        base_url=target.get("baseUrl") or None,
        default_headers=target.get("headers") or None,
        extras=options,
    )
    client_cls = ClientRegistry.get(provider)
    return client_cls(**_creds_to_client_kwargs(provider, creds))
