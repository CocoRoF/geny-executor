"""GatewayRunner — the inbound daemon.

For each registered :class:`PlatformAdapter` the runner polls for inbound
messages, hands each to the host's ``handler`` (which runs an agent turn and
returns reply text), and sends the reply back through the adapter. The runner
owns the loop, allow-list gating, error backoff, and concurrent per-message
dispatch; the host owns only "message in → reply text out".

Lifecycle mirrors :class:`geny_executor.cron.runner.CronRunner` — ``start()``
spawns one asyncio task per adapter, ``shutdown()`` stops them. Run it from the
host's app lifespan so the agent-running deps it touches are already wired.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Sequence, Union

from geny_executor.gateway.adapter import PlatformAdapter
from geny_executor.gateway.types import GatewayReply, InboundMessage

logger = logging.getLogger(__name__)

#: ``async def handler(msg) -> str | GatewayReply | None``. Return the reply
#: text (or a :class:`GatewayReply`), or ``None`` for no reply.
GatewayHandler = Callable[[InboundMessage], Awaitable[Optional[Union[str, GatewayReply]]]]


class GatewayRunner:
    """Drives inbound polling for a set of platform adapters."""

    def __init__(
        self,
        adapters: Sequence[PlatformAdapter],
        handler: GatewayHandler,
        *,
        error_backoff_seconds: float = 5.0,
        max_concurrent_turns: int = 8,
    ) -> None:
        self._adapters: List[PlatformAdapter] = list(adapters)
        self._handler = handler
        self._error_backoff = max(0.5, error_backoff_seconds)
        self._stop = asyncio.Event()
        self._poll_tasks: List[asyncio.Task] = []
        self._inflight: set[asyncio.Task] = set()
        # Bound concurrent agent turns so a burst of messages can't spawn
        # unbounded sessions.
        self._sem = asyncio.Semaphore(max(1, max_concurrent_turns))

    @property
    def adapters(self) -> List[PlatformAdapter]:
        return list(self._adapters)

    @property
    def running(self) -> bool:
        return bool(self._poll_tasks) and not self._stop.is_set()

    async def start(self) -> None:
        """Spawn one poll loop per adapter. Idempotent."""
        if self._poll_tasks:
            return
        if not self._adapters:
            logger.info("gateway_start_no_adapters")
            return
        self._stop.clear()
        for adapter in self._adapters:
            task = asyncio.create_task(self._poll_loop(adapter), name=f"gateway-{adapter.name}")
            self._poll_tasks.append(task)
        logger.info("gateway_started adapters=%s", [a.name for a in self._adapters])

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop all loops + in-flight turns and close adapters."""
        self._stop.set()
        for task in self._poll_tasks:
            task.cancel()
        for task in list(self._inflight):
            task.cancel()
        pending = [*self._poll_tasks, *self._inflight]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("gateway_shutdown_timeout")
        for adapter in self._adapters:
            try:
                await adapter.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("gateway_adapter_close_failed name=%s", adapter.name)
        self._poll_tasks = []
        self._inflight.clear()

    # ── internals ──────────────────────────────────────────────────────
    async def _poll_loop(self, adapter: PlatformAdapter) -> None:
        while not self._stop.is_set():
            try:
                messages = await adapter.fetch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.warning("gateway_fetch_error platform=%s err=%s", adapter.name, exc)
                await self._interruptible_sleep(self._error_backoff)
                continue
            for message in messages:
                if self._stop.is_set():
                    break
                if not adapter.allow(message):
                    logger.debug(
                        "gateway_message_not_allowed platform=%s chat=%s",
                        adapter.name,
                        message.chat_id,
                    )
                    continue
                # Dispatch concurrently so a slow agent turn doesn't stall the
                # poll loop (and other chats).
                task = asyncio.create_task(self._dispatch(adapter, message))
                self._inflight.add(task)
                task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, adapter: PlatformAdapter, message: InboundMessage) -> None:
        async with self._sem:
            try:
                reply = await self._handler(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "gateway_handler_error platform=%s chat=%s err=%s",
                    adapter.name,
                    message.chat_id,
                    exc,
                )
                return
            if reply is None:
                return
            text = reply if isinstance(reply, str) else reply.text
            to = message.chat_id
            if isinstance(reply, GatewayReply) and reply.chat_id:
                to = reply.chat_id
            if not text:
                return
            try:
                await adapter.send(chat_id=to, text=text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "gateway_send_error platform=%s chat=%s err=%s",
                    adapter.name,
                    to,
                    exc,
                )

    async def _interruptible_sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


__all__ = ["GatewayRunner", "GatewayHandler"]
