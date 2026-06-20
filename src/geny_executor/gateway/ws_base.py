"""Shared base for WebSocket gateway adapters (Discord / Slack).

A WS platform pushes messages asynchronously, which doesn't fit the runner's
``fetch()`` batch-poll directly — so this base runs the WS connection in a
background task that buffers inbound :class:`InboundMessage` into a queue, and
``fetch()`` simply drains that queue (blocking up to ``idle_timeout`` for the
first item). The connection auto-reconnects with backoff. Subclasses implement
``_run_connection`` (one full connection lifecycle) and ``send``.

WebSocket transport uses the ``websockets`` library (a declared dep). The
small REST calls (sending a reply, Slack's connection-open) go through
``_http_post`` over ``httpx`` with an injectable ``http_transport`` for tests.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from geny_executor.gateway.adapter import PlatformAdapter
from geny_executor.gateway.types import InboundMessage

logger = logging.getLogger(__name__)

#: ``async def http_transport(url, *, json, headers) -> dict`` — test hook for
#: the adapter's REST calls (reply / connection-open) without real HTTP.
HttpTransport = Callable[..., Awaitable[Dict[str, Any]]]


class _QueuedWSAdapter(PlatformAdapter):
    """Background WS connection → queue; ``fetch`` drains the queue."""

    def __init__(
        self,
        *,
        idle_timeout: float = 25.0,
        reconnect_backoff: float = 5.0,
        http_transport: Optional[HttpTransport] = None,
    ) -> None:
        self._idle_timeout = idle_timeout
        self._reconnect_backoff = max(0.5, reconnect_backoff)
        self._http_transport = http_transport
        self._queue: Optional[asyncio.Queue] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._closed = False

    def _ensure_queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    # ── runner-facing ──────────────────────────────────────────────────
    async def fetch(self) -> List[InboundMessage]:
        self._ensure_connected()
        queue = self._ensure_queue()
        try:
            first = await asyncio.wait_for(queue.get(), timeout=self._idle_timeout)
        except asyncio.TimeoutError:
            return []
        batch = [first]
        while not queue.empty():
            try:
                batch.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return [m for m in batch if m is not None]

    async def close(self) -> None:
        self._closed = True
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ── connection management ──────────────────────────────────────────
    def _ensure_connected(self) -> None:
        if self._closed:
            return
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop(), name=f"gateway-ws-{self.name}")

    async def _ws_loop(self) -> None:
        while not self._closed:
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect on any failure
                logger.warning("gateway_ws_disconnect platform=%s err=%s", self.name, exc)
            if self._closed:
                break
            await asyncio.sleep(self._reconnect_backoff)

    async def _run_connection(self) -> None:
        """Run one full WS connection (connect → handshake → read loop →
        queue inbound). Return/raise on disconnect; the loop reconnects."""
        raise NotImplementedError

    async def _put(self, message: Optional[InboundMessage]) -> None:
        if message is not None:
            await self._ensure_queue().put(message)

    # ── REST helper ────────────────────────────────────────────────────
    async def _http_post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        if self._http_transport is not None:
            return await self._http_transport(url, json=json, headers=headers)
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=json, headers=dict(headers) if headers else None)
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if isinstance(data, dict):
            data.setdefault("_status", resp.status_code)
            return data
        return {"_status": resp.status_code, "_body": data}


__all__ = ["_QueuedWSAdapter", "HttpTransport"]
