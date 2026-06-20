"""Inbound chat gateway — receive messages from chat platforms, run an agent
turn, reply.

Built-in: the executor owns the gateway framework + platform adapters —
**Telegram** (HTTP long-poll), **Discord** (Gateway WebSocket), and **Slack**
(Socket Mode WebSocket), none needing a public endpoint. A host declares
platforms in config and supplies a handler (``message in → reply text out``);
it ships no transport code. Run :class:`GatewayRunner` from the app lifespan.

    from geny_executor.gateway import build_gateway

    async def handler(msg):                 # msg: InboundMessage
        return await run_my_agent(msg.chat_id, msg.text)   # -> reply str

    runner = build_gateway(
        [{"platform": "telegram", "config": {"token": "123:abc"}}],
        handler,
    )
    await runner.start()
    ...
    await runner.shutdown()
"""

from geny_executor.gateway.adapter import PlatformAdapter
from geny_executor.gateway.discord import DiscordGatewayAdapter
from geny_executor.gateway.factory import (
    BUILTIN_GATEWAY_PLATFORMS,
    build_gateway,
    build_platform_adapter,
)
from geny_executor.gateway.runner import GatewayHandler, GatewayRunner
from geny_executor.gateway.slack import SlackGatewayAdapter
from geny_executor.gateway.telegram import TelegramGatewayAdapter
from geny_executor.gateway.types import GatewayReply, InboundMessage

__all__ = [
    "InboundMessage",
    "GatewayReply",
    "PlatformAdapter",
    "GatewayRunner",
    "GatewayHandler",
    "TelegramGatewayAdapter",
    "DiscordGatewayAdapter",
    "SlackGatewayAdapter",
    "BUILTIN_GATEWAY_PLATFORMS",
    "build_gateway",
    "build_platform_adapter",
]
