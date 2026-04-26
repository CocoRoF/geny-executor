"""Host-supplied delivery channels — files / messages / events.

The framework ships ABCs; hosts implement and inject impls via
``ToolContext.extras``. Keeping channel impls out of the framework
means the host owns transport (HTTP / Slack / SMS / queue) and we
don't add unnecessary deps.
"""

from geny_executor.channels.send_message_channel import (
    SendMessageChannel,
    SendMessageChannelRegistry,
    StdoutSendMessageChannel,
)
from geny_executor.channels.user_file_channel import UserFileChannel

__all__ = [
    "SendMessageChannel",
    "SendMessageChannelRegistry",
    "StdoutSendMessageChannel",
    "UserFileChannel",
]
