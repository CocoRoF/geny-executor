"""Notification endpoints registry — host-supplied webhook targets.

The registry is service-instantiated. ``PushNotificationTool`` reads
endpoints from ``ToolContext.extras["notification_endpoints"]``.
"""

from geny_executor.notifications.registry import (
    NotificationEndpoint,
    NotificationEndpointRegistry,
)

__all__ = ["NotificationEndpoint", "NotificationEndpointRegistry"]
