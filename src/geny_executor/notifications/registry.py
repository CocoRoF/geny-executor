"""NotificationEndpointRegistry — name → webhook URL + headers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationEndpoint:
    name: str
    url: str
    headers: Optional[Dict[str, str]] = None
    description: Optional[str] = None


class NotificationEndpointRegistry:
    def __init__(self) -> None:
        self._endpoints: Dict[str, NotificationEndpoint] = {}

    def register(self, endpoint: NotificationEndpoint) -> None:
        if endpoint.name in self._endpoints:
            logger.warning(
                "notification_endpoint_overwritten",
                extra={"endpoint_name": endpoint.name},
            )
        self._endpoints[endpoint.name] = endpoint

    def get(self, name: str) -> Optional[NotificationEndpoint]:
        return self._endpoints.get(name)

    def list(self) -> List[NotificationEndpoint]:
        return list(self._endpoints.values())


__all__ = ["NotificationEndpoint", "NotificationEndpointRegistry"]
