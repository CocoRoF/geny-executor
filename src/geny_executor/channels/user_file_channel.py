"""UserFileChannel ABC — host delivers files to the user's session."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class UserFileChannel(ABC):
    """Send a file to the end-user. The framework does not assume
    HTTP / WebSocket / Slack / etc — the host implements this and
    chooses the transport.

    Returns a dict the LLM-facing tool surfaces — typically a
    download URL plus expiry, or whatever the host wants the model
    (and downstream UI) to know about delivery.
    """

    @abstractmethod
    async def send(
        self,
        path: Path,
        *,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]: ...


__all__ = ["UserFileChannel"]
