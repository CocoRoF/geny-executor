"""Stage 3: System — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class PromptBuilder(Strategy):
    """Base interface for system prompt construction."""

    @abstractmethod
    def build(self, state: PipelineState) -> Union[str, List[Dict[str, Any]]]:
        """Build system prompt. Returns str or content blocks (for caching)."""

    def build_parts(self, state: PipelineState) -> Optional[List[Dict[str, Any]]]:
        """Optionally build the prompt as ordered stable/volatile parts.

        TTFT program (2.50.0): provider prompt caches key on the request
        PREFIX, so per-turn content (clock, retrieved memory) embedded in
        the system prompt re-prefills system + history on every turn.
        Builders that can tell the stable region from the volatile one
        return ``[{"name": str, "text": str, "volatile": bool}, ...]``
        in render order; Stage 3 then keeps the stable prefix cacheable
        and relocates the volatile tail (see ``volatile_placement``).

        Returning ``None`` (the default) means "no structure available"
        — Stage 3 falls back to :meth:`build` and behaves exactly as
        before, so custom host builders are unaffected.
        """
        return None


class PromptBlock(ABC):
    """A composable block of system prompt content."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this block."""

    @abstractmethod
    def render(self, state: PipelineState) -> str:
        """Render this block to text."""

    @property
    def cache_control(self) -> Optional[Dict[str, str]]:
        """Optional cache_control for this block."""
        return None

    @property
    def volatile(self) -> bool:
        """True when this block's text changes turn-to-turn.

        Volatile blocks (clock, per-turn retrieved memory) must stay OUT
        of the cached prompt prefix — one changed byte early in the
        prefix re-prefills everything after it on every turn. Stable is
        the safe default: mismarking stable-as-volatile only shrinks the
        cached region, mismarking volatile-as-stable thrashes the cache.
        """
        return False
