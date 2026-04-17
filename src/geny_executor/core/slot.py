"""StrategySlot — typed container for hot-swappable strategies.

A StrategySlot wraps a single Strategy slot inside a Stage,
tracking the current instance and the registry of available implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from geny_executor.core.schema import ConfigSchema
from geny_executor.core.stage import Strategy, StrategyInfo


@dataclass
class StrategySlot:
    """A named slot holding one Strategy and a registry of alternatives.

    Attributes:
        name: Slot identifier (e.g., ``"provider"``, ``"compactor"``).
        strategy: Currently active Strategy instance.
        registry: Mapping from implementation name → Strategy class.
        required: Whether this slot must always have an active strategy.
        description: Human-readable explanation of what this slot controls.
    """

    name: str
    strategy: Strategy
    registry: Dict[str, Type[Strategy]] = field(default_factory=dict)
    required: bool = True
    description: str = ""

    @property
    def current_impl(self) -> str:
        """Name of the active implementation."""
        return self.strategy.name

    @property
    def available_impls(self) -> List[str]:
        """Sorted list of registered implementation names."""
        return sorted(self.registry.keys())

    def swap(self, impl_name: str, config: Optional[Dict[str, Any]] = None) -> Strategy:
        """Replace the active strategy with *impl_name* from the registry.

        Args:
            impl_name: Key in :pyattr:`registry`.
            config: Optional configuration dict forwarded via ``configure()``.

        Returns:
            The newly created Strategy instance (also stored in ``self.strategy``).

        Raises:
            KeyError: If *impl_name* is not in the registry.
        """
        cls = self.registry.get(impl_name)
        if cls is None:
            raise KeyError(
                f"Strategy '{impl_name}' not found in slot '{self.name}'. "
                f"Available: {self.available_impls}"
            )
        instance = cls()
        if config:
            instance.configure(config)
        self.strategy = instance
        return instance

    def describe(self) -> StrategyInfo:
        """Produce a :class:`StrategyInfo` compatible with the existing API."""
        config: Dict[str, Any] = {}
        if hasattr(self.strategy, "get_config"):
            config = self.strategy.get_config()
        return StrategyInfo(
            slot_name=self.name,
            current_impl=self.current_impl,
            available_impls=self.available_impls,
            config=config,
        )
