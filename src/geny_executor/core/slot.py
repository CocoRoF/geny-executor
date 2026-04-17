"""StrategySlot / SlotChain — typed containers for strategies.

A :class:`StrategySlot` wraps a single Strategy slot inside a Stage,
tracking the current instance and the registry of available implementations.

A :class:`SlotChain` holds an ordered sequence of Strategy instances
(used by stages whose semantics require running many strategies in turn,
e.g. Guard and Emit). Both expose the same :meth:`describe` contract so
that :meth:`Stage.list_strategies` can treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

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


@dataclass
class SlotChain:
    """An ordered chain of Strategy instances with a shared registry.

    Designed for stages like Guard (s04) and Emit (s14) that run an
    ordered sequence of strategies rather than selecting exactly one.
    Supports :meth:`append`, :meth:`remove`, :meth:`reorder`, and
    :meth:`clear` for runtime mutation while keeping a single registry
    of legal implementations for introspection.

    Attributes:
        name: Chain identifier (e.g., ``"guards"``, ``"emitters"``).
        items: Current ordered list of Strategy instances.
        registry: Mapping from implementation name → Strategy class.
        description: Human-readable description of the chain's role.
    """

    name: str
    items: List[Strategy] = field(default_factory=list)
    registry: Dict[str, Type[Strategy]] = field(default_factory=dict)
    description: str = ""

    @property
    def current_impl(self) -> str:
        """Comma-separated list of active item names, or ``"none"``."""
        if not self.items:
            return "none"
        return ", ".join(item.name for item in self.items)

    @property
    def available_impls(self) -> List[str]:
        """Sorted list of registered implementation names."""
        return sorted(self.registry.keys())

    def add(self, item: Strategy) -> SlotChain:
        """Append a pre-constructed Strategy instance."""
        self.items.append(item)
        return self

    def append(
        self, impl_name: str, config: Optional[Dict[str, Any]] = None
    ) -> Strategy:
        """Instantiate *impl_name* from the registry and append it.

        Args:
            impl_name: Key in :pyattr:`registry`.
            config: Optional configuration dict forwarded via ``configure()``.

        Returns:
            The newly appended Strategy instance.

        Raises:
            KeyError: If *impl_name* is not in the registry.
        """
        cls = self.registry.get(impl_name)
        if cls is None:
            raise KeyError(
                f"Strategy '{impl_name}' not found in chain '{self.name}'. "
                f"Available: {self.available_impls}"
            )
        instance = cls()
        if config:
            instance.configure(config)
        self.items.append(instance)
        return instance

    def remove(self, item_name: str) -> Strategy:
        """Remove and return the first item whose ``name`` matches.

        Raises:
            KeyError: If no item with that name exists.
        """
        for idx, item in enumerate(self.items):
            if item.name == item_name:
                return self.items.pop(idx)
        raise KeyError(
            f"No item named '{item_name}' in chain '{self.name}'. "
            f"Current: {[i.name for i in self.items]}"
        )

    def reorder(self, order: List[str]) -> None:
        """Reorder items to match the given sequence of item names.

        The sequence must enumerate every current item exactly once.

        Raises:
            KeyError: If *order* references a name not currently present.
            ValueError: If *order* is not a permutation of current items.
        """
        by_name: Dict[str, Strategy] = {}
        for item in self.items:
            by_name.setdefault(item.name, item)
        missing = [n for n in order if n not in by_name]
        if missing:
            raise KeyError(
                f"Chain '{self.name}' has no items named {missing}. "
                f"Current: {list(by_name.keys())}"
            )
        if len(order) != len(self.items) or set(order) != set(by_name.keys()):
            raise ValueError(
                f"reorder() must list every item in chain '{self.name}'. "
                f"Expected {sorted(by_name.keys())}, got {sorted(order)}"
            )
        self.items = [by_name[n] for n in order]

    def clear(self) -> None:
        """Remove all items from the chain."""
        self.items = []

    def describe(self) -> StrategyInfo:
        """Produce a :class:`StrategyInfo` describing chain contents."""
        item_configs: List[Dict[str, Any]] = []
        for item in self.items:
            cfg: Dict[str, Any] = {}
            if hasattr(item, "get_config"):
                cfg = item.get_config()
            item_configs.append({"name": item.name, "config": cfg})
        return StrategyInfo(
            slot_name=self.name,
            current_impl=self.current_impl,
            available_impls=self.available_impls,
            config={"items": item_configs},
        )
