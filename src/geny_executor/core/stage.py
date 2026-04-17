"""Stage and Strategy abstract base classes — Dual Abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

if TYPE_CHECKING:
    from geny_executor.core.state import PipelineState

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@dataclass
class StrategyInfo:
    """Metadata about a strategy slot and its current implementation."""

    slot_name: str
    current_impl: str
    available_impls: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageDescription:
    """Metadata for Pipeline UI rendering."""

    name: str
    order: int
    category: str
    is_active: bool = True
    strategies: List[StrategyInfo] = field(default_factory=list)


class Strategy(ABC):
    """Stage 내부 로직의 교체 가능한 전략 — Level 2 추상화.

    각 Stage는 하나 이상의 Strategy 슬롯을 가지며,
    동일 Stage라도 Strategy 교체로 완전히 다른 동작을 수행할 수 있다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy unique name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description (for UI)."""
        return ""

    def configure(self, config: Dict[str, Any]) -> None:
        """Inject strategy-specific configuration."""
        pass

    @classmethod
    def config_schema(cls) -> Optional[Any]:
        """Return a :class:`ConfigSchema` describing configurable parameters.

        Override in subclasses to expose tunable parameters to the UI.
        Returns ``None`` when the strategy has no configurable parameters.
        """
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Strategy:
        """Create an instance and apply *config* in one step."""
        instance = cls()
        instance.configure(config)
        return instance

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a serializable dict.

        Override in subclasses that store runtime configuration.
        """
        return {}


class Stage(ABC, Generic[T_In, T_Out]):
    """파이프라인의 개별 단계 — Level 1 추상화.

    모든 Stage는 이 인터페이스를 구현해야 하며,
    execute()가 핵심 실행 로직, should_bypass()가 건너뛰기 판단을 담당한다.
    Stage 자체를 통째로 교체할 수 있다.
    """

    # Per-stage overrides populated lazily through accessors below. We avoid
    # requiring concrete stages to call ``super().__init__`` so that existing
    # stages keep working without edits.
    _tool_binding: Optional[Any] = None
    _model_override: Optional[Any] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage unique name (e.g., 'input', 'context', 'api')."""
        ...

    @property
    @abstractmethod
    def order(self) -> int:
        """Execution order within the pipeline (1-16)."""
        ...

    @property
    def category(self) -> str:
        """Stage classification: ingress, pre_flight, execution, decision, egress."""
        return "execution"

    @abstractmethod
    async def execute(self, input: T_In, state: PipelineState) -> T_Out:
        """Core execution logic.

        Args:
            input: Output from the previous stage, or initial input.
            state: Full pipeline state (read/write).

        Returns:
            Result to be passed as input to the next stage.
        """
        ...

    def should_bypass(self, state: PipelineState) -> bool:
        """Whether to skip this stage. Default False (always execute)."""
        return False

    async def on_enter(self, state: PipelineState) -> None:
        """Hook called when entering this stage (optional)."""
        pass

    async def on_exit(self, result: T_Out, state: PipelineState) -> None:
        """Hook called after stage execution (optional)."""
        pass

    async def on_error(self, error: Exception, state: PipelineState) -> Optional[T_Out]:
        """Hook called on error. Return None to propagate, or a value to recover."""
        return None

    def describe(self) -> StageDescription:
        """Return stage metadata for Pipeline UI."""
        return StageDescription(
            name=self.name,
            order=self.order,
            category=self.category,
            is_active=True,
            strategies=self.list_strategies(),
        )

    def list_strategies(self) -> List[StrategyInfo]:
        """List available strategies in this stage (for UI).

        Auto-generated from :meth:`get_strategy_slots` and
        :meth:`get_strategy_chains` when a concrete stage overrides them.
        Stages that still hardcode ``list_strategies`` should migrate to
        the slot/chain hooks.
        """
        infos: List[StrategyInfo] = []
        slots = self.get_strategy_slots()
        for slot in slots.values():
            infos.append(slot.describe())
        chains = self.get_strategy_chains()
        for chain in chains.values():
            infos.append(chain.describe())
        return infos

    # ── Mutation API (Phase 1) ──────────────────────────────────────

    def get_strategy_slots(self) -> Dict[str, Any]:
        """Return a dict of slot_name → :class:`StrategySlot`.

        Override in concrete stages that adopt the StrategySlot pattern.
        Stages that do not override this return an empty dict, preserving
        backward compatibility with the legacy ``list_strategies()`` approach.
        """
        return {}

    def get_strategy_chains(self) -> Dict[str, Any]:
        """Return a dict of chain_name → :class:`SlotChain`.

        Override in stages whose semantics require an ordered sequence
        of strategies (Guard, Emit). Default: no chains.
        """
        return {}

    def set_strategy(
        self, slot_name: str, impl_name: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Hot-swap a strategy in the named slot.

        Args:
            slot_name: Which slot to modify.
            impl_name: Registered implementation name.
            config: Optional configuration dict.

        Raises:
            KeyError: If *slot_name* does not exist or *impl_name* is unknown.
        """
        slots = self.get_strategy_slots()
        slot = slots.get(slot_name)
        if slot is None:
            raise KeyError(
                f"Stage '{self.name}' has no strategy slot '{slot_name}'. "
                f"Available: {list(slots.keys())}"
            )
        slot.swap(impl_name, config)

    def _get_chain(self, chain_name: str) -> Any:
        chains = self.get_strategy_chains()
        chain = chains.get(chain_name)
        if chain is None:
            raise KeyError(
                f"Stage '{self.name}' has no strategy chain '{chain_name}'. "
                f"Available: {list(chains.keys())}"
            )
        return chain

    def add_to_chain(
        self,
        chain_name: str,
        impl_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Append a new strategy onto the named chain."""
        return self._get_chain(chain_name).append(impl_name, config)

    def remove_from_chain(self, chain_name: str, item_name: str) -> Any:
        """Remove the first item named *item_name* from the chain."""
        return self._get_chain(chain_name).remove(item_name)

    def reorder_chain(self, chain_name: str, order: List[str]) -> None:
        """Reorder the chain to match *order* (a permutation of current names)."""
        self._get_chain(chain_name).reorder(order)

    def clear_chain(self, chain_name: str) -> None:
        """Empty the named chain."""
        self._get_chain(chain_name).clear()

    def get_config_schema(self) -> Optional[Any]:
        """Return a :class:`ConfigSchema` for this stage's own parameters.

        Override in concrete stages that have stage-level configuration
        (separate from strategy-level configs).
        """
        return None

    def get_config(self) -> Dict[str, Any]:
        """Return the stage's current configuration as a serializable dict."""
        return {}

    def update_config(self, config: Dict[str, Any]) -> None:
        """Apply partial configuration update to this stage."""
        pass

    # ── Per-stage overrides ───────────────────────────────────

    @property
    def tool_binding(self) -> Any:
        """Per-stage :class:`StageToolBinding` view of the tool registry.

        Lazily constructed on first access so existing stages work unchanged.
        """
        if self._tool_binding is None:
            from geny_executor.tools.stage_binding import StageToolBinding

            self._tool_binding = StageToolBinding(stage_order=self.order)
        return self._tool_binding

    @tool_binding.setter
    def tool_binding(self, value: Any) -> None:
        self._tool_binding = value

    @property
    def model_override(self) -> Any:
        """Optional per-stage :class:`ModelConfig` override. ``None`` = inherit."""
        return self._model_override

    @model_override.setter
    def model_override(self, value: Any) -> None:
        self._model_override = value

    def resolve_model(self, state: PipelineState) -> Any:
        """Return the effective model identifier for this stage.

        When :attr:`model_override` is set, its ``model`` attribute wins; otherwise
        falls back to the pipeline-wide ``state.model``. Model-using stages
        (API, agent sub-pipelines, evaluators, memory summarizers) should call
        this helper instead of reading ``state.model`` directly so the override
        is honored uniformly.
        """
        if self._model_override is not None:
            return getattr(self._model_override, "model", state.model)
        return state.model
