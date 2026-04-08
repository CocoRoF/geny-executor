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


class Stage(ABC, Generic[T_In, T_Out]):
    """파이프라인의 개별 단계 — Level 1 추상화.

    모든 Stage는 이 인터페이스를 구현해야 하며,
    execute()가 핵심 실행 로직, should_bypass()가 건너뛰기 판단을 담당한다.
    Stage 자체를 통째로 교체할 수 있다.
    """

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
        """List available strategies in this stage (for UI)."""
        return []
