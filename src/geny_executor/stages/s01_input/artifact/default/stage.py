"""Default implementation of Stage 1: Input."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.core.errors import StageError
from geny_executor.stages.s01_input.types import NormalizedInput
from geny_executor.stages.s01_input.interface import InputValidator, InputNormalizer
from geny_executor.stages.s01_input.artifact.default.validators import (
    DefaultValidator,
    PassthroughValidator,
    SchemaValidator,
    StrictValidator,
)
from geny_executor.stages.s01_input.artifact.default.normalizers import (
    DefaultNormalizer,
    MultimodalNormalizer,
)


class InputStage(Stage[Any, NormalizedInput]):
    """Stage 1: Input — default artifact.

    Dual abstraction:
      - Level 2 validator: validates raw input
      - Level 2 normalizer: transforms to NormalizedInput
    """

    def __init__(
        self,
        validator: Optional[InputValidator] = None,
        normalizer: Optional[InputNormalizer] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "validator": StrategySlot(
                name="validator",
                strategy=validator or DefaultValidator(),
                registry={
                    "default": DefaultValidator,
                    "passthrough": PassthroughValidator,
                    "strict": StrictValidator,
                    "schema": SchemaValidator,
                },
                description="Raw input validation strategy",
            ),
            "normalizer": StrategySlot(
                name="normalizer",
                strategy=normalizer or DefaultNormalizer(),
                registry={
                    "default": DefaultNormalizer,
                    "multimodal": MultimodalNormalizer,
                },
                description="Input normalization strategy",
            ),
        }

    @property
    def _validator(self) -> InputValidator:
        return self._slots["validator"].strategy  # type: ignore[return-value]

    @property
    def _normalizer(self) -> InputNormalizer:
        return self._slots["normalizer"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "input"

    @property
    def order(self) -> int:
        return 1

    @property
    def category(self) -> str:
        return "ingress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    async def execute(self, input: Any, state: PipelineState) -> NormalizedInput:
        # Validate
        error = self._validator.validate(input)
        if error:
            raise StageError(
                f"Input validation failed: {error}",
                stage_name=self.name,
                stage_order=self.order,
            )

        # Normalize
        normalized = self._normalizer.normalize(input)
        normalized.session_id = state.session_id

        # Add user message to state
        state.add_message("user", normalized.to_message_content())
        state.add_event("input.normalized", {"text_length": len(normalized.text)})

        return normalized
