"""Default implementation of Stage 1: Input."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.core.errors import StageError
from geny_executor.stages.s01_input.types import NormalizedInput
from geny_executor.stages.s01_input.interface import InputValidator, InputNormalizer
from geny_executor.stages.s01_input.artifact.default.validators import DefaultValidator
from geny_executor.stages.s01_input.artifact.default.normalizers import DefaultNormalizer


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
        self._validator = validator or DefaultValidator()
        self._normalizer = normalizer or DefaultNormalizer()

    @property
    def name(self) -> str:
        return "input"

    @property
    def order(self) -> int:
        return 1

    @property
    def category(self) -> str:
        return "ingress"

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

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="validator",
                current_impl=type(self._validator).__name__,
                available_impls=[
                    "DefaultValidator",
                    "PassthroughValidator",
                    "StrictValidator",
                    "SchemaValidator",
                ],
            ),
            StrategyInfo(
                slot_name="normalizer",
                current_impl=type(self._normalizer).__name__,
                available_impls=[
                    "DefaultNormalizer",
                    "MultimodalNormalizer",
                ],
            ),
        ]
