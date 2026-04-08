"""Pipeline engine — executes stages in order with loop control."""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from geny_executor.core.config import PipelineConfig
from geny_executor.core.errors import StageError
from geny_executor.core.result import PipelineResult
from geny_executor.core.stage import Stage, StageDescription
from geny_executor.core.state import PipelineState
from geny_executor.events.bus import EventBus
from geny_executor.events.types import PipelineEvent


class Pipeline:
    """Stage들을 순서대로 실행하는 파이프라인 엔진.

    Execution model:
      Phase A: Input (Stage 1, once)
      Phase B: Agent Loop (Stage 2~13, repeats)
      Phase C: Finalize (Stage 14~16, once)
    """

    # Loop boundary constants
    LOOP_START = 2
    LOOP_END = 13  # inclusive
    FINALIZE_START = 14
    FINALIZE_END = 16  # inclusive
    EVENT_DATA_TRUNCATE = 500  # max chars for event data preview

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._stages: Dict[int, Stage] = {}
        self._event_bus = EventBus()

    # ── Stage management ──

    def register_stage(self, stage: Stage) -> Pipeline:
        """Register or replace a stage. Supports chaining."""
        self._stages[stage.order] = stage
        return self

    def replace_stage(self, order: int, stage: Stage) -> Pipeline:
        """Replace stage at given order."""
        self._stages[order] = stage
        return self

    def remove_stage(self, order: int) -> Pipeline:
        """Remove stage (that slot will be bypassed)."""
        self._stages.pop(order, None)
        return self

    def get_stage(self, order: int) -> Optional[Stage]:
        """Get registered stage by order."""
        return self._stages.get(order)

    @property
    def stages(self) -> List[Stage]:
        """All registered stages, sorted by order."""
        return sorted(self._stages.values(), key=lambda s: s.order)

    # ── Execution ──

    async def run(self, input: Any, state: Optional[PipelineState] = None) -> PipelineResult:
        """Execute the full pipeline.

        Phase A: Stage 1 (Input) — runs once
        Phase B: Stage 2~13 (Agent Loop) — repeats until loop_decision != "continue"
        Phase C: Stage 14~16 (Finalize) — runs once
        """
        state = self._init_state(state)
        await self._emit("pipeline.start", data={"input": str(input)[:self.EVENT_DATA_TRUNCATE]})

        try:
            # Phase A: Input (stage 1)
            current = await self._run_stage(1, input, state)

            # Phase B: Agent Loop (stages 2~13)
            has_loop_stage = self.LOOP_END in self._stages
            while True:
                for order in range(self.LOOP_START, self.LOOP_END + 1):
                    current = await self._try_run_stage(order, current, state)

                # If no Loop stage is registered, auto-complete after one pass
                if not has_loop_stage and state.loop_decision == "continue":
                    state.loop_decision = "complete"

                if state.loop_decision != "continue":
                    break

                state.iteration += 1
                if state.is_over_iterations:
                    state.loop_decision = "complete"
                    state.completion_signal = "MAX_ITERATIONS"
                    state.add_event(
                        "loop.force_complete",
                        {"reason": "max_iterations", "iteration": state.iteration},
                    )
                    break

            # Phase C: Finalize (stages 14~16)
            for order in range(self.FINALIZE_START, self.FINALIZE_END + 1):
                current = await self._try_run_stage(order, current, state)

            result = PipelineResult.from_state(state)
            await self._emit("pipeline.complete", data={"iterations": state.iteration})
            return result

        except Exception as e:
            await self._emit("pipeline.error", data={"error": str(e)})
            return PipelineResult.error_result(str(e), state)

    async def run_stream(
        self, input: Any, state: Optional[PipelineState] = None
    ) -> AsyncIterator[PipelineEvent]:
        """Streaming mode — yields PipelineEvents in real-time.

        Each stage transition, API chunk, tool execution etc. is yielded.
        """
        state = self._init_state(state)
        collected_events: List[PipelineEvent] = []

        # Capture events
        def collector(event: PipelineEvent) -> None:
            collected_events.append(event)

        unsubscribe = self._event_bus.on("*", collector)

        try:
            yield PipelineEvent(type="pipeline.start", data={"input": str(input)[:self.EVENT_DATA_TRUNCATE]})

            # Phase A
            current = await self._run_stage(1, input, state)
            for ev in collected_events:
                yield ev
            collected_events.clear()

            # Phase B
            has_loop_stage = self.LOOP_END in self._stages
            while True:
                for order in range(self.LOOP_START, self.LOOP_END + 1):
                    current = await self._try_run_stage(order, current, state)
                    for ev in collected_events:
                        yield ev
                    collected_events.clear()

                if not has_loop_stage and state.loop_decision == "continue":
                    state.loop_decision = "complete"

                if state.loop_decision != "continue":
                    break
                state.iteration += 1
                if state.is_over_iterations:
                    state.loop_decision = "complete"
                    state.completion_signal = "MAX_ITERATIONS"
                    state.add_event(
                        "loop.force_complete",
                        {"reason": "max_iterations", "iteration": state.iteration},
                    )
                    break

            # Phase C
            for order in range(self.FINALIZE_START, self.FINALIZE_END + 1):
                current = await self._try_run_stage(order, current, state)
                for ev in collected_events:
                    yield ev
                collected_events.clear()

            yield PipelineEvent(
                type="pipeline.complete",
                data={"result": state.final_text[:self.EVENT_DATA_TRUNCATE], "iterations": state.iteration},
            )

        except Exception as e:
            yield PipelineEvent(type="pipeline.error", data={"error": str(e)})

        finally:
            unsubscribe()

    # ── Events ──

    def on(self, event_type: str, handler: Callable) -> Callable:
        """Register event handler. Returns unsubscribe function."""
        return self._event_bus.on(event_type, handler)

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus directly."""
        return self._event_bus

    # ── UI metadata ──

    def describe(self) -> List[StageDescription]:
        """Return pipeline structure for UI rendering."""
        descriptions = []
        for order in range(1, 17):
            stage = self._stages.get(order)
            if stage:
                desc = stage.describe()
                descriptions.append(desc)
            else:
                descriptions.append(
                    StageDescription(
                        name=f"stage_{order}",
                        order=order,
                        category="unregistered",
                        is_active=False,
                    )
                )
        return descriptions

    # ── Internal ──

    def _init_state(self, state: Optional[PipelineState]) -> PipelineState:
        """Initialize or apply config to state."""
        state = state or PipelineState()
        if not state.pipeline_id:
            state.pipeline_id = uuid.uuid4().hex[:12]
        self._config.apply_to_state(state)
        return state

    async def _try_run_stage(self, order: int, current: Any, state: PipelineState) -> Any:
        """Run a stage if it exists and should not be bypassed."""
        stage = self._stages.get(order)
        if stage is None:
            return current
        if stage.should_bypass(state):
            await self._emit("stage.bypass", stage=stage.name, iteration=state.iteration)
            return current
        return await self._run_stage(order, current, state)

    async def _run_stage(self, order: int, input: Any, state: PipelineState) -> Any:
        """Execute a single stage with lifecycle hooks."""
        stage = self._stages.get(order)
        if stage is None:
            return input

        state.current_stage = stage.name
        state.stage_history.append(stage.name)
        await self._emit("stage.enter", stage=stage.name, iteration=state.iteration)

        await stage.on_enter(state)
        try:
            result = await stage.execute(input, state)
            await stage.on_exit(result, state)
            await self._emit("stage.exit", stage=stage.name, iteration=state.iteration)
            return result
        except Exception as e:
            await self._emit(
                "stage.error",
                stage=stage.name,
                iteration=state.iteration,
                data={"error": str(e)},
            )
            recovery = await stage.on_error(e, state)
            if recovery is not None:
                return recovery
            raise StageError(
                str(e), stage_name=stage.name, stage_order=order, cause=e
            ) from e

    async def _emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit a pipeline event."""
        event = PipelineEvent(type=event_type, **kwargs)
        await self._event_bus.emit(event)
