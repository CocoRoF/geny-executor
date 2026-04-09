"""Phase 4 tests — Think, Agent, Evaluate stages."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s06_api.retry import NoRetry
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage

# Think imports
from geny_executor.stages.s08_think import (
    ThinkStage,
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingFilterProcessor,
    ThinkingBlock,
    ThinkingResult,
)

# Agent imports
from geny_executor.stages.s11_agent import (
    AgentStage,
    SingleAgentOrchestrator,
    DelegateOrchestrator,
    DefaultSubPipelineFactory,
)

# Evaluate imports
from geny_executor.stages.s12_evaluate import (
    EvaluateStage,
    SignalBasedEvaluation,
    CriteriaBasedEvaluation,
    QualityCriterion,
    NoScorer,
    WeightedScorer,
)


# ── Think Stage ──


@pytest.mark.asyncio
async def test_think_bypass_when_disabled():
    """ThinkStage bypasses when thinking is not enabled."""
    stage = ThinkStage()
    state = PipelineState()
    state.thinking_enabled = False
    assert stage.should_bypass(state) is True


@pytest.mark.asyncio
async def test_think_bypass_when_no_response():
    """ThinkStage bypasses when no API response."""
    stage = ThinkStage()
    state = PipelineState()
    state.thinking_enabled = True
    state.last_api_response = None
    assert stage.should_bypass(state) is True


@pytest.mark.asyncio
async def test_think_extracts_thinking_blocks():
    """ThinkStage separates thinking from response blocks."""
    processor = ExtractAndStoreProcessor()
    stage = ThinkStage(processor=processor)

    state = PipelineState()
    state.thinking_enabled = True
    state.last_api_response = {
        "content": [
            {"type": "thinking", "thinking": "Let me analyze this...", "budget_tokens_used": 150},
            {"type": "text", "text": "The answer is 42."},
        ]
    }

    result = await stage.execute(None, state)
    assert isinstance(result, ThinkingResult)
    assert len(result.thinking_blocks) == 1
    assert result.thinking_blocks[0].text == "Let me analyze this..."
    assert len(result.response_blocks) == 1
    assert result.response_blocks[0]["type"] == "text"


@pytest.mark.asyncio
async def test_think_stores_in_history():
    """ExtractAndStoreProcessor stores thinking in state."""
    processor = ExtractAndStoreProcessor()
    state = PipelineState()
    state.iteration = 3

    blocks = [ThinkingBlock(text="Reasoning step 1", budget_tokens_used=100)]
    await processor.process(blocks, state)

    assert len(state.thinking_history) == 1
    assert state.thinking_history[0]["text"] == "Reasoning step 1"
    assert state.thinking_history[0]["iteration"] == 3


@pytest.mark.asyncio
async def test_think_passthrough_processor():
    """PassthroughProcessor returns blocks unchanged."""
    processor = PassthroughProcessor()
    state = PipelineState()
    blocks = [ThinkingBlock(text="test")]
    result = await processor.process(blocks, state)
    assert result == blocks
    assert len(state.thinking_history) == 0  # no storage


@pytest.mark.asyncio
async def test_think_filter_processor():
    """ThinkingFilterProcessor removes matching blocks."""
    processor = ThinkingFilterProcessor(exclude_patterns=["SECRET"])
    state = PipelineState()

    blocks = [
        ThinkingBlock(text="Normal reasoning"),
        ThinkingBlock(text="Contains SECRET info"),
        ThinkingBlock(text="More reasoning"),
    ]
    result = await processor.process(blocks, state)
    assert len(result) == 2
    assert all("SECRET" not in b.text for b in result)


# ── Agent Stage ──


@pytest.mark.asyncio
async def test_agent_single_orchestrator_bypass():
    """SingleAgentOrchestrator bypasses with no delegates."""
    stage = AgentStage(orchestrator=SingleAgentOrchestrator())
    state = PipelineState()
    assert stage.should_bypass(state) is True


@pytest.mark.asyncio
async def test_agent_single_orchestrator_no_delegation():
    """SingleAgentOrchestrator returns no delegation."""
    orchestrator = SingleAgentOrchestrator()
    state = PipelineState()
    result = await orchestrator.orchestrate(state)
    assert result.delegated is False
    assert result.sub_results == []


@pytest.mark.asyncio
async def test_agent_delegate_orchestrator():
    """DelegateOrchestrator delegates to sub-pipeline."""
    # Create a sub-pipeline factory
    factory = DefaultSubPipelineFactory()

    def create_sub():
        p = Pipeline(PipelineConfig(name="sub"))
        p.register_stage(InputStage())
        p.register_stage(
            APIStage(provider=MockProvider(default_text="Sub result"), retry=NoRetry())
        )
        p.register_stage(ParseStage())
        p.register_stage(YieldStage())
        return p

    factory.register("researcher", create_sub)

    orchestrator = DelegateOrchestrator(factory=factory)
    state = PipelineState(session_id="main")
    state.delegate_requests = [
        {"agent_type": "researcher", "task": "Find information about X"},
    ]

    result = await orchestrator.orchestrate(state)
    assert result.delegated is True
    assert len(result.sub_results) == 1
    assert result.sub_results[0]["success"] is True
    assert result.sub_results[0]["text"] == "Sub result"


@pytest.mark.asyncio
async def test_agent_delegate_unknown_type():
    """DelegateOrchestrator handles unknown agent type."""
    factory = DefaultSubPipelineFactory()
    orchestrator = DelegateOrchestrator(factory=factory)
    state = PipelineState()
    state.delegate_requests = [
        {"agent_type": "unknown", "task": "Do something"},
    ]

    result = await orchestrator.orchestrate(state)
    assert result.delegated is True
    assert result.sub_results[0]["success"] is False
    assert "Unknown agent type" in result.sub_results[0]["error"]


@pytest.mark.asyncio
async def test_agent_stage_integrates_results():
    """AgentStage integrates sub-agent results into state."""
    factory = DefaultSubPipelineFactory()

    def create_sub():
        p = Pipeline(PipelineConfig(name="sub"))
        p.register_stage(InputStage())
        p.register_stage(APIStage(provider=MockProvider(default_text="Done"), retry=NoRetry()))
        p.register_stage(ParseStage())
        p.register_stage(YieldStage())
        return p

    factory.register("helper", create_sub)

    stage = AgentStage(orchestrator=DelegateOrchestrator(factory=factory))
    state = PipelineState(session_id="test")
    state.delegate_requests = [
        {"agent_type": "helper", "task": "Help me"},
    ]

    await stage.execute(None, state)
    assert len(state.agent_results) == 1
    assert state.agent_results[0]["success"] is True


# ── Evaluate Stage ──


@pytest.mark.asyncio
async def test_evaluate_signal_complete():
    """SignalBasedEvaluation detects complete signal."""
    strategy = SignalBasedEvaluation()
    state = PipelineState()
    state.completion_signal = "complete"
    state.completion_detail = "Task done."

    result = await strategy.evaluate(state)
    assert result.passed is True
    assert result.decision == "complete"
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_evaluate_signal_blocked():
    """SignalBasedEvaluation detects blocked signal."""
    strategy = SignalBasedEvaluation()
    state = PipelineState()
    state.completion_signal = "blocked"

    result = await strategy.evaluate(state)
    assert result.passed is False
    assert result.decision == "escalate"


@pytest.mark.asyncio
async def test_evaluate_signal_continue():
    """SignalBasedEvaluation continues with no signal."""
    strategy = SignalBasedEvaluation()
    state = PipelineState()
    state.completion_signal = None

    result = await strategy.evaluate(state)
    assert result.decision == "continue"


@pytest.mark.asyncio
async def test_evaluate_criteria_pass():
    """CriteriaBasedEvaluation passes with good scores."""
    criteria = [
        QualityCriterion(
            name="length",
            description="Response length check",
            weight=1.0,
            threshold=0.5,
            check=lambda s: 0.8 if len(s.final_text) > 0 else 0.0,
        ),
    ]
    strategy = CriteriaBasedEvaluation(criteria=criteria, pass_threshold=0.6)
    state = PipelineState()
    state.final_text = "Some response text"

    result = await strategy.evaluate(state)
    assert result.passed is True
    assert result.score >= 0.6


@pytest.mark.asyncio
async def test_evaluate_criteria_fail():
    """CriteriaBasedEvaluation fails with low scores."""
    criteria = [
        QualityCriterion(
            name="length",
            description="Response length check",
            weight=1.0,
            threshold=0.5,
            check=lambda s: 0.2,  # Always low
        ),
    ]
    strategy = CriteriaBasedEvaluation(criteria=criteria, pass_threshold=0.6)
    state = PipelineState()

    result = await strategy.evaluate(state)
    assert result.passed is False
    assert result.decision == "retry"


@pytest.mark.asyncio
async def test_evaluate_stage_sets_loop_decision():
    """EvaluateStage maps evaluation to loop_decision."""
    stage = EvaluateStage(strategy=SignalBasedEvaluation())
    state = PipelineState()
    state.completion_signal = "complete"

    await stage.execute(None, state)
    assert state.loop_decision == "complete"
    assert state.evaluation_score == 1.0


@pytest.mark.asyncio
async def test_evaluate_stage_escalation():
    """EvaluateStage maps blocked signal to escalate loop_decision."""
    stage = EvaluateStage(strategy=SignalBasedEvaluation())
    state = PipelineState()
    state.completion_signal = "blocked"

    await stage.execute(None, state)
    assert state.loop_decision == "escalate"


@pytest.mark.asyncio
async def test_evaluate_stage_error():
    """EvaluateStage maps error signal to error loop_decision."""
    stage = EvaluateStage(strategy=SignalBasedEvaluation())
    state = PipelineState()
    state.completion_signal = "error"

    await stage.execute(None, state)
    assert state.loop_decision == "error"


@pytest.mark.asyncio
async def test_weighted_scorer():
    """WeightedScorer calculates weighted average."""
    scorer = WeightedScorer(weights={"quality": 0.7, "relevance": 0.3})
    state = PipelineState()
    state.metadata["quality"] = 0.9
    state.metadata["relevance"] = 0.8

    score = scorer.score(state)
    expected = (0.9 * 0.7 + 0.8 * 0.3) / (0.7 + 0.3)
    assert abs(score - expected) < 0.01


@pytest.mark.asyncio
async def test_no_scorer_returns_one():
    """NoScorer always returns 1.0."""
    scorer = NoScorer()
    state = PipelineState()
    assert scorer.score(state) == 1.0


# ── Loop respects Evaluate decisions ──


@pytest.mark.asyncio
async def test_loop_respects_evaluate_complete():
    """LoopStage respects Stage 12's 'complete' decision."""
    from geny_executor.stages.s13_loop import LoopStage, StandardLoopController

    stage = LoopStage(StandardLoopController(max_turns=50))
    state = PipelineState()
    state.loop_decision = "complete"  # Set by Stage 12

    await stage.execute(None, state)
    assert state.loop_decision == "complete"  # Should NOT be overwritten


@pytest.mark.asyncio
async def test_loop_respects_evaluate_error():
    """LoopStage respects Stage 12's 'error' decision."""
    from geny_executor.stages.s13_loop import LoopStage, StandardLoopController

    stage = LoopStage(StandardLoopController(max_turns=50))
    state = PipelineState()
    state.loop_decision = "error"  # Set by Stage 12

    await stage.execute(None, state)
    assert state.loop_decision == "error"


@pytest.mark.asyncio
async def test_loop_overrides_continue_with_controller():
    """LoopStage applies controller logic when upstream says 'continue'."""
    from geny_executor.stages.s13_loop import LoopStage, StandardLoopController

    stage = LoopStage(StandardLoopController(max_turns=50))
    state = PipelineState()
    state.loop_decision = "continue"
    # No tool results, no pending tools, no signal → controller should say "complete"

    await stage.execute(None, state)
    assert state.loop_decision == "complete"


# ── Integration: Full pipeline with Think + Evaluate ──


@pytest.mark.asyncio
async def test_pipeline_with_think_and_evaluate():
    """Full pipeline with Think and Evaluate stages."""
    provider = MockProvider(default_text="Analyzed result")
    pipeline = Pipeline(PipelineConfig(name="think-eval"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ThinkStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(EvaluateStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    result = await pipeline.run("Think about this", state)
    assert result.success is True
    assert result.text == "Analyzed result"
