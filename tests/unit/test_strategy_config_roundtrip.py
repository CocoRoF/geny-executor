"""2.2.0 Wave 1 — strategy-level config as a real contract (audit §2.1).

Every strategy with constructor knobs must implement the
configure() / config_schema() / get_config() trio so that manifest
``strategy_configs`` actually land instead of vanishing into the base
no-op ``Strategy.configure``. These tests pin the round-trip half of
the contract:

    configure(cfg)  ⇒  get_config() ⊇ cfg

plus the validation half: bad input raises ``ValueError`` with a
message naming the offending key (the audit's "masked degradation"
pattern is a green check with no behaviour change — loud failure at
configure time is the fix).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import pytest

from geny_executor.core.schema import ConfigSchema
from geny_executor.stages.s06_api.artifact.default.retry import (
    ExponentialBackoffRetry,
    RateLimitAwareRetry,
)
from geny_executor.stages.s06_api.artifact.default.router import AdaptiveModelRouter
from geny_executor.stages.s10_tool.artifact.default.executors import (
    ParallelExecutor,
    PartitionExecutor,
)
from geny_executor.stages.s11_tool_review.artifact.default.reviewers import (
    DestructiveResultReviewer,
    NetworkAuditReviewer,
    SchemaReviewer,
    SensitivePatternReviewer,
    SizeReviewer,
)
from geny_executor.stages.s14_evaluate import (
    BinaryClassifyEvaluation,
    CriteriaBasedEvaluation,
    EvaluationChain,
    WeightedScorer,
)
from geny_executor.stages.s16_loop import (
    BudgetAwareLoopController,
    MultiDimensionalBudgetController,
    StandardLoopController,
)

# (factory, representative manifest config) — one row per strategy that
# gained the trio in this wave.
ROUNDTRIP_CASES: Tuple[Tuple[Callable[[], Any], Dict[str, Any]], ...] = (
    (
        EvaluationChain,
        {
            "evaluators": ["binary_classify", "signal_based"],
            "easy_max_turns": 2,
            "not_easy_max_turns": 12,
        },
    ),
    (BinaryClassifyEvaluation, {"easy_max_turns": 3, "not_easy_max_turns": 9}),
    (CriteriaBasedEvaluation, {"pass_threshold": 0.7}),
    (WeightedScorer, {"weights": {"relevance": 1.5, "length": 0.5}}),
    (
        MultiDimensionalBudgetController,
        {"dimensions": ["iterations"], "max_turns": 7},
    ),
    (StandardLoopController, {"max_turns": 5}),
    (
        BudgetAwareLoopController,
        {"cost_threshold_ratio": 0.5, "token_threshold_ratio": 0.6},
    ),
    (
        AdaptiveModelRouter,
        {
            "light_model": "claude-haiku-x",
            "balanced_model": "claude-sonnet-x",
            "heavy_model": "claude-opus-x",
            "light_threshold_chars": 100,
            "heavy_threshold_chars": 5_000,
            "thinking_promotes_heavy": False,
            "tools_promote_balanced": False,
        },
    ),
    (
        ExponentialBackoffRetry,
        {"max_retries": 2, "base_delay": 0.5, "max_delay": 10.0, "jitter": 0.2},
    ),
    (RateLimitAwareRetry, {"max_retries": 4, "fallback_delay": 2.0}),
    (SchemaReviewer, {"required_fields": {"Bash": ["command"]}}),
    (SensitivePatternReviewer, {"patterns": [["my_token", r"tok_[a-z0-9]{8}"]]}),
    (
        DestructiveResultReviewer,
        {"destructive_tools": ["Bash"], "severity": "warn"},
    ),
    (
        NetworkAuditReviewer,
        {"network_tools": ["WebFetch"], "allowed_hosts": ["example.com"]},
    ),
    (SizeReviewer, {"warn_threshold_bytes": 10, "error_threshold_bytes": 20}),
    (ParallelExecutor, {"max_concurrency": 3}),
    (PartitionExecutor, {"max_concurrency": 7}),
)

_IDS = [factory.__name__ for factory, _ in ROUNDTRIP_CASES]


@pytest.mark.parametrize("factory,cfg", ROUNDTRIP_CASES, ids=_IDS)
def test_configure_get_config_roundtrip(factory, cfg):
    strategy = factory()
    strategy.configure(cfg)
    observed = strategy.get_config()
    for key, expected in cfg.items():
        assert observed.get(key) == expected, (
            f"{factory.__name__}.get_config()[{key!r}] = {observed.get(key)!r}, "
            f"configured {expected!r}"
        )


@pytest.mark.parametrize("factory,cfg", ROUNDTRIP_CASES, ids=_IDS)
def test_config_schema_describes_every_configured_key(factory, cfg):
    schema = factory.config_schema()
    assert isinstance(schema, ConfigSchema), f"{factory.__name__} must expose a ConfigSchema"
    declared = {f.name for f in schema.fields}
    missing = set(cfg) - declared
    assert not missing, f"{factory.__name__}.config_schema() missing fields: {missing}"


@pytest.mark.parametrize("factory,cfg", ROUNDTRIP_CASES, ids=_IDS)
def test_from_config_one_step_construction(factory, cfg):
    """``Strategy.from_config`` (cls() + configure) must work for the
    slot-swap path — that is exactly what ``StrategySlot.swap`` does on
    manifest restore."""
    strategy = factory.from_config(cfg)
    observed = strategy.get_config()
    for key, expected in cfg.items():
        assert observed.get(key) == expected


@pytest.mark.parametrize("factory,cfg", ROUNDTRIP_CASES, ids=_IDS)
def test_empty_configure_is_a_noop(factory, cfg):
    """configure({}) must never change state — manifest entries without a
    strategy_configs block replay as empty dicts on restore."""
    strategy = factory()
    strategy.configure(cfg)
    before = strategy.get_config()
    strategy.configure({})
    assert strategy.get_config() == before


# ── Validation: precise ValueError on bad input ─────────────────────────


BAD_INPUT_CASES = (
    (EvaluationChain, {"evaluators": ["no_such_evaluator"]}, "no_such_evaluator"),
    (EvaluationChain, {"evaluators": "binary_classify"}, "evaluators"),
    (EvaluationChain, {"easy_max_turns": 0}, "easy_max_turns"),
    (EvaluationChain, {"not_easy_max_turns": True}, "not_easy_max_turns"),
    (BinaryClassifyEvaluation, {"easy_max_turns": "lots"}, "easy_max_turns"),
    (CriteriaBasedEvaluation, {"pass_threshold": 1.5}, "pass_threshold"),
    (WeightedScorer, {"weights": ["not", "a", "dict"]}, "weights"),
    (MultiDimensionalBudgetController, {"dimensions": ["bogus_dim"]}, "bogus_dim"),
    (MultiDimensionalBudgetController, {"dimensions": "iterations"}, "dimensions"),
    (MultiDimensionalBudgetController, {"dimensions": ["wall_clock"]}, "max_seconds"),
    (MultiDimensionalBudgetController, {"dimensions": ["tool_calls"]}, "max_tool_calls"),
    (MultiDimensionalBudgetController, {"max_turns": -1}, "max_turns"),
    (StandardLoopController, {"max_turns": True}, "max_turns"),
    (BudgetAwareLoopController, {"cost_threshold_ratio": 2.0}, "cost_threshold_ratio"),
    (AdaptiveModelRouter, {"light_model": ""}, "light_model"),
    (AdaptiveModelRouter, {"light_threshold_chars": -5}, "light_threshold_chars"),
    (
        AdaptiveModelRouter,
        {"light_threshold_chars": 5_000, "heavy_threshold_chars": 100},
        "heavy_threshold_chars",
    ),
    (AdaptiveModelRouter, {"thinking_promotes_heavy": "yes"}, "thinking_promotes_heavy"),
    (ExponentialBackoffRetry, {"max_retries": -1}, "max_retries"),
    (ExponentialBackoffRetry, {"base_delay": 30.0, "max_delay": 5.0}, "max_delay"),
    (ExponentialBackoffRetry, {"jitter": 3.0}, "jitter"),
    (RateLimitAwareRetry, {"fallback_delay": "soon"}, "fallback_delay"),
    (SchemaReviewer, {"required_fields": ["Bash"]}, "required_fields"),
    (SensitivePatternReviewer, {"patterns": [["broken", "("]]}, "broken"),
    (SensitivePatternReviewer, {"patterns": [["only-label"]]}, "pattern"),
    (DestructiveResultReviewer, {"severity": "fatal"}, "severity"),
    (NetworkAuditReviewer, {"allowed_hosts": "example.com"}, "allowed_hosts"),
    (SizeReviewer, {"warn_threshold_bytes": 100, "error_threshold_bytes": 5}, "error_threshold_bytes"),
    (ParallelExecutor, {"max_concurrency": 0}, "max_concurrency"),
    (PartitionExecutor, {"max_concurrency": False}, "max_concurrency"),
)


@pytest.mark.parametrize(
    "factory,bad_cfg,needle",
    BAD_INPUT_CASES,
    ids=[f"{f.__name__}-{n}" for f, _, n in BAD_INPUT_CASES],
)
def test_configure_rejects_bad_input_with_precise_message(factory, bad_cfg, needle):
    strategy = factory()
    with pytest.raises(ValueError, match=needle):
        strategy.configure(bad_cfg)


@pytest.mark.parametrize(
    "factory,bad_cfg,needle",
    BAD_INPUT_CASES,
    ids=[f"{f.__name__}-{n}" for f, _, n in BAD_INPUT_CASES],
)
def test_rejected_configure_does_not_partially_apply(factory, bad_cfg, needle):
    """A failed configure must not leave the strategy half-updated —
    operators retry after fixing the manifest and expect the previous
    (working) configuration to still be live."""
    strategy = factory()
    before = strategy.get_config()
    with pytest.raises(ValueError):
        strategy.configure(bad_cfg)
    assert strategy.get_config() == before
