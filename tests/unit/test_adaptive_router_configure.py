"""2.2.0 Wave 1 — AdaptiveModelRouter configure() + _estimate_chars fix.

Audit §1-2 found two problems in this router:
  1. every tuning knob was constructor-only (unreachable from a manifest), and
  2. ``_estimate_chars`` counted ``len(state.system)`` — after Stage 5's
     cache strategies convert the system prompt to a content-block LIST,
     that's the number of blocks (~1), so any system prompt collapsed to
     ~1 char and the size heuristic was skewed toward the light tier.
"""

from __future__ import annotations

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.artifact.default.router import AdaptiveModelRouter


def _state(
    *,
    system="",
    user_text: str = "hi",
    tools: list | None = None,
) -> PipelineState:
    state = PipelineState(session_id="s")
    state.system = system
    state.messages = [{"role": "user", "content": user_text}]
    state.tools = list(tools or [])
    return state


def _cfg(**overrides) -> ModelConfig:
    base = dict(model="some-baseline-model")
    base.update(overrides)
    return ModelConfig(**base)


class TestConfiguredThresholdsHonoured:
    def test_configured_light_threshold_routes_light(self):
        router = AdaptiveModelRouter.from_config(
            {
                "light_model": "tier-light",
                "balanced_model": "tier-balanced",
                "heavy_model": "tier-heavy",
                "light_threshold_chars": 10,
                "heavy_threshold_chars": 50,
            }
        )
        routed = router.route(_cfg(), _state(user_text="x" * 5))
        assert routed is not None and routed.model == "tier-light"

    def test_configured_heavy_threshold_routes_heavy(self):
        router = AdaptiveModelRouter.from_config(
            {
                "heavy_model": "tier-heavy",
                "light_threshold_chars": 10,
                "heavy_threshold_chars": 50,
            }
        )
        routed = router.route(_cfg(), _state(user_text="x" * 60))
        assert routed is not None and routed.model == "tier-heavy"

    def test_mid_size_routes_balanced(self):
        router = AdaptiveModelRouter.from_config(
            {
                "balanced_model": "tier-balanced",
                "light_threshold_chars": 10,
                "heavy_threshold_chars": 50,
            }
        )
        routed = router.route(_cfg(), _state(user_text="x" * 30))
        assert routed is not None and routed.model == "tier-balanced"

    def test_thinking_promotion_can_be_disabled(self):
        router = AdaptiveModelRouter.from_config(
            {
                "light_model": "tier-light",
                "light_threshold_chars": 100,
                "heavy_threshold_chars": 5_000,
                "thinking_promotes_heavy": False,
            }
        )
        routed = router.route(_cfg(thinking_enabled=True), _state(user_text="short"))
        assert routed is not None and routed.model == "tier-light"

    def test_tools_promotion_can_be_disabled(self):
        router = AdaptiveModelRouter.from_config(
            {
                "light_model": "tier-light",
                "light_threshold_chars": 100,
                "heavy_threshold_chars": 5_000,
                "tools_promote_balanced": False,
            }
        )
        state = _state(user_text="short", tools=[{"name": "Bash"}])
        routed = router.route(_cfg(), state)
        assert routed is not None and routed.model == "tier-light"


class TestEstimateCharsBlockListRegression:
    """Regression for the s05 interaction: system as a list of cache-marked
    content blocks must contribute its TEXT length, not the block count."""

    def test_block_list_system_counts_text_chars(self):
        # The exact shape SystemCacheStrategy leaves behind.
        system_blocks = [
            {
                "type": "text",
                "text": "S" * 20_000,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        estimate = AdaptiveModelRouter._estimate_chars(_state(system=system_blocks))
        assert estimate >= 20_000  # pre-fix this was ~1 + len("hi")

    def test_block_list_system_promotes_heavy_tier(self):
        system_blocks = [
            {
                "type": "text",
                "text": "S" * 20_000,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        router = AdaptiveModelRouter(heavy_model="tier-heavy")
        routed = router.route(_cfg(), _state(system=system_blocks))
        assert routed is not None and routed.model == "tier-heavy"

    def test_string_system_unchanged(self):
        estimate = AdaptiveModelRouter._estimate_chars(
            _state(system="S" * 500, user_text="hi")
        )
        assert estimate == 502

    def test_multi_block_and_string_block_forms(self):
        system_blocks = [
            {"type": "text", "text": "A" * 100},
            {"type": "text", "text": "B" * 200},
            "C" * 50,  # tolerated stray string block
        ]
        estimate = AdaptiveModelRouter._estimate_chars(_state(system=system_blocks, user_text=""))
        assert estimate == 350

    def test_none_system_is_zero(self):
        assert AdaptiveModelRouter._estimate_chars(_state(system=None, user_text="")) == 0
