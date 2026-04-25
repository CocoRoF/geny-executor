"""Stage 6: API — default artifact."""

from geny_executor.stages.s06_api.artifact.default.stage import APIStage
from geny_executor.stages.s06_api.artifact.default.providers import (
    AnthropicProvider,
    MockProvider,
    RecordingProvider,
)
from geny_executor.stages.s06_api.artifact.default.retry import (
    ExponentialBackoffRetry,
    NoRetry,
    RateLimitAwareRetry,
)
from geny_executor.stages.s06_api.artifact.default.router import (
    AdaptiveModelRouter,
    PassthroughRouter,
)

# Canonical alias
Stage = APIStage

__all__ = [
    "Stage",
    "APIStage",
    "AnthropicProvider",
    "MockProvider",
    "RecordingProvider",
    "ExponentialBackoffRetry",
    "NoRetry",
    "RateLimitAwareRetry",
    "AdaptiveModelRouter",
    "PassthroughRouter",
]
