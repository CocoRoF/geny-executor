"""Stage 6: API — Google Gemini artifact.

Reuses the default APIStage with GoogleProvider as the default provider.

Usage::

    from geny_executor.core.artifact import create_stage

    stage = create_stage("s06_api", artifact="google", api_key="AIza...")

Or via PipelineBuilder::

    PipelineBuilder("agent", api_key="AIza...", model="gemini-3-flash")
        .with_artifact("s06_api", "google")
        .build()
"""

from geny_executor.stages.s06_api.artifact.default.stage import APIStage
from geny_executor.stages.s06_api.artifact.google.providers import GoogleProvider


def _make_stage(**kwargs):
    """Factory that defaults to GoogleProvider."""
    api_key = kwargs.pop("api_key", "")
    if "provider" not in kwargs:
        if not api_key:
            raise ValueError("Google artifact requires 'api_key'")
        kwargs["provider"] = GoogleProvider(api_key=api_key)
    return APIStage(**kwargs)


Stage = _make_stage

__all__ = ["Stage", "GoogleProvider"]
