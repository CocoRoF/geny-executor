"""Stage 6: API — OpenAI artifact.

Reuses the default APIStage with OpenAIProvider as the default provider.

Usage::

    from geny_executor.core.artifact import create_stage

    stage = create_stage("s06_api", artifact="openai", api_key="sk-...")

Or via PipelineBuilder::

    PipelineBuilder("agent", api_key="sk-...", model="gpt-4.1")
        .with_artifact("s06_api", "openai")
        .build()
"""

from geny_executor.stages.s06_api.artifact.default.stage import APIStage
from geny_executor.stages.s06_api.artifact.openai.providers import OpenAIProvider


def _make_stage(**kwargs):
    """Factory that defaults to OpenAIProvider."""
    api_key = kwargs.pop("api_key", "")
    base_url = kwargs.pop("base_url", None)
    if "provider" not in kwargs:
        if not api_key:
            raise ValueError("OpenAI artifact requires 'api_key'")
        kwargs["provider"] = OpenAIProvider(api_key=api_key, base_url=base_url)
    return APIStage(**kwargs)


Stage = _make_stage

__all__ = ["Stage", "OpenAIProvider"]
